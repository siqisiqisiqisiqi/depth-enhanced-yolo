import math
import time
from copy import copy
from pathlib import Path

import torch
import numpy as np
from torch import nn, optim

from ultralytics.models import yolo
from ultralytics.utils import (
    RANK,
    DEFAULT_CFG,
    LOCAL_RANK,
    __version__,
    colorstr,
)
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils.torch_utils import (
    torch_distributed_zero_first,
)
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER, clean_url, emojis
# from ultralytics.utils.checks import check_amp
from ultralytics.utils.checks import check_imgsz
from src.data.build import build_yolo_dataset, build_dataloader
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    de_parallel,
    torch_distributed_zero_first,
    ModelEMA,
    one_cycle,
    EarlyStopping,
)
from ultralytics.utils.plotting import plot_labels


class Trainer:
    def __init__(self, model, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.plots = {}
        self.model = model
        self.start_epoch = 0
        self.device = torch.device("cuda")
        self.args = get_cfg(cfg, overrides)
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100
        self.save_dir = get_save_dir(self.args)
        # avoid auto-downloading dataset multiple times
        with torch_distributed_zero_first(LOCAL_RANK):
            self.trainset, self.testset = self.get_dataset()

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        try:
            if self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    # for validating 'yolo train data=url.zip' usage
                    self.args.data = data["yaml_file"]
        except Exception as e:
            raise RuntimeError(
                emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        self.data = data
        return data["train"], data.get("val") or data.get("test")

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max()
                 if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {
            "train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        # init dataset *.cache only once if DDP
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning(
                "WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=None
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def _setup_train(self, world_size):
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_layer_names = ['.dfl']
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        # TODO: debug this part
        # self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        # if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
        #     self.amp = torch.tensor(check_amp(self.model), device=self.device)
        # self.amp = bool(self.amp)  # as boolean
        self.amp = True
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(
            self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")

        self.test_loader = self.get_dataloader(
            self.testset, batch_size=batch_size * 2, rank=-1, mode="val")

        # Train Validation
        self.validator = self.get_validator()
        metric_keys = self.validator.metrics.keys + \
            self.label_loss_items(prefix="val")
        self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
        self.ema = ModelEMA(self.model)
        if self.args.plots:
            self.plot_training_labels()

        # Optimizer
        # accumulate loss before optimizing
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * \
            self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) /
                               max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(
            patience=self.args.patience), False
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move

        # TODO: callback function for future use
        # self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb),
                 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1

        # self.epoch_time = None
        # self.epoch_time_start = time.time()
        # self.train_time_start = time.time()
        # self.run_callbacks("on_train_start")
        # LOGGER.info(
        #     f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
        #     f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
        #     f"Logging results to {colorstr('bold', self.save_dir)}\n"
        #     f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        # )

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            # cosine 1->hyp['lrf']
            self.lf = one_cycle(1, self.args.lrf, self.epochs)
        else:
            self.lf = lambda x: max(
                1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lf)

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        # normalization layers, i.e. BatchNorm2d()
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            # lr0 fit equation to 6 decimal places
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else (
                "AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW",
                      "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(
                g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(
                g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        # add g0 with weight_decay
        optimizer.add_param_group({"params": g[0], "weight_decay": decay})
        # add g1 (BatchNorm2d weights)
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"]
                               for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"]
                             for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(
        ), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args
        self.model.kpt_shape = self.data["kpt_shape"]
