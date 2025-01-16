import math
import time
import random
import warnings
from copy import copy
# from pathlib import Path

import torch
import numpy as np
from torch import nn, optim
from torch import distributed as dist

from ultralytics.models import yolo
from ultralytics.utils import (
    RANK,
    TQDM,
    DEFAULT_CFG,
    LOCAL_RANK,
    __version__,
    colorstr,
    yaml_save,
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
    autocast,
    one_cycle,
    EarlyStopping,
    strip_optimizer,
)
from ultralytics.utils.plotting import plot_labels
# from src.utils.trainer_utils import progress_string, label_loss_items, get_memory, clear_memory
from src.utils.trainer_utils import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.plots = {}
        self.model = model
        self.start_epoch = 0
        self.task = model.task
        self.device = torch.device("cuda")
        self.args = get_cfg(cfg, overrides)
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100

        # Dirs
        self.save_dir = get_save_dir(self.args)
        # self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

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

    def _setup_train(self, world_size):
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        # freeze_layer_names = ['.dfl']
        # for k, v in self.model.named_parameters():
        #     # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        #     if any(x in k for x in freeze_layer_names):
        #         LOGGER.info(f"Freezing layer '{k}'")
        #         v.requires_grad = False
        #     # only floating point Tensor can require gradients
        #     elif not v.requires_grad and v.dtype.is_floating_point:
        #         LOGGER.info(
        #             f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
        #             "See ultralytics.engine.trainer for customization of frozen layers."
        #         )
        #         v.requires_grad = True

        # Check AMP
        # TODO: debug this part
        # self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        # if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
        #     self.amp = torch.tensor(check_amp(self.model), device=self.device)
        # self.amp = bool(self.amp)  # as boolean
        self.amp = True
        self.scaler = (
            torch.amp.GradScaler(
                "cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(
            self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(
            self.args.imgsz, stride=gs, floor=gs, max_dim=1)
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

    def _do_train(self, world_size=1):
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb),
                 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' +
            (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )

        # Image mosaic
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        epoch = self.start_epoch
        self.optimizer.zero_grad()
        while True:
            self.epoch = epoch
            with warnings.catch_warnings():
                # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self.model.train()

            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(
                        1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j ==
                                     0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(
                                ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) /
                        (i + 1) if self.tloss is not None else self.loss_items
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (
                            time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            # broadcast 'stop' to all ranks
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break
                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(
                        self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            # (GB) GPU memory util
                            f"{self.get_memory():.3g}G",
                            *(self.tloss if loss_length >
                              1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

            # for loggers
            self.lr = {f"lr/pg{ir}": x["lr"]
                       for ir, x in enumerate(self.optimizer.param_groups)}
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=[
                                     "yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(
                    metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(
                    epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (
                        time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / \
                    (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(
                    self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.clear_memory()

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                # broadcast 'stop' to all ranks
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
        self.clear_memory()

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
        
    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

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

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args
        self.model.kpt_shape = self.data["kpt_shape"]

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch
    
    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness
    
    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"  # update best.pt train_metrics from last.pt
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)