import torch
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    __version__,
)
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import (
    torch_distributed_zero_first,
)
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER, clean_url, emojis
from src.data.build import build_yolo_dataset, build_dataloader
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class Trainer:
    def __init__(self, model, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.model = model
        self.args = get_cfg(cfg, overrides)
        self.batch_size = self.args.batch
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

    def _setup_train(self):
        # i.e. device='0' or device='0,1,2,3'
        if isinstance(self.args.device, str) and len(self.args.device):
            world_size = len(self.args.device.split(","))
        # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
        elif isinstance(self.args.device, (tuple, list)):
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0
        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(
            self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
