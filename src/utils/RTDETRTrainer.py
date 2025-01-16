from copy import copy

import torch

from .trainer import Trainer
from ultralytics.utils import RANK, colorstr
from src.data.rtdetr_dataset import RTDETRDataset
from src.val import RTDETRValidator, RTDETRPoseValidator
from ultralytics.utils.plotting import plot_images


class RTDETRTrainer(Trainer):
    def build_dataset(self, img_path, mode="val", batch=None):
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def get_validator(self):
        """
        Returns a DetectionValidator suitable for RT-DETR model validation.

        Returns:
            (RTDETRValidator): Validator object for model validation.
        """
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        """
        Preprocess a batch of images. Scales and converts the images to float format.

        Args:
            batch (dict): Dictionary containing a batch of images, bboxes, and labels.

        Returns:
            (dict): Preprocessed batch.
        """
        batch = super().preprocess_batch(batch)
        bs = len(batch["img"])
        batch_idx = batch["batch_idx"]
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(
                batch["bboxes"][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch["cls"][batch_idx == i].to(
                device=batch_idx.device, dtype=torch.long))
        return batch

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )


class RTDETRPoseTrainer(Trainer):
    def build_dataset(self, img_path, mode="val", batch=None):
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
            task = self.task
        )

    def get_validator(self):
        """
        Returns a DetectionValidator suitable for RT-DETR model validation.

        Returns:
            (RTDETRValidator): Validator object for model validation.
        """
        self.loss_names = "giou_loss", "cls_loss", "l1_loss", "kpts_loss", "oks_loss"
        return RTDETRPoseValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        """
        Preprocess a batch of images. Scales and converts the images to float format.

        Args:
            batch (dict): Dictionary containing a batch of images, bboxes, and labels.

        Returns:
            (dict): Preprocessed batch.
        """
        batch = super().preprocess_batch(batch)
        bs = len(batch["img"])
        batch_idx = batch["batch_idx"]
        gt_bbox, gt_class, gt_kpts = [], [], []
        for i in range(bs):
            gt_bbox.append(
                batch["bboxes"][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch["cls"][batch_idx == i].to(
                device=batch_idx.device, dtype=torch.long))
            gt_kpts.append(batch["keypoints"][batch_idx == i].to(batch_idx.device))
        return batch

