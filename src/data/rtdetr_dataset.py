# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from .dataset import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
# from ultralytics.models.yolo.detect import DetectionValidator
# from ultralytics.utils import colorstr, ops

# __all__ = ("RTDETRValidator",)  # tuple or list


class RTDETRDataset(YOLODataset):
    """
    Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initialize the RTDETRDataset class by inheriting from the YOLODataset class."""
        super().__init__(*args, data=data, task=task, **kwargs)

    # NOTE: add stretch version load_image for RTDETR mosaic
    def load_image(self, i, rect_mode=False):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """Temporary, only for evaluation."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scaleFill=True)])
            transforms = Compose([])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms



