import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils import colorstr, ops
from src.data.rtdetr_dataset import RTDETRDataset

__all__ = ("RTDETRValidator",)  # tuple or list


class RTDETRValidator(DetectionValidator):
    """
    RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies Non-maximum suppression for
    post-processing, and updates evaluation metrics accordingly.

    Example:
        ```python
        from ultralytics.models.rtdetr import RTDETRValidator

        args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        validator = RTDETRValidator(args=args)
        validator()
        ```

    Note:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build an RTDETR Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def postprocess(self, preds):
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            # Do not need threshold for evaluation as only got 300 boxes here
            # idx = score > self.args.conf
            pred = torch.cat(
                [bbox, score[..., None], cls[..., None]], dim=-1)  # filter
            # Sort by confidence to correctly get internal metrics
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred  # [idx]

        return outputs

    def _prepare_batch(self, si, batch):
        """Prepares a batch for training or inference by applying transformations."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox)  # target boxes
            bbox[..., [0, 2]] *= ori_shape[1]  # native-space pred
            bbox[..., [1, 3]] *= ori_shape[0]  # native-space pred
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares and returns a batch with transformed bounding boxes and class labels."""
        predn = pred.clone()
        predn[..., [0, 2]] *= pbatch["ori_shape"][1] / \
            self.args.imgsz  # native-space pred
        predn[..., [1, 3]] *= pbatch["ori_shape"][0] / \
            self.args.imgsz  # native-space pred
        return predn.float()
    

class RTDETRPoseValidator(PoseValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
            task = self.task
        )

    def postprocess(self, preds):
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        bs, _, nd = preds[0][0].shape
        nk = self.kpt_shape[0] * self.kpt_shape[1]
        bboxes, scores, keypoints = preds[0][0].split((4, self.nc, nk), dim=-1)
        bboxes *= self.args.imgsz
        keypoints[..., 0::3] *= self.args.imgsz
        keypoints[..., 1::3] *= self.args.imgsz
        
        outputs = [torch.zeros((0, 6 + nk), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            # Do not need threshold for evaluation as only got 300 boxes here
            # idx = score > self.args.conf
            pred = torch.cat(
                [bbox, score[..., None], cls[..., None], keypoints[i]], dim=-1)  # filter
            # Sort by confidence to correctly get internal metrics
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred  # [idx]

        return outputs

    def _prepare_batch(self, si, batch):
        """Prepares a batch for training or inference by applying transformations."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        keypoints = batch["keypoints"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox)  # target boxes
            bbox[..., [0, 2]] *= ori_shape[1]  # native-space pred
            bbox[..., [1, 3]] *= ori_shape[0]  # native-space pred
            keypoints[..., 0] *= ori_shape[1]
            keypoints[..., 1] *= ori_shape[0]
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad, "kpts": keypoints}

    def _prepare_pred(self, pred, pbatch):
        """Prepares and returns a batch with transformed bounding boxes and class labels."""
        predn = pred.clone()
        predn[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # native-space pred
        predn[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # native-space pred
        nk = pbatch["kpts"].shape[1]
        pred_kpts = predn[:, 6:].view(len(predn), nk, -1)
        pred_kpts[..., 0] *= pbatch["ori_shape"][1] / self.args.imgsz
        pred_kpts[..., 1] *= pbatch["ori_shape"][0] / self.args.imgsz
        return predn.float(), pred_kpts.float()