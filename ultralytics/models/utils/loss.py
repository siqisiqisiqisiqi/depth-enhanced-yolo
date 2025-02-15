# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss, VarifocalLoss, OKSLoss
from ultralytics.utils.metrics import bbox_iou

from .ops import HungarianMatcher


class DETRLoss(nn.Module):
    """
    DETR (DEtection TRansformer) Loss class. This class calculates and returns the different loss components for the
    DETR object detection model. It computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary
    losses.

    Attributes:
        nc (int): The number of classes.
        loss_gain (dict): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary losses.
        use_fl (bool): Use FocalLoss or not.
        use_vfl (bool): Use VarifocalLoss or not.
        use_uni_match (bool): Whether to use a fixed layer to assign labels for the auxiliary branch.
        uni_match_ind (int): The fixed indices of a layer to use if `use_uni_match` is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss or None): Focal Loss object if `use_fl` is True, otherwise None.
        vfl (VarifocalLoss or None): Varifocal Loss object if `use_vfl` is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(
        self, nc=80, loss_gain=None, aux_loss=True, use_fl=True, use_vfl=False, use_uni_match=False, uni_match_ind=0, with_kpts=False,
    ):
        """
        Initialize DETR loss function with customizable components and gains.

        Uses default loss_gain if not provided. Initializes HungarianMatcher with
        preset cost gains. Supports auxiliary losses and various loss types.

        Args:
            nc (int): Number of classes.
            loss_gain (dict): Coefficients for different loss components.
            aux_loss (bool): Use auxiliary losses from each decoder layer.
            use_fl (bool): Use FocalLoss.
            use_vfl (bool): Use VarifocalLoss.
            use_uni_match (bool): Use fixed layer for auxiliary branch label assignment.
            uni_match_ind (int): Index of fixed layer for uni_match.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1,
                         "mask": 1, "dice": 1, "kpts": 1, "oks": 2, "kobj": 2}
            # loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1,
            #              "mask": 1, "dice": 1, "kpts": 1, "oks": 4, "kobj": 2}
        self.nc = nc
        self.matcher = HungarianMatcher(
            cost_gain={"class": 2, "bbox": 5, "giou": 2}, with_kpts=with_kpts)
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        if with_kpts:
            self.oks = OKSLoss(linear=True,
                            num_keypoints=3,
                            eps=1e-6,
                            reduction='mean',
                            loss_weight=1.0)
            self.bce_pose = nn.BCEWithLogitsLoss()
        self.fl = FocalLoss() if use_fl else None
        self.vfl = VarifocalLoss() if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=""):
        """Computes the classification loss based on predictions, target values, and ground truth scores."""
        # Logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
        one_hot = torch.zeros((bs, nq, self.nc + 1),
                              dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(
                pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=""):
        """Computes bounding box and GIoU losses for predicted and ground truth bounding boxes."""
        # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain["bbox"] * \
            F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        loss[name_giou] = 1.0 - \
            bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    def _get_loss_kpts(self, pred_kpts, gt_kpts, tgt_area, postfix=""):
        """Computes bounding box and GIoU losses for predicted and ground truth bounding boxes."""
        # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_kpts = f"loss_kpts{postfix}"
        name_oks = f"loss_oks{postfix}"

        loss = {}
        if len(gt_kpts) == 0:
            loss[name_kpts] = torch.tensor(0.0, device=self.device)
            loss[name_oks] = torch.tensor(0.0, device=self.device)
            return loss

        # TODO: Right now the dataloader doesn't have the area value
        # tgt_area = torch.full((gt_kpts.shape[0],), 0.3).cuda()
        Z_pred_x = pred_kpts[:, 0::3]
        Z_pred_y = pred_kpts[:, 1::3]
        V_pred = pred_kpts[:, 2::3]
        Z_pred = torch.cat((Z_pred_x.unsqueeze(-1), 
                            Z_pred_y.unsqueeze(-1)), dim=-1).flatten(start_dim=1)

        Z_gt_x = gt_kpts[:, 0::3]
        Z_gt_y = gt_kpts[:, 1::3]
        V_gt = gt_kpts[:, 2::3]
        Z_gt = torch.cat((Z_gt_x.unsqueeze(-1), 
                          Z_gt_y.unsqueeze(-1)), dim=-1).flatten(start_dim=1)

        pose_loss = F.l1_loss(Z_pred, Z_gt, reduction='none')
        pose_loss = pose_loss * V_gt.repeat_interleave(2, dim=1)

        kpts_obj_loss = self.loss_gain['kobj'] * self.bce_pose(V_pred, V_gt/2)
        loss[name_kpts] = self.loss_gain['kpts'] * pose_loss.sum() / \
            len(gt_kpts) + kpts_obj_loss

        oks_loss = self.oks(Z_pred, Z_gt, V_gt, tgt_area,
                            weight=None, avg_factor=None, reduction_override=None)
        loss[name_oks] = self.loss_gain['oks'] * oks_loss.sum() / len(gt_kpts)
        return {k: v.squeeze() for k, v in loss.items()}

    def _get_loss_aux(
        self,
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        pred_kpts=None,
        gt_kpts=None,
        match_indices=None,
        postfix="",
        masks=None,
        gt_mask=None,
    ):
        """Get auxiliary losses."""
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if pred_kpts is not None else 3,
                           device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                pred_kpts=pred_kpts[self.uni_match_ind] if pred_kpts is not None else None,
                gt_kpts=gt_kpts,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            aux_kpts = pred_kpts[i] if pred_kpts is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                aux_kpts,
                gt_kpts,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            if pred_kpts is not None and gt_kpts is not None:
                loss[3] += loss_[f'loss_kpts{postfix}']
                loss[4] += loss_[f'loss_oks{postfix}']
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']

        loss_dict = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }
        if pred_kpts is not None and gt_kpts is not None:
            loss_dict[f'loss_kpts_aux{postfix}'] = loss[3]
            loss_dict[f'loss_oks_aux{postfix}'] = loss[4]
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss_dict

    @staticmethod
    def _get_index(match_indices):
        """Returns batch indices, source indices, and destination indices from provided match indices."""
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(self, pred_bboxes, gt_bboxes, match_indices):
        """Assigns predicted bounding boxes to ground truth bounding boxes based on the match indices."""
        pred_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros(
                    0, t.shape[-1], device=self.device)
                for t, (i, _) in zip(pred_bboxes, match_indices)
            ]
        )
        gt_assigned = torch.cat(
            [
                t[j] if len(j) > 0 else torch.zeros(
                    0, t.shape[-1], device=self.device)
                for t, (_, j) in zip(gt_bboxes, match_indices)
            ]
        )
        return pred_assigned, gt_assigned

    def _get_loss(
        self,
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        pred_kpts=None,
        gt_kpts=None,
        masks=None,
        gt_mask=None,
        postfix="",
        match_indices=None,
    ):
        """Get losses."""
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                pred_kpts=pred_kpts, gt_kpts=gt_kpts, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]
        if gt_kpts is not None and pred_kpts is not None:
            pred_kpts, gt_kpts = pred_kpts[idx], gt_kpts[gt_idx]
            tgt_area = gt_bboxes[:, 2:].prod(1)

        bs, nq = pred_scores.shape[:2]
        targets = torch.full(
            (bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(
                pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        loss = {}
        loss.update(self._get_loss_class(pred_scores, targets,
                    gt_scores, len(gt_bboxes), postfix))
        loss.update(self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix))

        if pred_kpts is not None and gt_kpts is not None:
            loss.update(self._get_loss_kpts(pred_kpts, gt_kpts, tgt_area, postfix))
        return loss

    @staticmethod
    def get_dn_match_indices(dn_pos_idx, dn_num_group, gt_groups):
        """
        Get the match indices for denoising.

        Args:
            dn_pos_idx (List[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (List[int]): List of integers representing the number of ground truths for each image.

        Returns:
            (List[tuple]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(
                    end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(
                    gt_idx), "Expected the same length, "
                f"but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices

    def forward(self, pred_bboxes, pred_scores, batch, pred_kpts=None, postfix="", **kwargs):
        """
        Calculate loss for predicted bounding boxes and scores.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape [l, b, query, 4].
            pred_scores (torch.Tensor): Predicted class scores, shape [l, b, query, num_classes].
            batch (dict): Batch information containing:
                cls (torch.Tensor): Ground truth classes, shape [num_gts].
                bboxes (torch.Tensor): Ground truth bounding boxes, shape [num_gts, 4].
                gt_groups (List[int]): Number of ground truths for each image in the batch.
            postfix (str): Postfix for loss names.
            **kwargs (Any): Additional arguments, may include 'match_indices'.

        Returns:
            (dict): Computed losses, including main and auxiliary (if enabled).

        Note:
            Uses last elements of pred_bboxes and pred_scores for main loss, and the rest for auxiliary losses if
            self.aux_loss is True.
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get("match_indices", None)

        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
        if pred_kpts is not None:
            gt_kpts = batch["keypoints"]
            gt_kpts = gt_kpts.view(gt_kpts.shape[0],-1)
            total_loss = self._get_loss(
                pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups,
                pred_kpts[-1], gt_kpts, postfix=postfix, match_indices=match_indices
            )
        else:
            gt_kpts = None
            total_loss = self._get_loss(
                pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups,
                None, None, postfix=postfix, match_indices=match_indices
            )

        if self.aux_loss:
            if pred_kpts is not None:
                total_loss.update(
                    self._get_loss_aux(
                        pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls,
                        gt_groups, pred_kpts[:-1], gt_kpts, match_indices, postfix
                    )
                )
            else:
                total_loss.update(
                    self._get_loss_aux(
                        pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls,
                        gt_groups, None, None, match_indices, postfix
                    )
                )

        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    """
    Real-Time DeepTracker (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        """
        Forward pass to compute the detection loss.

        Args:
            preds (tuple): Predicted bounding boxes and scores.
            batch (dict): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes. Default is None.
            dn_scores (torch.Tensor, optional): Denoising scores. Default is None.
            dn_meta (dict, optional): Metadata for denoising. Default is None.

        Returns:
            (dict): Dictionary containing the total loss and, if applicable, the denoising loss.
        """
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(
                dn_pos_idx, dn_num_group, batch["gt_groups"])

            # Compute the denoising training loss
            dn_loss = super().forward(dn_bboxes, dn_scores, batch,
                                      postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(
                0.0, device=self.device) for k in total_loss.keys()})

        return total_loss


class RTDETRPoseLoss(DETRLoss):
    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_kpts=None, dn_meta=None):
        """
        Forward pass to compute the detection loss.

        Args:
            preds (tuple): Predicted bounding boxes and scores.
            batch (dict): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes. Default is None.
            dn_scores (torch.Tensor, optional): Denoising scores. Default is None.
            dn_meta (dict, optional): Metadata for denoising. Default is None.

        Returns:
            (dict): Dictionary containing the total loss and, if applicable, the denoising loss.
        """
        pred_bboxes, pred_scores, pred_kpts = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch, pred_kpts)

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(
                dn_pos_idx, dn_num_group, batch["gt_groups"])

            # Compute the denoising training loss
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, dn_kpts,
                                      postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(
                0.0, device=self.device) for k in total_loss.keys()})

        return total_loss
