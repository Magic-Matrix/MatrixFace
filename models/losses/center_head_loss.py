import torch
import torch.nn as nn

class CenterHeadLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, pred_obj, gt_obj):
        heatmap_loss = self.bce(pred_obj['heatmap'], gt_obj['heatmap'])
        mask = gt_obj["mask"].bool()
        size_loss = self.smooth_l1(pred_obj['size'][mask], gt_obj['size'][mask])
        offset_loss = self.smooth_l1(pred_obj['offset'][mask], gt_obj['offset'][mask])
        losses = {
            "heatmap_loss": heatmap_loss,
            "offset_loss": offset_loss,
            "size_loss": size_loss
        }
        return losses
