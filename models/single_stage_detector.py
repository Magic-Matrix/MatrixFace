
import torch
import torch.nn as nn

class SingleStageDetector(nn.Module):
    def __init__(self, backbone, neck, head, loss=None) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss
    
    def forward(self, target:dict):

        img = target["img"]
        featuares = self.backbone(img)
        featuares = self.neck(featuares)
        outputs = self.head(featuares)
        
        if self.training:
            losses = self.loss(outputs, target)
            losses["loss"] = sum([v for k, v in losses.items()])
            return losses
        else:
            return outputs





