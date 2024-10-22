import torch
import torch.nn as nn
from collections import OrderedDict


class CenterHead(nn.Module):
    def __init__(self, in_channels: int, feature_num: int=3) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(feature_num):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, 5, kernel_size=3, stride=1, padding=1),
            ))


    def forward(self, features: OrderedDict):
        assert len(features) == len(self.layers)
        outputs = {}
        heatmap = []
        size = []
        offset = []
        for i, k in enumerate(features.keys()):
            feature = self.layers[i](features[k])
            hm = feature[:, 0:1].flatten(2).permute(0, 2, 1)
            wh = feature[:, 1:3].flatten(2).permute(0, 2, 1)
            xy = feature[:, 3:5].flatten(2).permute(0, 2, 1)
            heatmap.append(hm)
            size.append(wh)
            offset.append(xy)
        
        outputs["heatmap"] = torch.cat(heatmap, dim=1)
        outputs["size"] = torch.cat(size, dim=1)
        outputs["offset"] = torch.cat(offset, dim=1)

        if not self.training:
            outputs["heatmap"] = torch.sigmoid(outputs["heatmap"])

        return outputs
