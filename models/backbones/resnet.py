
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from collections import OrderedDict

MODELS_DICT = {
    18: resnet18,
    34: resnet34,
    50: resnet50,
    101: resnet101,
    152: resnet152,
}

class ResNet(nn.Module):
    def __init__(self, deep) -> None:
        super().__init__()
        backbone = MODELS_DICT[deep](progress=True)
        self.layer1 = nn.Sequential(*list(backbone.children())[:5])
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        outputs = OrderedDict()
        # outputs["s4"] = x1
        outputs["s8"] = x2
        outputs["s16"] = x3
        outputs["s32"] = x4
        return outputs
