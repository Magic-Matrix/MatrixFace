
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import ResNet, FeaturePyramidNetwork, SingleStageDetector, CenterHead, CenterHeadLoss
from datasets import *

model = SingleStageDetector(
    ResNet(34),
    FeaturePyramidNetwork([128, 256, 512], 256), 
    CenterHead(256),
    CenterHeadLoss(),
)

train_dataset = CocoDataset("/media/gpu02/sdd1/DataSet/Detection/COCO2017/annotations/instances_val2017.json", [
    ImageLoad(),
    ResizeAffine(640),
    MakeDetectionTargetWithCenterNet(),
    TransImageToTensor(["img"]),
    ToTensor(['heatmap', 'size', 'offset', 'mask']),
    ReMoveItems(['img_path', 'boxes']),
])

train_dataloader = get_dataloader(train_dataset, 16, 0, shuffle=True, drop_last=True)

optimizer = AdamW(model.parameters(), lr=2e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader), eta_min=1e-6)

