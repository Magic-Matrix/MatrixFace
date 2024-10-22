
from configs.res34_fpn_detection_640x640_cfg import *




for i, data in enumerate(train_dataloader):
    losses = model(data)

    losses["loss"].backward()

    optimizer.step()

    print({k:losses[k].item() for k in losses})

