from .piplines import *
from .coco import CocoDataset
from torch.utils.data import DataLoader


def collate_func(datas_list):
    assert isinstance(datas_list[0], dict)
    outs = {}
    for k in datas_list[0]:
        for datas in datas_list:
            if k not in outs.keys():
                outs[k] = []
            outs[k].append(datas[k])
    for k in outs:
        if isinstance(outs[k][0], torch.Tensor):
            outs[k] = torch.stack(outs[k])
    return outs

def get_dataloader(dataset, bs, nw, shuffle=True, drop_last=True, collate_fn=collate_func) -> DataLoader:
    return DataLoader(dataset, bs, num_workers=nw, collate_fn=collate_fn, shuffle=shuffle, pin_memory=True, drop_last=drop_last)

