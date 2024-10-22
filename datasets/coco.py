import os
import numpy as np
from .base import BaseDataset
from pycocotools.coco import COCO

class CocoDataset(BaseDataset):

    def __init__(self, json_path, piplines=None):
        self.root = os.path.dirname(os.path.dirname(json_path))
        self.coco = COCO(json_path)
        self.category_id = self.coco.getCatIds('person')[0]
        imgIds = self.coco.catToImgs[self.category_id]
        super().__init__(imgIds, piplines)

    
    def get_one_data(self, idx):
        img_id = super().get_one_data(idx)
        imgInfo = self.coco.loadImgs(img_id)[0]
        annIds = self.coco.getAnnIds(imgIds=imgInfo['id'])
        anns = self.coco.loadAnns(annIds)
        anns = [ann for ann in anns if ann['category_id'] == self.category_id]

        boxes = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

        coco_url = imgInfo['coco_url']
        coco_url = coco_url.split('/')[-2:]
        img_path = os.path.join(self.root, *coco_url)

        outputs = {
            "img_path": img_path,
            "boxes": np.array(boxes),
        }
        return outputs




