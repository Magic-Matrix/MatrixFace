
from . import BasePipeline
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms.functional as F


class ResizeAffine(BasePipeline):
    def __init__(self, size=640) -> None:
        super().__init__()
        self.size = [size, size]

    def __call__(self, results: dict) -> dict:

        img = results['img']
        boxes = results['boxes']
        boxes = results['boxes'].reshape(-1, 2)
        boxes = np.concatenate([boxes, np.ones((boxes.shape[0], 1))], axis=1)
        boxes = boxes.T

        mat, mat_inv = self.get_crop_mat(img.size)
        mat_inv = mat_inv.reshape(-1)
        img = img.transform(tuple(self.size), Image.AFFINE, mat_inv)
        results['img'] = img

        boxes = (mat @ boxes).T.reshape(-1, 4)
        results['boxes'] = boxes
        
        return results

    def get_crop_mat(self, size):
        w, h = size
        if size[0] > size[1]:
            ih = self.size[0] / size[0] * h
            pts2 = [[0, 0], [self.size[0], 0], [0, ih]]
        else:
            iw = self.size[1] / size[1] * w
            pts2 = [[0, 0], [iw, 0], [0, self.size[1]]]
        pts2 = np.array(pts2).astype(np.float32)
        pts1 = np.array([[0, 0], [w, 0], [0, h]]).astype(np.float32)
        mat = cv2.getAffineTransform(pts1, pts2)
        mat_inv = cv2.getAffineTransform(pts2, pts1)
        return mat, mat_inv


class TransImageToTensor(BasePipeline):
    def __init__(self, keys=[], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.keys = keys
        self.mean = torch.tensor(mean).reshape(3, 1, 1)
        self.std = torch.tensor(std).reshape(3, 1, 1)
    
    def __call__(self, results: dict) -> dict:
        for k in self.keys:
            img = results[k]
            img = F.to_tensor(img)
            img = self.img_norm(img)
            results[k] = img
        return results

    def img_norm(self, img):
        img /= 255
        img -= self.mean
        img /= self.std
        return img
