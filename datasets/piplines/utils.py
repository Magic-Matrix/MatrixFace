

from . import BasePipeline
from PIL import Image
import torch
import numpy as np

class ImageLoad(BasePipeline):
    def __call__(self, results: dict) -> dict:
        img_path = results["img_path"]
        img = Image.open(img_path).convert('RGB')
        results["img"] = img
        return results

class ToTensor(BasePipeline):
    def __init__(self, keys=[]):
        self.keys = keys
        self.dtype = np.float32
        self.to_tensor = lambda m : torch.from_numpy(m.astype(self.dtype))
        self.unsqueeze = lambda m : torch.unsqueeze(m, 0) if len(m.shape) == 2 else m
    def __call__(self, results: dict) -> dict:
        for k in self.keys:
            data = results[k]
            data = self.to_tensor(data)
            # data = self.unsqueeze(data)
            results[k] = data
        return results
    def __repr__(self):
        text = f"{self.__class__.__name__}(keys={self.keys})"
        return text

class ReMoveItems(BasePipeline):
    def __init__(self, keys=[]):
        self.keys = keys

    def __call__(self, results: dict) -> dict:
        for k in self.keys:
            results.pop(k)
        return results
    
