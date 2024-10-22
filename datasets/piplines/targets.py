
from . import BasePipeline
import numpy as np
import cv2
import math
from PIL import Image


class MakeDetectionTargetWithCenterNet(BasePipeline):
    def __init__(self, down=[8, 16, 32]) -> None:
        super().__init__()
        self.down = down

    def __call__(self, results: dict) -> dict:
        iw, ih = results["img"].size
        heatmap = []
        mask = []
        size = []
        offset = []
        for d in self.down:
            tw, th = iw // d, ih // d
            hm = np.zeros((th, tw), dtype=np.float32)
            wh = np.zeros((th, tw, 2), dtype=np.float32)
            xy = np.zeros((th, tw, 2), dtype=np.float32)
            ms = np.zeros((th, tw), dtype=np.float32)
            boxes  = results["boxes"] / d
            for box in boxes:
                radius = self.get_gaussian_radius_with_box(box)
                ct = np.array([(box[2] + box[0]) / 2, (box[3] + box[1]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                hm = MakeDetectionTargetWithCenterNet.draw_gaussian(hm, ct_int, radius)

                wh[ct_int[1], ct_int[0]] = float(box[2] - box[0]), float(box[3] - box[1])
                xy[ct_int[1], ct_int[0]] = ct - ct_int
                ms[ct_int[1], ct_int[0]] = 1


            mask.append(ms.reshape(-1))
            heatmap.append(hm.reshape(-1, 1))
            size.append(wh.reshape(-1, 2))
            offset.append(xy.reshape(-1, 2))


        results["mask"] = np.concatenate(mask, axis=0)
        results["heatmap"] = np.concatenate(heatmap, axis=0)
        results["size"] = np.concatenate(size, axis=0)
        results["offset"] = np.concatenate(offset, axis=0)

        return results
    
    def get_gaussian_radius_with_box(self, box):
        w, h = float(box[2] - box[0]), float(box[3] - box[1])
        radius = MakeDetectionTargetWithCenterNet.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        return radius

    
    def draw_gaussian(heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = MakeDetectionTargetWithCenterNet.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def gaussian2D(shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def gaussian_radius(det_size, min_overlap=0.7):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)