from typing import *

import numpy as np

import torch
import torchvision.transforms.v2 as T


class Transform:

    def __init__(self):
        pass

    def __call__(self, item):
        return item


class PresetImage(Transform):
    
    def __init__(
            self,
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.transforms = T.Compose([
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
            T.ToPureTensor()
        ])

    def __call__(self, item):
        item["image"] = self.transforms(item["image"])
        return item
    
