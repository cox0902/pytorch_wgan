from typing import *

import h5py

import numpy as np

import torch
from torch.utils.data import Dataset

from .transforms import Transform


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = np.split(x, 4, axis=1)
    b = [((x0 + x1) * 0.5), ((y0 + y1) * 0.5),
         (x1 - x0), (y1 - y0)]
    return np.concatenate(b, axis=1)


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = np.split(x, 4, axis=1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h),
         (cx + 0.5 * w), (cy + 0.5 * h)]
    return np.concatenate(b, axis=1)


def make_mask(rect):
    mask = np.random.randn((256, 256), dtype=np.float32)
    mask = -np.abs(mask)
    x0, y0, x1, y1 = np.clip(rect, 0, 255)
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    mask[y0:y1 + 1, x0:x1 + 1] = -mask[y0:y1 + 1, x0:x1 + 1]
    mask = torch.FloatTensor(mask)
    return mask


class ImageCodeDataset(Dataset):

    def __init__(self,
                 image_path: str, 
                 code_path: str, 
                 split: Optional[Any] = None,
                 transforms: Optional[List[Transform]] = None,
                 return_mask: bool = False):
        super().__init__()
        self.image_path = image_path
        self.code_path = code_path
        self.split = split
        self.return_mask = return_mask
        self.transforms = transforms
        if self.transforms is None:
            from .transforms import PresetImage
            self.transforms = [ PresetImage() ]

        self._load()

    def _load(self):
        self.hi = h5py.File(self.image_path, "r")
        self.images = self.hi["images"]
        self.labels = self.hi["labels"] 
        self.rects = self.hi["rects"]
        self.hc = h5py.File(self.code_path, "r")
        self.max_len = self.hc.attrs["max_len"]
        self.codes = self.hc["ivs"]
        self.code_lens = self.hc["les"]
        self.ids = self.hc["ids"]

    def summary(self, header: Optional[str] = None):
        print(f"{header} {len(self):,} @ {self.max_len:,}")

    def __len__(self) -> int:
        if self.split is not None:
            return len(self.split)
        return len(self.codes)
    
    def __idx(self, i: int) -> int:
        if self.split is not None:
            return self.split[i]
        return i
    
    def __getitem__(self, index: int) -> Dict:
        code_idx = self.__idx(index)
        img_idx = code_idx

        item = {
            "image": torch.from_numpy(self.images[img_idx]),
            "code": self.codes[code_idx],
            "code_len": self.code_lens[code_idx]
        }

        # 

        if not self.return_mask:
            ids = self.ids[code_idx]
            ivs = item["code"]
            rects = np.stack((np.zeros_like(ivs, dtype=np.float32), ) * 4, axis=-1)
            for i, (each_id, each_iv) in enumerate(zip(ids, ivs)):
                if each_iv <= 7:
                    continue
                loc = np.where(np.logical_and(
                    self.labels[:, 0] == img_idx,
                    self.labels[:, 1] == each_id
                ))
                assert len(loc[0]) == 1, item["code"]
                rects[i] = self.rects[loc[0]]

            item["rect"] = box_xyxy_to_cxcywh(rects) / item["image"].size(-1)
        else:
            masks = np.zeros((item["code"].shape[0], 256, 256), dtype=np.float32)
            ids = self.ids[code_idx]
            ivs = self.codes[code_idx]
            for i, (each_id, each_iv) in enumerate(zip(ids, ivs)):
                if each_iv <= 7:
                    continue
                loc = np.where(np.logical_and(
                    self.labels[:, 0] == img_idx,
                    self.labels[:, 1] == each_id
                ))
                assert len(loc[0]) == 1, item["code"]
                rect = self.rects[loc[0]]
                
                # mask = Image.new("L", (image.size(1), image.size(2)), 0)
                # mask_draw = ImageDraw.Draw(mask)
                # mask_draw.rectangle(rect, fill=255)

                # masks[i] = torch.FloatTensor(np.asarray(mask) / 255.)
                masks[i] = make_mask(rect)
            item["mask"] = masks

        #

        for transform in self.transforms:
            item = transform(item)

        return item
