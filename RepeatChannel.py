import torch
from monai.transforms import MapTransform
from typing import Dict

class RepeatChannelsd(MapTransform):    
    def __init__(self, keys, target_channels=4):
        super().__init__(keys)
        self.target_channels = target_channels

    def __call__(self, data: Dict):
        d = dict(data)
        for key in self.keys:
            img = d[key]  # img shape: (C, H, W, D)
            current_channels = img.shape[0]

            if current_channels == self.target_channels:
                continue  # already correct
            elif current_channels == 1:
                d[key] = img.repeat(self.target_channels, 1, 1, 1)
            else:
                raise ValueError(f"[{key}] has {current_channels} channels. Expected 1 or {self.target_channels}.")
        return d