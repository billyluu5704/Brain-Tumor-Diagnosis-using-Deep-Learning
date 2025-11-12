import torch
from monai.transforms import MapTransform

class ConvertToMultiChannel(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            lbl = d[key]
            wt = (lbl == 1) | (lbl == 2) | (lbl == 3)   # whole tumor
            tc = (lbl == 2) | (lbl == 3)                # tumor core
            et = (lbl == 3)                             # enhancing tumor
            d[key] = torch.stack(
                [wt, tc, et], dim=0
            ).float()
        return d
