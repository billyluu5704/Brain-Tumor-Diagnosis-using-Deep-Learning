import torch
from monai.transforms import MapTransform

class ConvertToMultiChannel(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            #merge label 2 and 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            #merge label 1,2,3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 2), d[key] == 3))
            #label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
