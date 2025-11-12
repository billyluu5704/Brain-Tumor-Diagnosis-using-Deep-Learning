import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.data import DataLoader, decollate_batch, Dataset, CacheDataset
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    ResizeWithPadOrCropd,
    EnsureChannelFirstd,
    RandZoomd,
    RandCropByPosNegLabeld,
    MapTransform
)
from monai.utils import set_determinism
import onnxruntime
from tqdm import tqdm
import random
from torch.utils.data.distributed import DistributedSampler
import torch
import time
import engine, model_builder, utils
from tomultichannel import ConvertToMultiChannel
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_determinism(seed=0)

BATCH_SIZE = 2
NUM_WORKERS = 2
BASE_DIR_LINUX = r"/home/luudh/luudh/MyFile/medical_image_lab/monai/data/Task01_BrainTumour"

class MakeUnionLabel(MapTransform):
    """
    Creates a single-channel union mask from your 3-channel multi-label 'label'.
    Keeps original 'label' untouched; writes union into 'label_union'.
    """
    def __init__(self, keys=("label",), out_key="label_union"):
        super().__init__(keys)
        self.out_key = out_key

    def __call__(self, data):
        d = dict(data)
        lab = d["label"]  # shape: [1, D, H, W] or [D, H, W]
        # ensure channel-first
        if lab.ndim == 3:
            lab = lab[None, ...]  # [1, D, H, W]
        union = (lab > 0).astype(lab.dtype)
        d[self.out_key] = union  # [1, D, H, W]
        d["label"] = lab          # keep label single-channel
        return d
    


# keep a single spatial size constant across train/val (matches your train crop)
TRAIN_ROI = (96, 96, 96)  #(128, 128, 64)   # used this in RandSpatialCropd

def preprocess_data(world_size=None, rank=None):
    train_images = sorted(glob(os.path.join(BASE_DIR_LINUX, "imagesTr", "*.nii.gz")))
    val_images   = sorted(glob(os.path.join(BASE_DIR_LINUX, "imagesVal", "*.nii.gz")))
    train_labels = sorted(glob(os.path.join(BASE_DIR_LINUX, "labelsTr", "*.nii.gz")))
    val_labels   = sorted(glob(os.path.join(BASE_DIR_LINUX, "labelsVal", "*.nii.gz")))

    train_data = [{"image": i, "label": l} for i, l in zip(train_images, train_labels)]
    val_data   = [{"image": i, "label": l} for i, l in zip(val_images,   val_labels)]

    train_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),   # image: [4,D,H,W], label: [1,D,H,W]
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        # make binary mask for sampling, keep label as 1 channel with class IDs
        MakeUnionLabel(keys=("label",), out_key="label_union"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label_union",
            spatial_size=TRAIN_ROI,
            pos=16,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.9, max_zoom=1.1),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=TRAIN_ROI),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=TRAIN_ROI),
        EnsureTyped(keys=["image", "label"]),
    ])

    train_ds = CacheDataset(train_data, transform=train_transform, cache_rate=0.1, num_workers=NUM_WORKERS)
    val_ds   = CacheDataset(val_data,   transform=val_transform,   cache_rate=0.1, num_workers=NUM_WORKERS)

    if world_size and world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)
        train_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
        val_loader    = DataLoader(val_ds,   batch_size=BATCH_SIZE, sampler=val_sampler,   num_workers=NUM_WORKERS)
    else:
        train_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
        val_loader    = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # quick sanity print
    batch = next(iter(train_loader))
    print("image:", batch["image"].shape)  # [B, 4, 96, 96, 96]
    print("label:", batch["label"].shape)  # [B, 1, 96, 96, 96]
    print("unique labels:", torch.unique(batch["label"]))

    return train_loader, val_loader, val_ds

def show_image(data):
    val_data_example = data
    print(f"image shape: {val_data_example['image'].shape}")
    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.title(f"image channel {i}")
        plt.imshow(val_data_example["image"][i, :, :, 60].detach().cpu(), cmap="gray")
    plt.show()
    print(f"label shape: {val_data_example['label'].shape}")
    plt.figure("label", (24, 6))
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.title(f"label channel {i}")
        plt.imshow(val_data_example["label"][i, :, :, 60].detach().cpu())
    plt.show()