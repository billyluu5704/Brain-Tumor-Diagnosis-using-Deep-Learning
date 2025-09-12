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
NUM_WORKERS = 1
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
        lab = d["label"]  # [3, H, W, D], 0/1
        union = (lab.sum(0, keepdim=True) > 0).to(lab.dtype)  # [1, H, W, D]
        d[self.out_key] = union
        return d
    


# keep a single spatial size constant across train/val (matches your train crop)
TRAIN_ROI = (128, 128, 64)   # you used this in RandSpatialCropd

def preprocess_data(world_size=None, rank=None):
    train_images = sorted(glob(os.path.join(BASE_DIR_LINUX, "imagesTr", "*.nii.gz")))
    val_images = sorted(glob(os.path.join(BASE_DIR_LINUX, "imagesVal", "*.nii.gz")))
    train_labels = sorted(glob(os.path.join(BASE_DIR_LINUX, "labelsTr", "*.nii.gz")))
    val_labels = sorted(glob(os.path.join(BASE_DIR_LINUX, "labelsVal", "*.nii.gz")))

    train_data = [{"image": img, "label": lb} for img, lb in zip(train_images, train_labels)]
    val_data = [{"image": img, "label": lb} for img, lb in zip(val_images, val_labels)]
    train_transform = Compose(
        [
            #Load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),  #ensure images has channels in 1st dimension (from [h,w,d,c] to [c,h,w,d])
            EnsureTyped(keys=["image", "label"]), #convert to Pytorch tensors
            ConvertToMultiChannel(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"), #reorients images and labels to RAS coordinate system
            MakeUnionLabel(keys=("label",), out_key="label_union"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            #RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
            #RandSpatialCropd(keys=["image", "label"], roi_size=TRAIN_ROI, random_size=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label_union",
                spatial_size=TRAIN_ROI,
                pos=4,           # ratio of positive samples
                neg=1,           # ratio of negative samples
                num_samples=2,   # you can increase to 2–4 for more diversity
                image_key="image",
                image_threshold=0
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2), #flip along all 3 axes
            RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.9, max_zoom=1.1),  # Zoom in or out randomly
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            #ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(240, 240, 144)),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(128, 128, 64)),
            EnsureTyped(keys=["image", "label"])
        ]
    )

    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannel(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=TRAIN_ROI),
            EnsureTyped(keys=["image", "label"])
        ]
    )

    #create datasets
    train_ds = CacheDataset(
        data=train_data,
        transform=train_transform,
        cache_rate=0.1,
        num_workers=NUM_WORKERS
    )
    val_ds = CacheDataset(
        data=val_data,
        transform=val_transform,
        cache_rate=0.1,
        num_workers=NUM_WORKERS
    )

    # Set up distributed samplers
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

        # Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=NUM_WORKERS, pin_memory=False)
    else:
        # For single GPU training, use regular DataLoader without sampler
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    
    for batch in train_loader:
        print("Batch structure:", batch.keys())  # Should contain 'image' and 'label'
        print("Image shape:", batch["image"].shape)
        print("Label shape:", batch["label"].shape)
        print("Image type:", type(batch["image"]))  # Should be a tensor
        print("Label type:", type(batch["label"]))  # Should be a tensor
        break
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