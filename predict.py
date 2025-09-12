import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.losses import DiceLoss
from monai.utils import set_determinism
import onnxruntime
from tqdm import tqdm
import random
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
    RandZoomd
)
from monai.data import DataLoader, CacheDataset
import numpy as np

if 'MASTER_ADDR' not in os.environ:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
if 'MASTER_PORT' not in os.environ:
    os.environ['MASTER_PORT'] = '29500'

import torch
import time
from model_builder import UNet3D
from tomultichannel import ConvertToMultiChannel
from preprocess import preprocess_data, show_image
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from glob import glob
import nibabel as nib
from RepeatChannel import RepeatChannelsd
#from monai.data import NiftiSaver
import numpy as np
from U_Mamba_net import U_Mamba_net
from monai.transforms import AsDiscrete, Activations, Compose, Resize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_determinism(seed=0)

NUM_WORKERS = 1
#MODEL_PATH = r"model/Medical_Image_UNet3D.pth"
#MODEL_PATH = r"/home/luudh/luudh/MyFile/medical_image_lab/monai/going_modular/model/Medical_Image_U_Mamba_Net_ssm_16_3D.pth"
MODEL_PATH = r"/home/luudh/luudh/MyFile/medical_image_lab/monai/going_modular/model/Medical_Image_U_Mamba_Net_ssm_16_3D_add_learnable_weight.pth"
BASE_DIR_LINUX = r"/home/luudh/luudh/MyFile/medical_image_lab/monai/data/test_data/"
#BASE_DIR_LINUX = r"/home/luudh/luudh/MyFile/medical_image_lab/monai/data/Task01_BrainTumour/imagesVal"
torch.cuda.empty_cache()

def preprocess_val():
    #val_images = sorted(glob(os.path.join(BASE_DIR_LINUX, "imagesVal", "*.nii.gz")))
    val_images = sorted(glob(os.path.join(BASE_DIR_LINUX, "*.nii.gz")))
    val_data = [{"image": img} for img in val_images]
    val_transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image"]),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ResizeWithPadOrCropd(keys="image", spatial_size=(240, 240, 144)), #also try (128, 128, 64)
        RepeatChannelsd(keys=["image"], target_channels=4),
        EnsureTyped(keys=["image"]),
    ])
    val_ds = CacheDataset(
        data=val_data,
        transform=val_transform,
        cache_rate=0.1,
        num_workers=NUM_WORKERS
    )
    return val_ds

def predict(image_name):
    #model = UNet3D(in_channels=4, out_channels=3).to(device)
    model = U_Mamba_net(in_channels=4, num_classes=3).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    #searh image in directory
    val_images = sorted(glob(os.path.join(BASE_DIR_LINUX, "*.nii.gz")))
    matched_images = [img for img in val_images if os.path.basename(img) == image_name]
    if not matched_images:
        raise FileNotFoundError(f"No image found with name {image_name}.nii.gz in {BASE_DIR_LINUX}")
    image_path = matched_images[0]
    image_location = val_images.index(image_path)
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    basename = os.path.basename(image_path)
    if basename.endswith(".nii.gz"):
        image_basename = basename.replace(".nii.gz", "")
    else:
        image_basename = os.path.splitext(basename)[0]
    
    #prepare dataset
    val_ds = preprocess_val()
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # If keys are prefixed with "module." (from DDP), remove them:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key.replace("module.", "")] = value
    # Load the state_dict with strict=False to ignore missing or extra keys
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    with torch.no_grad():
        # Select one image to evaluate and visualize the model output
        val_input = val_ds[image_location]["image"].unsqueeze(0).to(device)
        input_image = val_ds[image_location]["image"].shape[1:]
        val_output = model(val_input)
        val_output = post_trans(val_output[0])
        print(f"Input Shape: {input_image}")
        print(f"Val input shape: {val_input.shape}")
        print(f"Val output shape: {val_output.shape}")
        val_output_cpu = val_output.cpu() #move to cpu
        predicted_data = np.array(val_output_cpu, dtype=np.float32)

        # Resize the output to the required dimensions (240, 240, 155)
        input_shape = val_input.shape[2:] #exclude batch & channel in val_input
        print(f"Input shape for resizing: {input_shape}")
        #resize_transform = Resize(spatial_size=input_shape) 
        resize_transform = Resize(spatial_size=(240, 240, 155))
        resized_output = resize_transform(predicted_data)
        print(f"Resized output shape: {resized_output.shape}")
        segmentation_mask = resized_output[0]
        #resized_output_np = resized_output.numpy().astype(np.float32)
        segmentation_mask_np = segmentation_mask.astype(np.float32)

        nii_image = nib.Nifti1Image(segmentation_mask_np, affine=np.eye(4))  # Use identity matrix as affine for simplicity
        filename = f"{image_basename}_predicted.nii.gz"
        save_path = os.path.join("predictions", filename)
        nib.save(nii_image, save_path)
        print(f"[INFO] Nifti file saved as {save_path}")

if __name__ == "__main__":
    image_name = input("Enter the image file name: ")
    if not image_name.endswith(".nii.gz"):
        image_name += ".nii.gz"
    # Check if the image file exists
    while not os.path.isfile(os.path.join(BASE_DIR_LINUX, image_name)):
        print(f"Image file {image_name} not found in {BASE_DIR_LINUX}. Please enter a valid file name.")
        image_name = input("Enter the image file name: ")
        if not image_name.endswith(".nii.gz"):
            image_name += ".nii.gz"
    predict(image_name)