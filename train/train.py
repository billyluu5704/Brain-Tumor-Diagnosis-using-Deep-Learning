import os
import shutil
from tabnanny import verbose
import tempfile
import time
import matplotlib.pyplot as plt
from monai.losses import DiceLoss, DiceCELoss
from monai.utils import set_determinism
import onnxruntime
from tqdm import tqdm
import random
import torch
import time
import engine.engine as engine, architecture.U_Net.model_builder as model_builder, utils.utils as utils
from preprocess.utils.tomultichannel import ConvertToMultiChannel
from preprocess.preprocess import preprocess_data, show_image
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from glob import glob
from architecture.U_Mamba_Net.U_Mamba_net import U_Mamba_net
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------Config---------------
NUM_EPOCHS = 200
LEARNING_RATE = 3e-4
BASE_DIR_LINUX = r"/home/luudh/luudh/MyFile/medical_image_lab/monai/data/Task01_BrainTumour"
BASE_DIR_WIN = r"D:/medical image lab/monai/data/Task01_BrainTumour"
#MODEL_PATH = r"/home/luudh/luudh/MyFile/medical_image_lab/monai/going_modular/model/Medical_Image_U_Mamba_Net_ssm_16_3D.pth"
MODEL_PATH = r"/home/luudh/luudh/MyFile/medical_image_lab/monai/going_modular/model/Medical_Image_U_Mamba_Net_ssm_8_3D_add_AUX_W2_W3_pos_16_version3.pth"
#MODEL_PATH = r"model/Medical_Image_UNet3D.pth"

# Reasonable NCCL / CUDA defaults for debugging
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29601")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def init_distributed(local_rank: int = None):
    """
    initialized DDP from either torchrun (env vars present) or mp.spawn (args). 
    Return (rank, world_size, local_rank, device)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank)) if local_rank is None else local_rank
    else:
        #mp.spawn path; assume caller passed local_rank
        assert local_rank is not None, "local_rank must be provided when not using torchrun"
        rank = local_rank
        world_size = torch.cuda.device_count()
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    #increase timeout in case first few steps are heavy
    import datetime as dt
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=dt.timedelta(seconds=120)
    )
    dist.barrier()
    return rank, world_size, local_rank, device

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()

def build_dataloaders(world_size, rank):
    train_loader, val_loader, val_ds = preprocess_data(world_size=world_size, rank=rank)
    return train_loader, val_loader, val_ds

def load_checkpoint_if_any(model, optimizer, device, rank):
    if os.path.exists(MODEL_PATH):
        if rank == 0:
            print(f"[INFO] Loading model from: {MODEL_PATH} ...", flush=True)
        """ ckpt = torch.load(MODEL_PATH, map_location=device)
        # If model wrapped by DDP, access .module
        module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        module.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"]) """
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if rank == 0:
            print("[INFO] Model loaded successfully", flush=True)
    else:
        if rank == 0:
            print("[INFO] Model not found, starting training from scratch ...", flush=True)

# after you create the model (once), set a negative bias so initial p << 0.5
def init_head_bias_to_prior(model, p=(0.03, 0.02, 0.01)):  # per-class priors
    import math, torch.nn as nn, torch
    def logit(q): q = max(min(q, 0.499), 1e-4); return math.log(q/(1-q))
    bias_vals = torch.tensor([logit(pi) for pi in p], dtype=torch.float32)
    for m in model.modules():
        if isinstance(m, nn.Conv3d) and m.out_channels == len(p):
            if m.bias is not None:
                m.bias.data.copy_(bias_vals.to(m.bias.data.device, m.bias.data.dtype))
            break

#multi-label (3-channel) loss: Dice(sigmoid) + BCE-with-logits
dice_loss = DiceLoss(sigmoid=True, reduction="mean", smooth_nr=0, smooth_dr=1e-5, squared_pred=True)
BCE_WEIGHT = 0.7 
POS_WEIGHT = torch.tensor([50.0, 50.0, 50.0])

def focal_tversky_loss(logits, target, alpha=0.7, beta=0.3, gamma=0.75, eps=1e-6):
    p = torch.sigmoid(logits)
    dims = [0,2,3,4]
    tp = (p * target).sum(dims)
    fp = (p * (1 - target)).sum(dims)
    fn = ((1 - p) * target).sum(dims)
    t = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1 - t).pow(gamma).mean()

def multilabel_loss(logits, target):
    # Ensure all aux tensors are on the SAME device/dtype as logits
    pos_weight = POS_WEIGHT.to(device=logits.device, dtype=logits.dtype).view(1, -1, 1, 1, 1)

    # logits: [B, 3, ...], target: [B, 3, ...] in {0,1}
    bce = F.binary_cross_entropy_with_logits(
        logits, target.float(),
        pos_weight=pos_weight #broadcast to spatial dims
    )
    ftv = focal_tversky_loss(logits, target)
    return BCE_WEIGHT * bce + (1.0 - BCE_WEIGHT) * ftv

def main_worker(local_rank=None):
    set_determinism(seed=42)

    #Init DDP
    rank, world_size, local_rank, device = init_distributed(local_rank)
    if rank == 0:
        print(f"[DDP] world_size={world_size}, rank={rank}, local_rank={local_rank}", flush=True)

    #make dataloaders
    train_loader, val_loader, val_ds = build_dataloaders(world_size=world_size, rank=rank)
    if rank == 0:
        print("[INFO] Data preprocessing completed", flush=True)
    #show_image(val_ds[4])
    
    #build model and wrap with DDP
    use_checkpointing = True
    #model = model_builder.UNet3D(in_channels=4, out_channels=3).to(device)
    model = U_Mamba_net(in_channels=4, num_classes=4, use_checkpointing=use_checkpointing).to(device)
    init_head_bias_to_prior(model, p=(0.15, 0.10, 0.05))  # set bias for final layer
    
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    #Optimizer/loss/scheduler
    #loss = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    #loss = multilabel_loss
    loss = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) #optimizer
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=4, min_lr=1e-6) #learning rate scheduler
    #scheduler = CosineAnnealingLR(opt, T_max=max(1, len(train_loader)) * NUM_EPOCHS, eta_min=1e-6)

    # Load model if it exists
    load_checkpoint_if_any(model, opt, device, rank)

    print(device)
    results, train_loss_graph, test_dice_graph = engine.train(
        gpu=local_rank,
        model = model,
        train_dataloader = train_loader,
        test_dataloader = val_loader,
        optimizer = opt,
        loss_fn = loss,
        epochs = NUM_EPOCHS,
        device = device,
        scheduler=scheduler
    )

    #Only rank 0 does any plotting / printing-heavy work
    if rank == 0:
        try:
            engine.graphing_stats(train_loss_graph, test_dice_graph)
        except Exception as e:
            print(f"[WARNING] graphing_stats failed on rank 0: {e}", flush=True)

    #Clean up
    cleanup_distributed()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.cuda.empty_cache()
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        main_worker()
    else:
        world_size = torch.cuda.device_count()
        print(f"Using {world_size} GPUs for DistributedDataParallel training", flush=True)
        mp.spawn(main_worker, nprocs=world_size, args=())

"""
To run:
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=3,4 python train.py
"""
