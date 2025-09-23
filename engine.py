
import torch
import matplotlib.pyplot as plt
import utils
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.amp import GradScaler, autocast
from torchmetrics import Dice
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Activations, Compose
from monai.data import DataLoader, decollate_batch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

# use amp to accelerate training
scaler = torch.amp.GradScaler('cuda')
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# multi-label eval; we have 3 foreground channels, no explicit background
epoch_dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
post_trans = Compose([
    Activations(sigmoid=True),
    AsDiscrete(threshold=0.5),
])

#MODEL_PATH = r"model/Medical_Image_UNet3D.pth"
#MODEL_PATH = r"/home/luudh/luudh/MyFile/medical_image_lab/monai/going_modular/model/Medical_Image_U_Mamba_Net_ssm_16_3D.pth"
MODEL_PATH = r"/home/luudh/luudh/MyFile/medical_image_lab/monai/going_modular/model/Medical_Image_U_Mamba_Net_ssm_8_3D_add_learnable_weight.pth"

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    model.train()
    train_loss = 0.0
    step = 0
    batch_bar = tqdm(dataloader, desc="Training", leave=False, unit="batch")
    AUX_W2, AUX_W3 = 0.3, 0.2 #weights for deep supervision head

    for batch, data in enumerate(batch_bar):
        X, y = data["image"], data["label"]

        # Guard against path strings
        if isinstance(X, str) or isinstance(y, str):
            raise TypeError(f"DataLoader is returning file paths instead of tensors: X={X}, y={y}")

        X, y = X.to(device), y.to(device)

        # Quick overall foreground ratio (all classes combined)
        if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and batch < 3:
            with torch.no_grad():
                pos_ratio = y.float().mean().item()
                print(f"[dbg][train] batch pos-ratio = {pos_ratio:.6f}")

        optimizer.zero_grad(set_to_none=True)

        # Forward + loss under AMP
        with autocast(device_type="cuda"):
            out = model(X)
            
            if isinstance(out, tuple):
                # Expect (main, aux2, aux3)
                main_log, aux2_log, aux3_log = out

                # If aux logits are at different scales, upsample to main size
                main_size = main_log.shape[2:]
                if aux2_log.shape[2:] != main_size:
                    aux2_log = F.interpolate(aux2_log, size=main_size, mode="trilinear", align_corners=False)
                if aux3_log.shape[2:] != main_size:
                    aux3_log = F.interpolate(aux3_log, size=main_size, mode="trilinear", align_corners=False)

                loss_main = loss_fn(main_log, y)
                loss_aux2 = loss_fn(aux2_log, y)
                loss_aux3 = loss_fn(aux3_log, y)
                loss = loss_main + AUX_W2 * loss_aux2 + AUX_W3 * loss_aux3

                y_pred = main_log  # use main head for debug stats
            else:
                y_pred = out
                loss = loss_fn(y_pred, y)

        # Per-class ratios/probs (first few batches on rank 0)
        if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and batch < 3:
            with torch.no_grad():
                pr = y.float().mean(dim=[0, 2, 3, 4])                 # per-class label prevalence
                mp = torch.sigmoid(y_pred).mean(dim=[0, 2, 3, 4])     # per-class mean prob
                print(f"[dbg][train] label pos-ratio per class = {pr.tolist()}, "
                      f"mean prob per class = {mp.tolist()}")

        # Backward + step with GradScaler and gradient clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # unscale before clipping when using GradScaler
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        step += 1
        batch_bar.set_postfix(Loss=f"{loss.item():.4f}")

    train_loss /= max(step, 1)
    return train_loss

def predict_full(model, X, roi=(128, 128, 64), overlap=0.5):
    def _forward(inp):
        with autocast(device_type="cuda", enabled=True):
            out = model(inp)
            if isinstance(out, tuple):
                out = out[0]  # main head only
            return out
    preds = []
    for dims in [(), (2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]:
        xa = torch.flip(X, dims=list(dims)) if dims else X
        log = sliding_window_inference(
            xa, roi, sw_batch_size=1, predictor=_forward, overlap=overlap, mode="gaussian"
        )
        prb = torch.sigmoid(log)
        prb = torch.flip(prb, dims=list(dims)) if dims else prb
        preds.append(prb)
    return torch.stack(preds, 0).mean(0)  # [B, C, H, W, D]


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> float:
    model.eval()
    epoch_dice_metric.reset()

    # Inference settings
    USE_SLIDING_TTA = True              # set False to use a single forward pass
    ROI = (128, 128, 64)                # match your TRAIN_ROI (or a larger eval ROI)
    OVERLAP = 0.5
    binarize = AsDiscrete(threshold=0.5)

    with torch.inference_mode():
        batch_bar = tqdm(dataloader, desc="Evaluating", leave=False, unit="batch")
        for batch, data in enumerate(batch_bar):
            X, y = data["image"], data["label"]
            if isinstance(X, str) or isinstance(y, str):
                raise TypeError(f"DataLoader is returning file paths instead of tensors: X={X}, y={y}")

            X, y = X.to(device), y.to(device)

            # ---- Get probabilities (B, C, H, W, D) ----
            if USE_SLIDING_TTA:
                probs = predict_full(model, X, roi=ROI, overlap=OVERLAP)
            else:
                # fast path: single forward on the batch
                out = model(X)
                if isinstance(out, tuple):
                    out = out[0]
                probs = torch.sigmoid(out)

            # ---- Debug prints (rank 0, first few batches) ----
            if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and batch < 3:
                pr = y.float().mean(dim=[0, 2, 3, 4])   # per-class label prevalence
                mp = probs.mean(dim=[0, 2, 3, 4])       # per-class mean prob
                print(f"[dbg][val]   label pos-ratio per class = {pr.tolist()}, "
                      f"mean prob per class = {mp.tolist()}")
                print(f"[dbg][val] mean prob={probs.mean().item():.6f}, "
                      f"label pos-ratio={y.float().mean().item():.6f}")

            # ---- Threshold to binaries for Dice ----
            val_outputs = [binarize(p) for p in decollate_batch(probs)]  # already probs → just binarize
            val_labels  = [i for i in decollate_batch(y)]

            # Per-class Dice (debug)
            pc = DiceMetric(include_background=False, reduction="none", ignore_empty=True)
            pc(y_pred=val_outputs, y=val_labels)
            per_cls = [float(x) for x in pc.aggregate().cpu().flatten()]
            pc.reset()
            if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and batch < 3:
                print("[dbg][val] per-class Dice:", per_cls)

            # Accumulate epoch Dice (mean over classes)
            epoch_dice_metric(y_pred=val_outputs, y=val_labels)

            # Per-batch Dice for the progress bar
            _batch_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
            _batch_metric(y_pred=val_outputs, y=val_labels)
            batch_dice = _batch_metric.aggregate().item()
            _batch_metric.reset()
            batch_bar.set_postfix(Dice=f"{batch_dice:.4f}")

    test_dice = epoch_dice_metric.aggregate().item()
    epoch_dice_metric.reset()
    return test_dice


def train(gpu, model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module, epochs: int, device: torch.device, scheduler=None) -> Dict[str, List[float]]:
        torch.cuda.set_device(gpu)
        results = {'train_loss': [], 'test_dice': []}
        train_loss_graph = []
        test_dice_graph = []
        best = -1.0

        for epoch in tqdm(range(epochs)):
            # Clear unused GPU memory before each epoch
            torch.cuda.empty_cache()
            
            if hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)

            train_loss = train_step(model=model, dataloader=train_dataloader,
            loss_fn=loss_fn, optimizer=optimizer, device=device)

            test_dice = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)

            #call scheduler with test_dice
            if scheduler is not None:
                scheduler.step(test_dice)
            
            print(
                f"Epoch: {epoch+1}/{epochs}: | "
                f"train_loss: {train_loss:.4f} | "
                f"dice_score: {test_dice:.4f}"
            )
            
            results["train_loss"].append(train_loss)
            results["test_dice"].append(test_dice)
            train_loss_graph.append(train_loss)
            test_dice_graph.append(test_dice)

            # Save the model from GPU 0 only
            if gpu == 0:
                if test_dice > best:
                    best = test_dice
                    utils.save_model(model=model, optimizer=optimizer, target_dir="model", model_name=MODEL_PATH)
                    print(f"[INFO] Model saved to: {MODEL_PATH}")
        return results, train_loss_graph, test_dice_graph

def graphing_stats(train_loss_graph: list, test_dice_graph: list):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(train_loss_graph))]
    y = train_loss_graph
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [i + 1 for i in range(len(test_dice_graph))]
    y = test_dice_graph
    plt.xlabel("epoch")
    plt.plot(x, y, color="green")
    plt.savefig("U_Mamba_train_stat_record.png")
    plt.close()

