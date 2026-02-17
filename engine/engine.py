
import torch
import os
import matplotlib.pyplot as plt
import utils.utils as utils
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.amp import GradScaler, autocast
from torchmetrics import Dice
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Activations, Compose
from monai.data import DataLoader, decollate_batch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from train.train import MODEL_PATH
from pathlib import Path

# use amp to accelerate training
scaler = torch.amp.GradScaler('cuda')
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# multi-label eval; we have 3 foreground channels, no explicit background
epoch_dice_metric = DiceMetric(include_background=True, reduction="mean")
post_trans = Compose([
    Activations(softmax=True),
    AsDiscrete(argmax=True),
])

#MODEL_PATH = r"model/Medical_Image_UNet3D.pth"
#MODEL_PATH = r"/home/luudh/luudh/MyFile/medical_image_lab/monai/going_modular/model/Medical_Image_U_Mamba_Net_ssm_16_3D.pth"
#MODEL_PATH = r"/home/luudh/luudh/MyFile/medical_image_lab/monai/going_modular/model/Medical_Image_U_Mamba_Net_ssm_8_3D_add_AUX_W2_W3_pos_16_version3.pth"

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
    AUX_W2, AUX_W3 = 0.3, 0.2  # weights for deep supervision head

    for batch, data in enumerate(batch_bar):
        X, y = data["image"], data["label"]

        # safety
        if isinstance(X, str) or isinstance(y, str):
            raise TypeError(f"DataLoader returned file paths, not tensors")

        X, y = X.to(device), y.to(device)

        # debug: foreground ratio (still fine even if y is [B,1,D,H,W])
        if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and batch < 3:
            with torch.no_grad():
                pos_ratio = y.float().mean().item()
                print(f"[dbg][train] batch pos-ratio = {pos_ratio:.6f}")

        optimizer.zero_grad(set_to_none=True)

        # forward with AMP
        with autocast(device_type="cuda"):
            out = model(X)

            if isinstance(out, tuple):
                # Expect (main, aux2, aux3)
                main_log, aux2_log, aux3_log = out

                main_size = main_log.shape[2:]
                if aux2_log.shape[2:] != main_size:
                    aux2_log = F.interpolate(aux2_log, size=main_size, mode="trilinear", align_corners=False)
                if aux3_log.shape[2:] != main_size:
                    aux3_log = F.interpolate(aux3_log, size=main_size, mode="trilinear", align_corners=False)

                loss_main = loss_fn(main_log, y)
                loss_aux2 = loss_fn(aux2_log, y)
                loss_aux3 = loss_fn(aux3_log, y)
                loss = loss_main + AUX_W2 * loss_aux2 + AUX_W3 * loss_aux3

                y_pred = main_log
            else:
                y_pred = out
                loss = loss_fn(y_pred, y)

        # debug: look at softmax probs, not sigmoid
        if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and batch < 3:
            with torch.no_grad():
                # y is [B,1,D,H,W]; this will just print that single channel
                pr = y.float().mean(dim=[0, 2, 3, 4])
                probs = torch.softmax(y_pred, dim=1)
                mp = probs.mean(dim=[0, 2, 3, 4])
                print(f"[dbg][train] label pos-ratio per class = {pr.tolist()}, "
                      f"mean prob per class = {mp.tolist()}")

        # backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
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
        prb = torch.softmax(log, dim=1)
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

    USE_SLIDING_TTA = True
    ROI = (128, 128, 64)
    OVERLAP = 0.5

    # post: softmax -> argmax
    post_trans = Compose([
        Activations(softmax=True),
        AsDiscrete(argmax=True),
    ])

    with torch.inference_mode():
        batch_bar = tqdm(dataloader, desc="Evaluating", leave=False, unit="batch")
        for batch, data in enumerate(batch_bar):
            X, y = data["image"], data["label"]

            if isinstance(X, str) or isinstance(y, str):
                raise TypeError("DataLoader returned file paths, not tensors")

            X, y = X.to(device), y.to(device)

            # ---- forward ----
            if USE_SLIDING_TTA:
                probs = predict_full(model, X, roi=ROI, overlap=OVERLAP)  # already softmax + TTA
            else:
                out = model(X)
                if isinstance(out, tuple):
                    out = out[0]
                probs = torch.softmax(out, dim=1)

            # ---- debug ----
            if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and batch < 3:
                pr = y.float().mean(dim=[0, 2, 3, 4])
                mp = probs.mean(dim=[0, 2, 3, 4])
                print(f"[dbg][val] label pos-ratio per class = {pr.tolist()}, "
                      f"mean prob per class = {mp.tolist()}")
                print(f"[dbg][val] mean prob={probs.mean().item():.6f}, "
                      f"label pos-ratio={y.float().mean().item():.6f}")

            # ---- to discrete ----
            val_outputs = [post_trans(p) for p in decollate_batch(probs)]
            val_labels  = [i for i in decollate_batch(y)]

            # accumulate epoch dice
            epoch_dice_metric(y_pred=val_outputs, y=val_labels)

            # per-batch dice for progress bar
            _batch_metric = DiceMetric(include_background=True, reduction="mean")
            _batch_metric(y_pred=val_outputs, y=val_labels)
            batch_dice = _batch_metric.aggregate().item()
            _batch_metric.reset()
            batch_bar.set_postfix(Dice=f"{batch_dice:.4f}")

    test_dice = epoch_dice_metric.aggregate().item()
    epoch_dice_metric.reset()
    return test_dice

# plot training stats function
def graphing_stats(train_loss_graph: list, test_dice_graph: list, model_name: str):
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
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{model_name}_train_stat_record.png")
    plt.close()

def train(
    gpu,
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    scheduler=None
) -> Dict[str, List[float]]:
    torch.cuda.set_device(gpu)
    results = {"train_loss": [], "test_dice": []}
    train_loss_graph = []
    test_dice_graph = []

    #extract model name
    model_name = Path(MODEL_PATH).stem

    # helper: am I rank 0?
    def is_main():
        # DDP: rank 0 only
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        # single GPU
        return True

    try:
        for epoch in tqdm(range(epochs)):
            torch.cuda.empty_cache()

            # DDP sampler epoch
            if hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)

            train_loss = train_step(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )

            test_dice = test_step(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device,
            )

            if scheduler is not None:
                scheduler.step(test_dice)

            if is_main():
                print(
                    f"Epoch: {epoch+1}/{epochs}: | "
                    f"train_loss: {train_loss:.4f} | "
                    f"dice_score: {test_dice:.4f}"
                )

            # update history
            results["train_loss"].append(train_loss)
            results["test_dice"].append(test_dice)
            train_loss_graph.append(train_loss)
            test_dice_graph.append(test_dice)

            # save model + stats every epoch (main rank only)
            if is_main():
                utils.save_model(
                    model=model,
                    optimizer=optimizer,
                    target_dir="model",
                    model_name=MODEL_PATH,
                )
                # also dump stats so you can replot later
                torch.save(results, "model/training_stats.pth")
                print(f"[INFO] Model saved to: {MODEL_PATH}")
    except KeyboardInterrupt:
        # user / ssh interrupt
        if is_main():
            print("[INFO] Interrupted, saving last model and stats...")
            # save current model state
            utils.save_model(
                model=model,
                optimizer=optimizer,
                target_dir="model",
                model_name=MODEL_PATH,
            )
            torch.save(results, "model/training_stats.pth")
            # make the plot with what we have
            graphing_stats(train_loss_graph, test_dice_graph, model_name=model_name)
        # re-raise so you still see the interrupt
        raise
    except Exception as e:
        # unexpected error -> save what we can
        if is_main():
            print(f"[ERROR] {e}, saving partial model and stats...")
            utils.save_model(
                model=model,
                optimizer=optimizer,
                target_dir="model",
                model_name=MODEL_PATH,
            )
            torch.save(results, "model/training_stats.pth")
            try:
                graphing_stats(train_loss_graph, test_dice_graph, model_name=model_name)
            except Exception:
                pass
        raise
    return results, train_loss_graph, test_dice_graph

