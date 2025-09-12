
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

# use amp to accelerate training
scaler = torch.amp.GradScaler('cuda')
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# multi-label eval; we have 3 foreground channels, no explicit background
epoch_dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
post_trans = Compose([Activations(sigmoid=True)])

#MODEL_PATH = r"model/Medical_Image_UNet3D.pth"
#MODEL_PATH = r"/home/luudh/luudh/MyFile/medical_image_lab/monai/going_modular/model/Medical_Image_U_Mamba_Net_ssm_16_3D.pth"
MODEL_PATH = r"/home/luudh/luudh/MyFile/medical_image_lab/monai/going_modular/model/Medical_Image_U_Mamba_Net_ssm_16_3D_add_learnable_weight.pth"

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train() #put model in training mode
    train_loss= 0.0 #initialize loss and dice score to 0
    step = 0
    batch_bar = tqdm(dataloader, desc="Training", leave=False, unit="batch")

    for batch, data in enumerate(batch_bar):
        X, y = data["image"], data["label"]  # Get batch
        
        # If they are still strings, raise an error
        if isinstance(X, str) or isinstance(y, str):
            raise TypeError(f"DataLoader is returning file paths instead of tensors: X={X}, y={y}")
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            pos_ratio = y.float().mean().item()
            print(f"[dbg][train] batch pos-ratio = {pos_ratio:.6f}")
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda'):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            
        train_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_bar.set_postfix(Loss=loss.item())
        step += 1

    train_loss /= step
    return train_loss


def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval() #put model in evaluation mode
    epoch_dice_metric.reset()
    step = 0

    with torch.inference_mode():
        step += 1
        batch_bar = tqdm(dataloader, desc="Evaluating", leave=False, unit="batch")
        for batch, data in enumerate(batch_bar):
            #X for image, y for label
            X, y = data["image"], data["label"]  # Get batch
            # If they are still strings, raise an error
            if isinstance(X, str) or isinstance(y, str):
                raise TypeError(f"DataLoader is returning file paths instead of tensors: X={X}, y={y}")
        
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            with torch.no_grad():
                probs = torch.sigmoid(test_pred_logits)
                print(f"[dbg][val] mean prob={probs.mean().item():.6f}, label pos-ratio={y.float().mean().item():.6f}")
            val_outputs = [post_trans(i) for i in decollate_batch(test_pred_logits)]  # sigmoid->0/1
            val_labels = [i for i in decollate_batch(y)]
            
            #accumulate into epoch metric
            epoch_dice_metric(y_pred=val_outputs, y=val_labels)               # accumulate

            #per batch display using TEMP metric
            _batch_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
            _batch_metric(y_pred=val_outputs, y=val_labels)
            batch_dice = _batch_metric.aggregate().item()
            batch_bar.set_postfix(Dice=f"{batch_dice:.4f}")

    test_dice = epoch_dice_metric.aggregate().item() #get batch dice score
    epoch_dice_metric.reset()
    return test_dice

def train(gpu, model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module, epochs: int, device: torch.device, scheduler=None) -> Dict[str, List[float]]:
        torch.cuda.set_device(gpu)
        results = {'train_loss': [], 'test_dice': []}
        train_loss_graph = []
        test_dice_graph = []

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
                scheduler.step()
            
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

