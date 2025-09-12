#contains various utility functions for training and saving model
import torch
from pathlib import Path

def save_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    #create model save path
    assert model_name.endswith('.pth') or model_name.endswith('.pt'), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    #save model to state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, model_save_path)