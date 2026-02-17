import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd,
    Spacingd, NormalizeIntensityd, ResizeWithPadOrCropd,
    AsDiscreted, Activationsd
)
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Activations

# local imports
from architecture.U_Mamba_Net.U_Mamba_net import U_Mamba_net
from architecture.U_Net.model_builder import UNet3D
from preprocess.utils.RepeatChannel import RepeatChannelsd


# ----------------------
#  Helpers
# ----------------------

def build_model(name: str, in_channels: int, num_classes: int, device: torch.device) -> nn.Module:
    name = name.lower()
    if name in {"u_mamba", "mamba", "u-mamba"}:
        model = U_Mamba_net(in_channels=in_channels, num_classes=num_classes)
    elif name in {"unet", "unet3d"}:
        model = UNet3D(in_channels=in_channels, out_channels=num_classes)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from ['u_mamba','unet3d'].")
    return model.to(device)


def load_weights(model: nn.Module, ckpt_path: str) -> None:
    # robust to PyTorch 2.6 weights_only change
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        # older PyTorch versions without weights_only argument
        ckpt = torch.load(ckpt_path, map_location="cpu")

    state = ckpt.get("model_state_dict", ckpt)
    #strip 'module.' if present
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[warn] Missing keys: {sorted(missing)[:10]} ...")
    if unexpected:
        print(f"[warn] Unexpected keys: {sorted(unexpected)[:10]} ...")


def make_preprocess(roi: Tuple[int, int, int], target_channels: int):
    """
    Preprocess for BRATS-like images and labels.
    Assumes labels are integer masks (0..C).
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], track_meta=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Spacingd(keys=["label"], pixdim=(1.0, 1.0, 1.0), mode="nearest"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=roi),
        RepeatChannelsd(keys=["image"], target_channels=target_channels),
        EnsureTyped(keys=["image", "label"], track_meta=True),
    ])


def parse_args():
    p = argparse.ArgumentParser(description="3D brain tumor preliminary evaluation (Dice)")
    p.add_argument("--model", default="u_mamba", help="u_mamba or unet3d")
    p.add_argument("--weights", required=True, help="Path to .pth checkpoint")
    p.add_argument("--images", required=True, nargs="+", help="Glob(s) or path(s) to image .nii/.nii.gz")
    p.add_argument("--labels", required=True, nargs="+", help="Glob(s) or path(s) to label .nii/.nii.gz")
    p.add_argument("--roi", type=int, nargs=3, default=[128, 128, 64], help="Sliding-window ROI size")
    p.add_argument("--sw-batch", type=int, default=1, help="Sliding window batch size")
    p.add_argument("--overlap", type=float, default=0.3, help="Sliding window overlap [0-1]")
    p.add_argument("--channels", type=int, default=4, help="Model input channels")
    p.add_argument("--num-classes", type=int, default=3, help="Output channels/classes")
    p.add_argument("--activation", choices=["sigmoid", "softmax"], default="softmax")
    p.add_argument("--multilabel", action="store_true", help="Treat outputs as independent classes (sigmoid)")
    p.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for multilabel")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision")
    p.add_argument("--out-csv", default="", help="CSV file to store per-case Dice")
    return p.parse_args()


def resolve_paths(patterns: List[str]) -> List[str]:
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No files matched: {patterns}")
    return files


# ----------------------
#  Main eval
# ----------------------

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # model
    model = build_model(args.model, in_channels=args.channels, num_classes=args.num_classes, device=device)
    load_weights(model, args.weights)
    model.eval()

    # data
    img_files = resolve_paths(args.images)
    lbl_files = resolve_paths(args.labels)
    if len(img_files) != len(lbl_files):
        raise RuntimeError(f"#images ({len(img_files)}) != #labels ({len(lbl_files)})")

    print(f"[INFO] Found {len(img_files)} image/label pairs for evaluation.")

    data = [
        {"image": i, "label": l, "case_id": os.path.basename(i)}
        for i, l in zip(img_files, lbl_files)
    ]

    pre_tf = make_preprocess(tuple(args.roi), args.channels)
    ds = Dataset(data=data, transform=pre_tf)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)

    # post-processing for metrics
    if args.multilabel:
        post_pred = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=args.threshold),
        ])
        post_label = AsDiscrete(threshold=0.5)  # assumes labels already 0/1 per channel if multilabel GT
    else:
        # standard multi-class softmax + argmax + one-hot
        post_pred = Compose([
            Activations(softmax=True),
            AsDiscrete(argmax=True, to_onehot=args.num_classes),
        ])
        post_label = AsDiscrete(to_onehot=args.num_classes)

    case_ids = []
    per_case_dice = []  # [N, C]

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            case_id = batch["case_id"][0]
            case_ids.append(case_id)

            # sliding-window inference
            def _fwd(inp):
                out = model(inp)
                return out[0] if isinstance(out, (list, tuple)) else out

            if device.type == "cuda" and args.amp:
                with torch.amp.autocast("cuda"):
                    logits = sliding_window_inference(
                        inputs=images,
                        roi_size=tuple(args.roi),
                        sw_batch_size=args.sw_batch,
                        predictor=_fwd,
                        overlap=args.overlap,
                        mode="gaussian",
                    )
            else:
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=tuple(args.roi),
                    sw_batch_size=args.sw_batch,
                    predictor=_fwd,
                    overlap=args.overlap,
                    mode="gaussian",
                )

            # decollate & post-process
            logits_list = decollate_batch(logits)
            labels_list = decollate_batch(labels)

            preds = [post_pred(p) for p in logits_list]
            labs = [post_label(l) for l in labels_list]

            # update metric
            dice_metric(y_pred=preds, y=labs)
            # get per-case (for this batch size=1)
            per_case = dice_metric.aggregate(reduction="none").cpu().numpy()  # shape [B, C]
            dice_metric.reset()  # reset because we're tracking per-case ourselves

            per_case_dice.append(per_case[0])  # [C]

    per_case_dice = np.stack(per_case_dice, axis=0)  # [N, C]
    mean_per_class = per_case_dice.mean(axis=0)      # [C]
    mean_dice = per_case_dice.mean()

    # print summary
    print("\n=== Preliminary Dice Results ===")
    print(f"Num cases: {len(case_ids)}")
    print(f"Mean Dice (all classes, all cases): {mean_dice:.4f}")
    num_reported_classes = mean_per_class.shape[0]  # this will be 3
    for c in range(num_reported_classes):
        print(f"  Class {c}: mean Dice = {mean_per_class[c]:.4f}")

    # save CSV
    import csv
    # if user didn't specify --out-csv, name it based on the model
    if not args.out_csv:
        args.out_csv = f"prelim_results/{args.model}_prelim_results.csv"

    print(f"\n[INFO] Saving per-case Dice to {args.out_csv}")
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["case_id"] + [f"class_{c}_dice" for c in range(mean_per_class.shape[0])]
        writer.writerow(header)
        for cid, dice_vec in zip(case_ids, per_case_dice):
            writer.writerow([cid] + [float(x) for x in dice_vec])

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
