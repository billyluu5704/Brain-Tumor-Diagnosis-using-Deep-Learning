import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd,
    Spacingd, NormalizeIntensityd, ResizeWithPadOrCropd,
    Activationsd, AsDiscreted, Invertd, SaveImaged
)

# local imports
from U_Mamba_net import U_Mamba_net
from model_builder import UNet3D
from RepeatChannel import RepeatChannelsd
from monai.data import decollate_batch
from monai.transforms import (
    Compose, Activationsd, AsDiscreted, Invertd, SaveImaged, Lambdad
)


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
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    # strip 'module.' if present
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[warn] Missing keys: {sorted(missing)[:10]} ...")
    if unexpected:
        print(f"[warn] Unexpected keys: {sorted(unexpected)[:10]} ...")


def make_preprocess(roi: Tuple[int, int, int], target_channels: int):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"], track_meta=True),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=roi),
        RepeatChannelsd(keys=["image"], target_channels=target_channels),
        EnsureTyped(keys=["image"], track_meta=True),
    ])


def make_postprocess(pre_tf, outdir: str, activation: str, multilabel: bool, threshold: float):
    act_kwargs = {"sigmoid": activation == "sigmoid", "softmax": activation == "softmax"}

    if multilabel:
        discretize = AsDiscreted(keys=["pred"], threshold=threshold)
    else:
        discretize = AsDiscreted(keys=["pred"], argmax=True)

    return Compose([
        Activationsd(keys=["pred"], **act_kwargs),
        discretize,
        # invert to original space first
        Invertd(
            keys=["pred"], transform=pre_tf, orig_keys=["image"],
            meta_keys=["pred_meta_dict"], orig_meta_keys=["image_meta_dict"],
            meta_key_postfix="meta_dict", nearest_interp=True, to_tensor=True,
        ),
        # squeeze channel: (1, H, W, D) -> (H, W, D)
        Lambdad(keys=["pred"], func=lambda x: x[0] if x.ndim == 4 and x.shape[0] == 1 else x),
        SaveImaged(
            keys=["pred"], output_dir=outdir, output_postfix="pred",
            separate_folder=False, resample=False, print_log=True,
            output_dtype=np.uint8,  # nice for labels
        ),
    ])


def parse_args():
    p = argparse.ArgumentParser(description="3D brain tumor inference")
    p.add_argument("--model", default="u_mamba", help="u_mamba or unet3d")
    p.add_argument("--weights", required=True, help="Path to .pth checkpoint")
    p.add_argument("--inputs", required=True, nargs="+", help="Glob(s) or path(s) to .nii/.nii.gz")
    p.add_argument("--outdir", default="predictions", help="Output dir for NIfTI")
    p.add_argument("--roi", type=int, nargs=3, default=[128, 128, 64], help="Sliding-window ROI size")
    p.add_argument("--sw-batch", type=int, default=1, help="Sliding window batch size")
    p.add_argument("--overlap", type=float, default=0.3, help="Sliding window overlap [0-1]")
    p.add_argument("--channels", type=int, default=4, help="Model input channels")
    p.add_argument("--num-classes", type=int, default=3, help="Output channels/classes")
    p.add_argument("--activation", choices=["sigmoid", "softmax"], default="sigmoid")
    p.add_argument("--multilabel", action="store_true", help="Treat outputs as independent classes")
    p.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for multilabel")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision")
    return p.parse_args()


def resolve_inputs(patterns: List[str]) -> List[str]:
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No files matched: {patterns}")
    return files


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, in_channels=args.channels, num_classes=args.num_classes, device=device)
    load_weights(model, args.weights)
    model.eval()

    # dataset
    pre_tf = make_preprocess(tuple(args.roi), args.channels)
    files = resolve_inputs(args.inputs)
    data = [{"image": f, "image_name": os.path.basename(f)} for f in files]
    ds = Dataset(data=data, transform=pre_tf)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    post_tf = make_postprocess(pre_tf, args.outdir, args.activation, args.multilabel, args.threshold)

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)

            # sliding-window predictor must use the same model
            def _fwd(inp):
                res = model(inp)
                return res[0] if isinstance(res, (list, tuple)) else res

            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = sliding_window_inference(
                    inputs=x,
                    roi_size=tuple(args.roi),
                    sw_batch_size=args.sw_batch,
                    predictor=_fwd,
                    overlap=args.overlap,
                    mode="gaussian",
                )

            # decollate the batch into list of dicts (here it's 1 item, but still safer)
            batch_list = decollate_batch(batch)
            pred_list = decollate_batch({"pred": logits.cpu()})  # makes pred_list[i]["pred"]

            for b, p in zip(batch_list, pred_list):
                d = {
                    "image": b["image"],  # MetaTensor with correct meta
                    "pred": p["pred"],    # tensor
                    # use the single meta dict, not the list
                    "pred_meta_dict": b.get("image_meta_dict"),
                }
                d = post_tf(d)  # this will SaveImaged

                image_name = b.get("image_name", "image")
                print(f"[OK] Saved prediction for {image_name} -> {args.outdir}")


if __name__ == "__main__":
    main()

""" 
tO RUN:
CUDA_VISIBLE_DEVICES=3 python test_predict.py \
  --model u_mamba \
  --weights /home/luudh/luudh/MyFile/medical_image_lab/monai/going_modular/model/Medical_Image_U_Mamba_Net_ssm_8_3D_add_AUX_W2_W3_pos_16_version2.pth \
  --inputs /home/luudh/luudh/MyFile/medical_image_lab/monai/data/test_data/BraTS-SSA-00009-000-t1c.nii.gz \
  --outdir ./predictions \
  --roi 128 128 64 \
  --activation softmax \
  --channels 4 \
  --num-classes 4

 """