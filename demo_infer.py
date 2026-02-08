import argparse
import os
import importlib
import inspect
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF


def _build_model(module_name: str, class_name: str, in_channels: int, num_classes: int) -> torch.nn.Module:
    module = importlib.import_module(module_name)
    ModelClass = getattr(module, class_name)

    sig = inspect.signature(ModelClass.__init__)
    kwargs = {}
    if "in_channels" in sig.parameters:
        kwargs["in_channels"] = in_channels
    if "num_classes" in sig.parameters:
        kwargs["num_classes"] = num_classes

    return ModelClass(**kwargs)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    try:
        state = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state_dict = state["model_state"]
    else:
        state_dict = state

    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace("ssm_c4.", "axial_c4.").replace("ssm_c5.", "axial_c5.")
        new_key = new_key.replace(".ssm.", ".axial.")
        new_state[new_key] = v

    model.load_state_dict(new_state, strict=True)


def _resize_and_center_crop(img: Image.Image, img_size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size

    if min(w, h) < img_size:
        scale = img_size / float(min(w, h))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        w, h = img.size

    if w != img_size or h != img_size:
        left = max(0, (w - img_size) // 2)
        top = max(0, (h - img_size) // 2)
        left = min(left, w - img_size)
        top = min(top, h - img_size)
        right = left + img_size
        bottom = top + img_size
        img = img.crop((left, top, right, bottom))

    return img


def _iter_images(input_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)


def _save_mask(mask01: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray((mask01.astype(np.uint8) * 255), mode="L")
    im.save(str(path))


def _save_overlay(img: Image.Image, mask01: np.ndarray, path: Path, alpha: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = img.convert("RGBA")

    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    a = Image.fromarray((mask01.astype(np.uint8) * alpha), mode="L")
    overlay.putalpha(a)

    out = Image.alpha_composite(base, overlay).convert("RGB")
    out.save(str(path))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Run demo inference on a few sample images.")
    parser.add_argument("--input", type=str, default="demo/images", help="Input image folder")
    parser.add_argument("--output", type=str, default="demo/outputs", help="Output folder")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/unet_gaofen.pth", help="Path to .pth checkpoint")
    parser.add_argument("--model-module", type=str, default="baselines.unet.model_unet")
    parser.add_argument("--model-class", type=str, default="UNet")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    checkpoint_path = Path(args.checkpoint)

    if not input_dir.exists():
        raise SystemExit(f"Input folder not found: {input_dir}")
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = _build_model(
        module_name=args.model_module,
        class_name=args.model_class,
        in_channels=3,
        num_classes=1,
    ).to(device)
    _load_checkpoint(model, checkpoint_path, device)
    model.eval()

    img_paths = _iter_images(input_dir)
    if not img_paths:
        raise SystemExit(f"No images found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for p in img_paths:
        img = Image.open(str(p)).convert("RGB")
        proc = _resize_and_center_crop(img, args.img_size)
        x = TF.to_tensor(proc).unsqueeze(0).to(device)

        logits = model(x)
        probs = torch.sigmoid(logits)
        pred = (probs > float(args.threshold)).float()

        mask01 = pred[0, 0].detach().cpu().numpy().astype(np.uint8)

        stem = p.stem
        _save_mask(mask01, output_dir / f"{stem}_pred.png")
        _save_overlay(proc, mask01, output_dir / f"{stem}_overlay.jpg")

        print(f"Saved: {stem}_pred.png, {stem}_overlay.jpg")

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
