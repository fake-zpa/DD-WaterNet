# Water Body Segmentation (Minimal Runnable Demo)

This repository provides the implementation of a custom water body segmentation model (**`WaterSegModel` / DD-WaterNet**) and a minimal, runnable demo.

## Repository Structure

```
├── config.py                     # Global config (used by WaterSegModel)
├── baselines
│   └── unet
│       └── model_unet.py          # UNet implementation used by the demo
├── model.py                       # Custom model implementation (WaterSegModel / DD-WaterNet)
├── checkpoints
│   └── unet_gaofen.pth            # Demo checkpoint (UNet)
├── demo
│   ├── images                     # A few sample RGB images
│   └── masks                      # Optional GT masks for reference
├── demo_infer.py                  # One-command inference script
├── requirements.txt               # Minimal dependencies
└── README.md                      # This file
```

## Features

- **One-command inference**: Run inference on `demo/images/` and save results to `demo/outputs/`.
- **Custom model code included**: `model.py` contains `WaterSegModel` (DD-WaterNet).
- **Lightweight baseline**: UNet model for easy installation and reproducibility.
- **Result export**: Saves binary prediction masks (`*_pred.png`) and overlay visualizations (`*_overlay.jpg`).
- **Configurable**: Change `--img-size`, `--threshold`, `--device`, input/output folders.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Demo Inference

```bash
python demo_infer.py
```

Outputs are written to `demo/outputs/`.

### Inference on Your Own Images

```bash
python demo_infer.py --input /path/to/images --output /path/to/outputs
```

### Using the Custom Model (`WaterSegModel`)

The custom model implementation is provided in `model.py` as `WaterSegModel`.

To run inference with `WaterSegModel`, you need to provide your own checkpoint:

```bash
python demo_infer.py \
  --model-module model \
  --model-class WaterSegModel \
  --checkpoint /path/to/your_watersegmodel.pth
```

Note: `WaterSegModel` reads default settings from `config.py` (e.g., backbone selection). If your environment does not support the default backbone, you can update `cfg.backbone` in `config.py`.

## File Descriptions

### `demo_infer.py`

Runs image preprocessing (resize + center-crop), loads the UNet checkpoint, performs inference, and saves prediction masks and overlays.

### `baselines/unet/model_unet.py`

UNet segmentation model definition.

### `model.py`

Custom model implementation containing `WaterSegModel`.

### `config.py`

Defines `cfg` (global configuration) used by `WaterSegModel`.

### `checkpoints/unet_gaofen.pth`

Demo weights used for running the inference example.

## Dataset Sources

This repository does not include full datasets. Please obtain them from the original sources:

- DeepGlobe Land Cover Classification (Water class used for binary masks)
  - Challenge page: https://competitions.codalab.org/competitions/18468

- LoveDA (Water label used for binary masks)
  - Official repository: https://github.com/Junjue-Wang/LoveDA
  - Paper: https://arxiv.org/abs/2110.08733

- Gaofen water body segmentation (GF-2)
  - Data mirror: https://github.com/AICyberTeam/2020Gaofen

## What Will Be Submitted to GitHub

The `.gitignore` is configured to only keep the minimal runnable demo files.

Excluded by default:

- Full datasets and experiment artifacts (`data/`, `results/`)
- Training/evaluation/plotting/preprocessing scripts

## Requirements

- Python 3.10+
- PyTorch
- TorchVision
- Pillow
- NumPy
