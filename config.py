from dataclasses import dataclass


@dataclass
class Config:
    data_dir: str = "data/deepglobe"
    save_dir: str = "results/train/deepglobe/segformer_deepglobe"
    model_dir: str = "results/train/deepglobe/segformer_deepglobe"
    out_dir: str = "results/train/deepglobe/segformer_deepglobe"

    model_module: str = "model"
    model_class: str = "WaterSegModel"
    img_size: int = 224
    in_channels: int = 3
    num_classes: int = 1
    backbone: str = "resnet34"
    pretrained_backbone: bool = True

    use_axial_encoder: bool = False
    use_axial_decoder: bool = False
    use_freq_c3: bool = False

    batch_size: int = 8
    num_epochs: int = 80
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4

    use_random_flip: bool = True
    use_random_rotate: bool = True
    max_rotate_degree: float = 10.0

    use_color_jitter: bool = True
    cj_brightness: float = 0.2
    cj_contrast: float = 0.2
    cj_saturation: float = 0.2
    cj_hue: float = 0.05

    infer_th: float = 0.7
    eval_split: str = "test"
    per_image_split: str = "test"
    per_image_top_k: int = 10
    vis_split: str = "val"
    vis_num_samples: int = 16

    loss_type: str = "bce_tversky"
    tversky_alpha: float = 0.85
    tversky_beta: float = 0.15
    tversky_gamma: float = 1.33

    device: str = "cuda"
    seed: int = 42

    split_train: float = 0.7
    split_val: float = 0.15
    split_test: float = 0.15

    save_best_only: bool = True
    monitor_metric: str = "val_dice"
    monitor_mode: str = "max"


cfg = Config()
