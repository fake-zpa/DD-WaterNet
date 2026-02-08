import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config import cfg


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AxialGatedDWConv2D(nn.Module):
    def __init__(self, channels, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.dw_h = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, padding), groups=channels, bias=False)
        self.dw_v = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(padding, 0), groups=channels, bias=False)
        self.pw_in = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)
        self.pw_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        h = self.dw_h(x)
        v = self.dw_v(x)
        u = h + v

        u = self.norm(u)
        u = self.pw_in(u)
        c, g = torch.chunk(u, 2, dim=1)
        u = torch.tanh(c) * torch.sigmoid(g)
        u = self.pw_out(u)

        return x + u


class FrequencyFilter2D(nn.Module):
    def __init__(self, channels, init_scale: float = 0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * init_scale)

    def forward(self, x):
        b, c, h, w = x.shape
        x_fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        x_fft_enhanced = x_fft * (1.0 + self.scale)
        x_ifft = torch.fft.irfft2(x_fft_enhanced, s=(h, w), dim=(-2, -1), norm="ortho")
        return x_ifft


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_axial: bool = False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvBNReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)
        self.axial = AxialGatedDWConv2D(out_channels) if use_axial else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.axial(x)
        return x


def _get_resnet34(pretrained: bool = True):
    try:
        from torchvision.models import resnet34, ResNet34_Weights

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)
    except Exception:
        backbone = models.resnet34(pretrained=pretrained)
    return backbone


def _get_convnext_tiny(pretrained: bool = True):
    try:
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)
    except Exception as e:
        if hasattr(models, "convnext_tiny"):
            backbone = models.convnext_tiny(pretrained=pretrained)
        else:
            raise RuntimeError(
                "ConvNeXt-Tiny backbone requires a newer torchvision version that provides convnext_tiny. "
                "Please upgrade torchvision or set cfg.backbone='resnet34'."
            ) from e
    return backbone


class WaterSegModel(nn.Module):
    def __init__(self, in_channels: int = None, num_classes: int = None, pretrained_backbone: bool = None):
        super().__init__()

        if in_channels is None:
            in_channels = cfg.in_channels
        if num_classes is None:
            num_classes = cfg.num_classes
        if pretrained_backbone is None:
            pretrained_backbone = cfg.pretrained_backbone

        self.backbone_type = getattr(cfg, "backbone", "resnet34").lower()

        self.use_axial_encoder = getattr(cfg, "use_axial_encoder", getattr(cfg, "use_ssm_encoder", True))
        self.use_axial_decoder = getattr(cfg, "use_axial_decoder", getattr(cfg, "use_ssm_decoder", True))
        self.use_freq_c3 = getattr(cfg, "use_freq_c3", True)

        if self.backbone_type == "resnet34":
            self.backbone = _get_resnet34(pretrained=pretrained_backbone)

            if in_channels != 3:
                old_conv = self.backbone.conv1
                self.backbone.conv1 = nn.Conv2d(
                    in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False,
                )

            self.conv1 = self.backbone.conv1
            self.bn1 = self.backbone.bn1
            self.relu = self.backbone.relu
            self.maxpool = self.backbone.maxpool

            self.layer1 = self.backbone.layer1
            self.layer2 = self.backbone.layer2
            self.layer3 = self.backbone.layer3
            self.layer4 = self.backbone.layer4

            self.proj_c2 = nn.Identity()
            self.proj_c3 = nn.Identity()
            self.proj_c4 = nn.Identity()
            self.proj_c5 = nn.Identity()

        elif self.backbone_type == "convnext_tiny":
            self.backbone = _get_convnext_tiny(pretrained=pretrained_backbone)

            if in_channels != 3:
                raise ValueError("ConvNeXt backbone currently only supports 3 input channels (RGB)")

            self.proj_c2 = nn.Conv2d(96, 64, kernel_size=1, bias=False)
            self.proj_c3 = nn.Conv2d(192, 128, kernel_size=1, bias=False)
            self.proj_c4 = nn.Conv2d(384, 256, kernel_size=1, bias=False)
            self.proj_c5 = nn.Conv2d(768, 512, kernel_size=1, bias=False)

        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")

        self.axial_c4 = AxialGatedDWConv2D(256) if self.use_axial_encoder else nn.Identity()
        self.axial_c5 = AxialGatedDWConv2D(512) if self.use_axial_encoder else nn.Identity()

        self.freq_c3 = FrequencyFilter2D(128) if self.use_freq_c3 else nn.Identity()

        self.center = ConvBNReLU(512, 256)
        self.dec4 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=256, use_axial=self.use_axial_decoder)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128, use_axial=self.use_axial_decoder)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64, use_axial=False)

        self.final_conv = nn.Sequential(
            ConvBNReLU(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        if self.backbone_type == "resnet34":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.maxpool(x)
            c2 = self.layer1(x)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)

        elif self.backbone_type == "convnext_tiny":
            feats = self.backbone.features
            x = feats[0](x)
            x = feats[1](x)
            c2 = x
            x = feats[2](x)
            x = feats[3](x)
            c3 = x
            x = feats[4](x)
            x = feats[5](x)
            c4 = x
            x = feats[6](x)
            x = feats[7](x)
            c5 = x

        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")

        c2 = self.proj_c2(c2)
        c3 = self.proj_c3(c3)
        c4 = self.proj_c4(c4)
        c5 = self.proj_c5(c5)

        c4 = self.axial_c4(c4)
        c5 = self.axial_c5(c5)

        c3_freq = self.freq_c3(c3)

        x = self.center(c5)
        x = self.dec4(x, c4)
        x = self.dec3(x, c3_freq)
        x = self.dec2(x, c2)

        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        logits = self.final_conv(x)

        return logits


if __name__ == "__main__":
    model = WaterSegModel()
    x = torch.randn(2, cfg.in_channels, cfg.img_size, cfg.img_size)
    y = model(x)
    print("input:", x.shape, "output:", y.shape)
