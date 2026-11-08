"""Paper-faithful UNet (Sayez et al. 2023, sec. 4.1 / fig. 1).

- 5 levels: 32, 64, 128, 256, 512 filters going down.
- 4 downsampling steps in the encoder, 4 upsampling steps in the decoder.
- 2 convolutional blocks at each level ("lum").
- Single binary output (1 logit channel).
- ``width_scale`` shrinks ALL level widths uniformly (0.5 -> 16/32/64/128/256) as an
  escape hatch for 2 GB GPUs; default 1.0 is paper-faithful.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

PAPER_LEVEL_WIDTHS = (32, 64, 128, 256, 512)


class ConvBlock(nn.Module):
    """(conv 3x3, ReLU, conv 3x3, ReLU) — one of the 2 blocks per level."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PaperUNet(nn.Module):
    """UNet with 4 down / 4 up steps and arbitrary level widths."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        widths: tuple[int, ...] = PAPER_LEVEL_WIDTHS,
        width_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("need at least 2 levels")
        widths = tuple(int(w * width_scale) for w in widths)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 4 downsampling steps
        self.encoders = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i, width in enumerate(widths):
            in_ch = in_channels if i == 0 else widths[i - 1]
            self.encoders.append(ConvBlock(in_ch, width))

        # decoder walks back up the widths, skipping the bottleneck (built once per step)
        for i in range(len(widths) - 1, 0, -1):
            self.decoder_ups.append(nn.ConvTranspose2d(widths[i], widths[i - 1], kernel_size=2, stride=2))
            self.decoder_blocks.append(ConvBlock(widths[i - 1] * 2, widths[i - 1]))

        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        for up, block, skip in zip(self.decoder_ups, self.decoder_blocks, reversed(skips[:-1])):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([skip, x], dim=1))

        if x.shape[-2:] != skips[0].shape[-2:]:
            x = F.interpolate(x, size=skips[0].shape[-2:], mode="bilinear", align_corners=False)
        return self.head(x)


def build_paper_unet(width_scale: float = 1.0) -> PaperUNet:
    """Paper architecture (fig. 1, sec 4.1); width_scale=1.0 -> 32/64/128/256/512."""
    return PaperUNet(width_scale=width_scale)
