"""Lightning module for stochastic-multi-threshold pseudo-label training.

Training objective (paper sec. 4.2.2): sum of
- binary cross-entropy (with logits)
- log-cosh Dice loss (differentiable pseudo-Dice whose log/cosh makes it robust)

Optimizer: AdamW, initial learning rate 1e-4. Epochs 40, batch 16 in the paper -
scaled down for 2 GB VRAM via config (batch + gradient accumulation).
"""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import JaccardIndex

from sunscc.unet import PaperUNet


def dice_loss(pred_prob: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Continuous soft-Dice loss averaging over the batch."""
    axes = (-3, -2, -1)
    inter = torch.sum(pred_prob * target, dim=axes)
    cardinality = torch.sum(pred_prob + target, dim=axes)
    dice_score = (2.0 * inter + eps) / (cardinality + eps)
    return 1.0 - dice_score.mean()


def log_cosh_dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """log-cosh of the dice loss: t = soft-dice loss, loss = log(cosh(t))."""
    t = dice_loss(torch.sigmoid(pred_logits), target, eps=eps)
    return torch.log(torch.cosh(t + 1e-12))


def bce_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, target)


class SegmentationModule(LightningModule):
    """Binary-sunspot UNet with the paper's loss and AdamW optimizer."""

    def __init__(
        self,
        width_scale: float = 1.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        dice_eps: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = PaperUNet(width_scale=width_scale)
        self.iou = JaccardIndex(task="binary", threshold=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def compute_losses(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        bce = bce_loss(logits, target)
        dice = log_cosh_dice_loss(logits, target, eps=self.hparams.dice_eps)
        return {"bce": bce, "dice": dice, "loss": bce + dice}

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, float], batch_idx: int) -> torch.Tensor:
        images, masks, _thresholds = batch
        logits = self(images)
        losses = self.compute_losses(logits, masks)
        self.log("train/loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/bce", losses["bce"], on_step=False, on_epoch=True)
        self.log("train/dice", losses["dice"], on_step=False, on_epoch=True)
        return losses["loss"]

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, float], batch_idx: int) -> None:
        images, masks, _thresholds = batch
        logits = self(images)
        losses = self.compute_losses(logits, masks)
        probs = torch.sigmoid(logits)
        self.iou.update(probs, masks.int())
        self.log("val/loss", losses["loss"], prog_bar=True)
        self.log("val/iou", self.iou, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, float], batch_idx: int) -> None:
        images, masks, _thresholds = batch
        logits = self(images)
        losses = self.compute_losses(logits, masks)
        probs = torch.sigmoid(logits)
        self.iou.update(probs, masks.int())
        self.log("test/loss", losses["loss"])
        self.log("test/iou", self.iou, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )
        return optimizer

    @property
    def example_input_array(self) -> torch.Tensor:  # enables Lightning shape inference / ckpt sanity
        return torch.zeros(1, 1, 512, 512, device=self.device)


def count_parameters(module: LightningModule) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def dummy_batch(device: str = "cpu") -> dict[str, Any]:
    """Small synthetic batch for smoke tests (2 patches, one empty one spotty)."""
    images = torch.zeros(2, 1, 128, 128, device=device)
    masks = torch.ones(2, 1, 128, 128, device=device) * 0.0
    masks[1, 0, 40:80, 40:80] = 1.0
    images[1, 0, 40:80, 40:80] = 0.2
    return {"images": images, "masks": masks}
