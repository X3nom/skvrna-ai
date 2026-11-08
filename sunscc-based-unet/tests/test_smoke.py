"""Smoke tests: run components on tiny synthetic data + the downloaded sample FITS
(if present in /tmp/opencode). No GPU required."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sunscc.unet import PaperUNet


def test_unet_paper_shapes():
    model = PaperUNet()
    x = torch.randn(1, 1, 512, 512)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 512, 512)


def test_unet_scaled_shapes():
    model = PaperUNet(width_scale=0.5)
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 256, 256)


def test_paper_parameter_count():
    """Paper fig-1 UNet should be ~7-8M params with 32..512 widths."""
    model = PaperUNet()
    n = sum(p.numel() for p in model.parameters())
    assert 6e6 < n < 1e7, f"unexpected param count {n}"


def test_losses():
    from sunscc.lightning_module import bce_loss, log_cosh_dice_loss

    logits = torch.randn(2, 1, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.9).float()
    bce = bce_loss(logits, target)
    dice = log_cosh_dice_loss(logits, target)
    assert torch.isfinite(bce) and torch.isfinite(dice)
    assert bce.item() > 0 and dice.item() > 0

    perfect = torch.where(target > 0, 20.0, -20.0)
    assert bce_loss(perfect, target).item() < 1e-3
    assert log_cosh_dice_loss(perfect, target).item() < 1e-2


def test_detection_metrics():
    from sunscc.metrics import DetectionF1Accumulator

    gt = np.zeros((256, 256), bool)
    gt[10:20, 10:30] = True   # 200 px -> bin2
    gt[200:210, 200:230] = True
    pred = np.zeros((256, 256), bool)
    pred[9:21, 9:31] = True   # matches spot 1
    # spot 2 missed; one false positive
    pred[100:106, 100:106] = True

    acc = DetectionF1Accumulator()
    acc.update(gt, pred)
    r = acc.compute()
    assert r["tp"] == 1
    assert r["fp"] == 1
    assert r["fn"] == 1
    assert r["per_bin"]["100-400"]["tp"] == 1
    assert r["per_bin"]["100-400"]["fn"] == 1


def test_pseudolabels_synthetic():
    """No FITS needed fully synthetic disk with a dark spot."""
    from sunscc.pseudolabels import (
        contrast_tophat,
        limb_darkening_correct,
        solar_disk_mask,
        threshold_mask,
    )

    size = 512
    yy, xx = np.indices((size, size), dtype=np.float32)
    cy = cx = size / 2
    radius = 200
    r = np.hypot(yy - cy, xx - cx)
    image = 30000.0 * (0.6 + 0.4 * np.cos(0.5 * np.pi * np.clip(r / radius, 0, 1.2)))
    image += np.random.RandomState(0).normal(0, 30, image.shape).astype(np.float32)
    image[r > radius] = 10.0
    # dark spot
    spot = np.hypot(yy - (cy - 60), xx - (cx + 40)) < 15
    image[spot] *= 0.35

    corrected = limb_darkening_correct(image, cy, cx, radius)
    disk_m = solar_disk_mask((size, size), cy, cx, radius)
    tophat = contrast_tophat(corrected, footprint_radius=25, mask=disk_m)
    mask = threshold_mask(tophat, threshold=3000.0, min_area=5)

    # the spot should be detected: overlap ratio
    assert (mask & spot).sum() / spot.sum() > 0.5
    # quiet-sun pixels should NOT trigger
    quiet = (disk_m & ~spot & (r < radius * 0.8))
    assert mask[quiet].mean() < 0.02
