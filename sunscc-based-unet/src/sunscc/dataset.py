"""Dataset with stochastic multi-threshold pseudo-labels (paper sec. 4.2.2).

Each white-light FITS image is split into a 4x4 grid of 512x512 patches (paper sec.
4.1). For every patch we load its set of pseudo-labels (one mask per threshold, e.g.
T325, T375, T425) and, at training time, pick ONE mask at random per __getitem__
call. This is the paper's core trick: alternating pseudo-labels prevent the CNN from
learning a single biased labeling policy.

Randomness note: mask selection happens in __getitem__, i.e. once per worker per
epoch pass, so with num_workers>0 each worker uses its own random stream (seeded by
Lightning's worker seeding) — enough stochasticity in practice.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

PATCH_SIZE = 512
GRID = 4  # 2048 / 512 = 4x4 patches per image
PAPER_THRESHOLDS: tuple[float, ...] = (325.0, 375.0, 425.0)


def image_id_from_filename(path: Path | str) -> str:
    """UPH20130101133418.FTS -> 20130101133418 (sortable timestamp id)."""
    stem = Path(path).stem  # e.g. UPH20130101133418
    return "".join(ch for ch in stem if ch.isdigit())


def patch_grid_offsets() -> list[tuple[int, int]]:
    """(row, col) offsets of every patch in the 4x4 grid."""
    return [(r * PATCH_SIZE, c * PATCH_SIZE) for r in range(GRID) for c in range(GRID)]


def load_image_patch(path: Path | str, row: int, col: int) -> np.ndarray:
    """Load a 512x512 patch (float32, normalized 0..1) from FITS or PNG.

    PNG path: already gain-stretched to 0..65535, normalize by fixed /65535.0.
    FITS path: raw counts, normalized per-image by 99.9th percentile (ImageScaleCache).
    """
    path = Path(path)
    if path.suffix.lower() in (".png", ".PNG"):
        patch = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if patch is None:
            raise ValueError(f"Failed to read PNG: {path}")
        patch = np.asarray(patch[row : row + PATCH_SIZE, col : col + PATCH_SIZE], dtype=np.float32)
        return patch / 65535.0
    # FITS path
    with fits.open(path) as hdul:
        patch = hdul[0].data[row : row + PATCH_SIZE, col : col + PATCH_SIZE]
    return np.asarray(patch, dtype=np.float32) / ImageScaleCache()(path)


class ImageScaleCache:
    """Per-image normalization scale (99.9th percentile of the full frame).

    USET L1c quiet-sun intensity is ~2400-3300 counts (NOT 0-65535); normalizing
    per image handles exposure drift between snapshots. Scales are cached.
    """

    def __init__(self) -> None:
        self._scales: dict[str, float] = {}

    def __call__(self, path: Path | str) -> float:
        key = Path(path).stem
        if key in self._scales:
            return self._scales[key]
        with fits.open(Path(path)) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32)
        scale = float(np.percentile(data[data > 0], 99.9))
        scale = max(scale, 1.0)
        self._scales[key] = scale
        return scale

    def set_override(self, path: Path | str, scale: float) -> None:
        """Used by predict.py to force a scale computed at inference time."""
        self._scales[Path(path).stem] = float(scale)


class PseudoLabelPatchDataset(Dataset):
    """Patches from full-disk images with a random pseudo-label target per access.

    Parameters
    ----------
    image_files:
        White-light FITS paths.
    mask_dirs:
        Mapping threshold value -> directory containing '{stem}.png' masks,
        e.g. {325: '/data/T325_fgbg', 375: '/data/T375_fgbg', 425: '/data/T425_fgbg'}.
    thresholds:
        Which of `mask_dirs` to train with (paper: 325/375/425).
    transform:
        Optional albumentations-like callable: transform(image=..., mask=...) -> dict.
    """

    def __init__(
        self,
        image_files: Sequence[Path | str],
        mask_dirs: dict[float, Path | str],
        thresholds: Sequence[float] = PAPER_THRESHOLDS,
        transform: Callable | None = None,
    ) -> None:
        self.transform = transform
        self.scale_cache = ImageScaleCache()
        self.thresholds = [float(t) for t in thresholds]
        self.mask_dirs = {float(k): Path(v) for k, v in mask_dirs.items()}
        for t in self.thresholds:
            if t not in self.mask_dirs:
                raise ValueError(f"threshold {t} not present in mask_dirs")

        self.samples: list[tuple[Path, tuple[int, int], dict[float, Path]]] = []
        skipped = 0
        for img_path in image_files:
            stem = Path(img_path).stem
            masks = {
                t: self.mask_dirs[t] / f"{stem}.png"
                for t in self.thresholds
                if (self.mask_dirs[t] / f"{stem}.png").exists()
            }
            if not masks:
                skipped += 1
                continue
            for offsets in patch_grid_offsets():
                self.samples.append((Path(img_path), offsets, masks))
        if skipped:
            log.warning("%d images skipped: no mask file for any threshold", skipped)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        img_path, (row, col), masks = self.samples[index]

        image = load_image_patch(img_path, row, col)
        threshold = float(random.choice(sorted(masks)))
        raw = cv2.imread(str(masks[threshold]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask = (raw > 127.0)[row : row + PATCH_SIZE, col : col + PATCH_SIZE].astype(np.float32)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        image_t = torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(np.ascontiguousarray(mask), dtype=torch.float32).unsqueeze(0)
        return image_t, mask_t, threshold
