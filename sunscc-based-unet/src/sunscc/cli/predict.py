"""Inference: full-disk white-light image -> binary sunspot mask.

Paper fig. 1: the 2048x2048 input is segmented in a 4x4 grid of 512x512 patches
(no overlap), each passed through the UNet; outputs (sigmoid) are re-assembled at
full resolution and thresholded at 0.5.

    sunscc-predict --checkpoint CKPT --image IN.FTS --output OUT.png
    sunscc-predict --checkpoint CKPT --image IN.png --meta-csv META.csv --output OUT.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from sunscc.dataset import GRID, PATCH_SIZE, load_image_patch, patch_grid_offsets
from sunscc.pseudolabels import get_meta, read_solar_disk_geometry, solar_disk_mask


def _get_scale_and_geometry(image_path: Path, meta_csv: Path | None):
    """Return (scale, cy, cx, radius) for the given image."""
    path = Path(image_path)
    if path.suffix.lower() in (".png", ".PNG"):
        if meta_csv is None:
            raise ValueError("PNG input requires --meta-csv for geometry and stretch_hi")
        meta = get_meta(path.stem, meta_csv)
        if meta is None:
            raise ValueError(f"stem {path.stem} not found in meta CSV")
        scale = float(meta["stretch_hi"])
        cy = float(meta["center_y"])
        cx = float(meta["center_x"])
        radius = float(meta["solar_r"])
        return scale, cy, cx, radius
    # FITS fallback
    from sunscc.dataset import ImageScaleCache
    scale = ImageScaleCache()(image_path)
    cy, cx, radius = read_solar_disk_geometry(image_path)
    return scale, cy, cx, radius


def predict_full_disk(
    checkpoint_path: Path | str,
    image_path: Path | str,
    device: str | None = None,
    border_margin: float = 1.0,
    meta_csv: Path | str | None = None,
) -> np.ndarray:
    """Return a 2048x2048 bool mask for one image (FITS or PNG)."""
    from sunscc.lightning_module import SegmentationModule

    model = SegmentationModule.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image_path = Path(image_path)
    size = GRID * PATCH_SIZE
    out = np.zeros((size, size), dtype=np.float32)

    scale, cy, cx, radius = _get_scale_and_geometry(image_path, meta_csv)

    with torch.no_grad():
        for row, col in patch_grid_offsets():
            patch = load_image_patch(image_path, row, col)
            inputs = torch.from_numpy(patch / scale)[None, None].to(device)
            prob = torch.sigmoid(model(inputs))[0, 0].cpu().numpy()
            out[row : row + PATCH_SIZE, col : col + PATCH_SIZE] = prob

    mask = out > 0.5
    inside = solar_disk_mask((size, size), cy, cx, radius * border_margin)
    return (mask & inside).astype(bool)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--meta-csv", type=Path, default=None,
                   help="sidecar meta CSV (required for PNG inputs)")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    from cv2 import imwrite

    mask = predict_full_disk(args.checkpoint, args.image, device=args.device, meta_csv=args.meta_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(args.output), (mask.astype(np.uint8) * 255))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
