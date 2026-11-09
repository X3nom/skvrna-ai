"""FITS -> 16-bit PNG preprocessing with pure-gain stretch and sidecar metadata.

Per-image stretch: pure gain (no lower clip) anchored at in-disk p99.7.
  out = clip(img / p_hi * 65535, 0, 65535)  -> uint16

Sidecar CSV columns:
  stem, orig_w, orig_h, solar_r, center_x, center_y, stretch_hi, stretch_lo,
  native_scale, p_low, p_high, qc, geometry_source, notes
"""

from __future__ import annotations

import csv
import hashlib
import logging
import warnings
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits

from sunscc.pseudolabels import (
    _estimate_geometry_from_image,
    solar_disk_mask,
)

log = logging.getLogger(__name__)

# Disk region for percentile: r <= 0.95 * SOLAR_R (excludes limb rim ambiguity)
DISK_MARGIN = 0.95
# Percentile for gain anchor (high end)
P_HIGH = 99.7
# Percentile for low-end reference (not subtracted, just recorded)
P_LOW = 0.5
# Resize target for pre-2008 images
TARGET_SIZE = 2048
# PNG compression level (1=fastest, 9=smallest; 4 is sweet spot)
PNG_COMPRESSION = 4


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_raw_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    """Load FITS with BZERO/BSCALE applied, return float32 array + header."""
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header
    if data.ndim != 2:
        raise ValueError(f"{path}: expected 2D image, got {data.ndim}D")
    return data, header


def _estimate_geometry(data: np.ndarray, header: fits.Header) -> tuple[float, float, float, str]:
    """Return (cy, cx, radius, source) from header or fallback."""
    # First try header SOLAR_R
    radius = header.get("SOLAR_R")
    if radius is not None:
        cy = (data.shape[0] - 1) / 2.0
        cx = (data.shape[1] - 1) / 2.0
        return float(cy), float(cx), float(radius), "header"
    # Fallback: estimate from image
    cy, cx, radius = _estimate_geometry_from_image(data)
    return float(cy), float(cx), float(radius), "fallback"


def _disk_percentiles(
    data: np.ndarray, cy: float, cx: float, radius: float
) -> tuple[float, float]:
    """Compute p_low and p_high over in-disk pixels (r <= DISK_MARGIN * radius)."""
    mask = solar_disk_mask(data.shape, cy, cx, radius, margin=DISK_MARGIN)
    vals = data[mask]
    if vals.size == 0:
        raise ValueError("disk mask empty")
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("no finite in-disk pixels")
    return float(np.percentile(vals, P_LOW)), float(np.percentile(vals, P_HIGH))


def _nn_upsample(data: np.ndarray, target: int) -> np.ndarray:
    """Nearest-neighbor upsample to target x target (assumes square)."""
    h, w = data.shape
    if h == target and w == target:
        return data
    # Use cv2 resize with INTER_NEAREST
    return cv2.resize(data, (target, target), interpolation=cv2.INTER_NEAREST)


def preprocess_image(
    fits_path: Path,
    png_dir: Path,
    overwrite: bool = False,
) -> dict:
    """
    Convert one FITS to stretched PNG. Returns the metadata row (NOT written
    to CSV here - the caller/batch function owns the CSV writer).
    """
    stem = fits_path.stem
    png_path = png_dir / f"{stem}.png"

    if png_path.exists() and not overwrite:
        return {"stem": stem, "qc": "skipped", "notes": "PNG exists, overwrite=False"}

    try:
        data, header = _load_raw_fits(fits_path)
        orig_h, orig_w = data.shape

        # Geometry
        cy, cx, radius, geo_source = _estimate_geometry(data, header)

        # Percentiles on native resolution (before any resize)
        p_low, p_high = _disk_percentiles(data, cy, cx, radius)

        # Upsample 1024 -> 2048 if needed (after percentiles, on native data)
        if orig_h < TARGET_SIZE or orig_w < TARGET_SIZE:
            data = _nn_upsample(data, TARGET_SIZE)
            # Update geometry for new size
            scale = TARGET_SIZE / orig_h
            cy *= scale
            cx *= scale
            radius *= scale
            orig_h = orig_w = TARGET_SIZE
            notes = "nn_upsampled_1024to2048"
        else:
            notes = ""

        # Pure-gain stretch: out = clip(img / p_high * 65535, 0, 65535)
        # No lower clip; sky/NaN -> 0
        gain = 65535.0 / max(p_high, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stretched = np.clip(data * gain, 0.0, 65535.0)
        # NaN -> 0
        stretched = np.nan_to_num(stretched, nan=0.0, posinf=65535.0, neginf=0.0)
        stretched_u16 = stretched.astype(np.uint16)

        # Write PNG
        png_dir.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(
            str(png_path),
            stretched_u16,
            [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
        )
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {png_path}")

        # Metadata row
        row = {
            "stem": stem,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "solar_r": radius,
            "center_x": cx,
            "center_y": cy,
            "stretch_hi": p_high,
            "stretch_lo": p_low,
            "native_scale": 1.0 if orig_h == TARGET_SIZE else TARGET_SIZE / orig_h,
            "p_low": p_low,
            "p_high": p_high,
            "qc": "ok",
            "geometry_source": geo_source,
            "notes": notes,
        }
        return row

    except Exception as e:
        log.exception("Failed %s", fits_path.name)
        row = {
            "stem": stem,
            "orig_w": -1, "orig_h": -1,
            "solar_r": -1.0, "center_x": -1.0, "center_y": -1.0,
            "stretch_hi": -1.0, "stretch_lo": -1.0,
            "native_scale": -1.0,
            "p_low": -1.0, "p_high": -1.0,
            "qc": "rotten", "geometry_source": "error", "notes": str(e)
        }
        return row


def _row_to_csv(row: dict) -> list:
    return [
        row["stem"], row["orig_w"], row["orig_h"],
        f"{row['solar_r']:.4f}", f"{row['center_x']:.4f}", f"{row['center_y']:.4f}",
        f"{row['stretch_hi']:.4f}", f"{row['stretch_lo']:.4f}",
        f"{row['native_scale']:.4f}",
        f"{row['p_low']:.4f}", f"{row['p_high']:.4f}",
        row["qc"], row["geometry_source"], row["notes"]
    ]


def _preprocess_worker(fits_path: Path, png_dir: Path, overwrite: bool) -> dict:
    """Module-level wrapper so ProcessPoolExecutor can pickle it."""
    return preprocess_image(fits_path, png_dir, overwrite=overwrite)


def preprocess_batch(
    input_dir: Path,
    png_dir: Path,
    meta_csv: Path,
    jobs: int = 1,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Batch preprocess directory of FITS. Returns (ok, rotten) counts.

    Only the parent process touches the CSV (workers return row dicts),
    and each completed row is flushed immediately so a crash keeps history.
    Already-successful stems (qc=ok in CSV and PNG present) are skipped on resume.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    fits_files = sorted(list(input_dir.glob("**/*.FTS")) + list(input_dir.glob("**/*.fts")))
    if not fits_files:
        log.warning("No FITS files found in %s", input_dir)
        return 0, 0

    png_dir.mkdir(parents=True, exist_ok=True)
    meta_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "stem", "orig_w", "orig_h", "solar_r", "center_x", "center_y",
        "stretch_hi", "stretch_lo", "native_scale", "p_low", "p_high",
        "qc", "geometry_source", "notes"
    ]

    # Stems already processed successfully in a previous run -> skip (resume)
    done_stems: set[str] = set()
    if meta_csv.exists() and not overwrite:
        try:
            with meta_csv.open(newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("qc") == "ok":
                        done_stems.add(r["stem"])
        except csv.Error as e:
            log.warning("could not read existing meta CSV for resume (%s); rebuilding rows", e)
    if overwrite:
        meta_csv.write_text("")  # start fresh
        done_stems.clear()
    todo = [f for f in fits_files if f.stem not in done_stems]
    n_skip = len(fits_files) - len(todo)
    if n_skip:
        log.info("skipping %d already-processed images", n_skip)

    file_exists = meta_csv.exists() and meta_csv.stat().st_size > 0
    ok = rotten = 0
    with meta_csv.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
            csvfile.flush()

        if jobs <= 1:
            for f in todo:
                row = _preprocess_worker(f, png_dir, overwrite)
                writer.writerow(_row_to_csv(row))
                csvfile.flush()
                if row["qc"] == "ok":
                    ok += 1
                elif row["qc"] == "rotten":
                    rotten += 1
        else:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = [ex.submit(_preprocess_worker, f, png_dir, overwrite) for f in todo]
                for fut in as_completed(futs):
                    row = fut.result()
                    writer.writerow(_row_to_csv(row))
                    csvfile.flush()
                    if row["qc"] == "ok":
                        ok += 1
                    elif row["qc"] == "rotten":
                        rotten += 1

    log.info("done: %d ok, %d rotten, %d skipped", ok, rotten, n_skip)
    return ok, rotten