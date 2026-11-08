"""Pseudo-label generation for SunSCC segmentation (paper doi:10.1029/2023JA031548, sec 4.2.1).

Top-level entry point is :func:`generate_pseudolabel`: a FITS filepath in, a mask
filepath (or array) out. Steps:

1. Load raw white-light image.
2. Limb darkening correction: fit a polynomial to the mean radial intensity profile
   (polar projection) with sigma-clipping so dark spots/sunburns don't bias the fit,
   then divide the raw image by the fitted profile (profile normalized to 1 at center,
   keeping raw intensity units).
3. Black top-hat transform (contrast map: dark structures on bright surround).
4. Threshold the top-hat image -> binary sunspot mask.
5. Restrict to the solar disk (SOLAR_R header, with a small margin inside the limb).

NOTE on threshold units: the paper/repo use integer values (250..500). The exact
normalization used by the (private) original pipeline is unknown; here thresholds are
expressed as *top-hat contrast* in the image's native scale (which is 0..65535 for
USET 16-bit FITS after our limb correction). The defaults below (325/375/425 mapped
onto whatever scale results) must be empirically calibrated against paper F1 values.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_holes, remove_small_objects

log = logging.getLogger(__name__)

PAPER_THRESHOLDS = (325, 375, 425)
THRESHOLD_STEP = 25
THRESHOLD_MIN = 250
THRESHOLD_MAX = 500

DEFAULT_FOOTPRINT_RADIUS = 25  # SE must exceed typical spot radius to fill interiors
DEFAULT_MIN_DETECTION_AREA = 8
DEFAULT_LIMB_MARGIN = 0.995


# --------------------------------------------------------------------------- #
# FITS loading
# --------------------------------------------------------------------------- #
def load_wl_image(path: Path | str) -> np.ndarray:
    """Load the first image HDU of a USET white-light FITS file as float32."""
    path = Path(path)
    with fits.open(path) as hdul:  # memmap=False implicitly: BZERO/BSCALE keywords present
        data = hdul[0].data
        header = hdul[0].header
    if data.ndim != 2:
        raise ValueError(f"{path}: expected 2D image, got shape {data.shape}")
    if header.get("NAXIS1", data.shape[1]) != 2048:
        log.warning("%s: image is %sx%s (pre-2006 USET?), not 2048x2048",
                    path.name, data.shape[1], data.shape[0])
    return np.asarray(data, dtype=np.float32)


def read_solar_disk_geometry(path: Path | str, image: np.ndarray | None = None) -> tuple[float, float]:
    """Return (center_y, center_x, radius) ... actually (row, col) offsets and radius.

    Reads SOLAR_R from the FITS header (present in USET L1c); falls back to a
    half-maximum threshold estimate when missing. Centers are taken as image center
    (repo convention: SOLAR_R and mass center coincide for L1c).
    """
    with fits.open(Path(path)) as hdul:
        header = hdul[0].header
    radius = header.get("SOLAR_R")
    image = image if image is not None else load_wl_image(path)
    ny, nx = image.shape
    center_y, center_x = (ny - 1) / 2.0, (nx - 1) / 2.0
    if radius is None:
        log.warning("SOLAR_R missing in %s header; estimating from half-max contour", Path(path).name)
        radius = estimate_solar_radius(image, center_y, center_x)
    return float(center_y), float(center_x), float(radius)


def estimate_solar_radius(image: np.ndarray, center_y: float, center_x: float) -> float:
    """Rough solar radius estimate: where the intensity crosses 50% between
    the sky background level and the quiet-sun level along the central row/column."""
    corners = np.concatenate([image[:50, :50].ravel(), image[:50, -50:].ravel(),
                              image[-50:, :50].ravel(), image[-50:, -50:].ravel()])
    level = (np.nanmedian(corners) + np.nanpercentile(image, 99)) / 2.0

    radii = []
    for profile, offsets in (
        (image[round(center_y)], np.arange(image.shape[1]) - center_x),
        (image[:, round(center_x)], np.arange(image.shape[0]) - center_y),
    ):
        above = offsets[profile > level]
        if above.size:
            radii.append(float(np.abs(above).max()))
    return max(radii) if radii else float(min(image.shape) / 2.0)


# --------------------------------------------------------------------------- #
# Limb darkening correction
# --------------------------------------------------------------------------- #
def fit_limb_darkening_profile(
    image: np.ndarray,
    center_y: float,
    center_x: float,
    radius: float,
    degree: int = 5,
) -> tuple[np.ndarray, float]:
    """Fit a polynomial to the azimuthally averaged radial intensity profile.

    Returns (ppoly(r/rmax)->intensity, fit_value_at_center) with the profile
    normalized such that its value at r=0 is 1.0. Sigma-clipping discards radial
    bins whose mean sits well below the smooth fit (sunspots pull means down).
    """
    yy, xx = np.indices(image.shape, dtype=np.float32)
    rr = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    rnorm = (rr / radius).ravel()
    inside = rnorm < 0.98
    rin = np.clip(rnorm[inside], 0.0, None)
    win = image.ravel()[inside]

    nbins = 200
    bin_edges = np.linspace(0, 0.98, nbins + 1)
    bin_idx = np.digitize(rin, bin_edges) - 1
    sums = np.bincount(bin_idx, weights=win, minlength=nbins)
    counts = np.bincount(bin_idx, minlength=nbins)
    valid = counts > 0
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])[valid]
    means = sums[valid] / counts[valid]

    # iteratively refit discarding bins far below the fit (spots, transits)
    keep = np.ones(centers.size, dtype=bool)
    coeffs = np.ones(degree + 1)
    for _ in range(5):
        coeffs = np.polyfit(centers[keep], means[keep], deg=degree)
        smooth = np.polyval(coeffs, centers)
        resid = means - smooth
        sigma = resid[keep].std()
        keep = resid > -1.5 * sigma  # only clip low outliers (dark contamination)

    pfit = np.poly1d(coeffs)
    central = float(pfit(0.0))

    def profile_of_r(r):
        return pfit(r) / central

    return profile_of_r, central


def limb_darkening_correct(
    image: np.ndarray,
    center_y: float,
    center_x: float,
    radius: float,
    degree: int = 5,
    model: dict | None = None,
) -> np.ndarray:
    """Divide the raw image by the fitted limb-darkening profile.

    Output keeps the raw intensity scale near the center (profile normalized to 1
    at r=0), so later threshold values remain comparable to the paper's integers.
    Outside the disk pixels are left as-is (they'll be masked out anyway).
    """
    if model is None:
        profile_of_r, _ = fit_limb_darkening_profile(image, center_y, center_x, radius, degree)
    else:
        profile_of_r = np.poly1d(model["coeffs"]) / model["central"]

    yy, xx = np.indices(image.shape, dtype=np.float32)
    rnorm = np.hypot(yy - center_y, xx - center_x) / radius
    gain = profile_of_r(np.clip(rnorm, 0.0, 1.5))
    corrected = image / np.maximum(gain, 1e-3)

    disk = rnorm <= 1.0
    corrected[~disk] = image[~disk]
    return corrected.astype(np.float32)


# --------------------------------------------------------------------------- #
# Top-hat + threshold pseudo-label
# --------------------------------------------------------------------------- #
def contrast_tophat(
    corrected: np.ndarray,
    footprint_radius: int = DEFAULT_FOOTPRINT_RADIUS,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Black top-hat transform restricted to the solar disk.

    Uses OpenCV MORPH_CLOSE with a circular structuring element: faster and more
    robust than skimage for the LARGE structuring elements needed to fill big
    sunspot groups (a SE smaller than a spot cannot fill its dark interior).
    Operates on the limb-darkening-corrected image kept in raw intensity units,
    so threshold values stay comparable to the paper's integer scale.
    Pixels outside the disk (when `mask` given) are zeroed to avoid limb
    artefacts.
    """
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * footprint_radius + 1, 2 * footprint_radius + 1))
    closed = cv2.morphologyEx(corrected, cv2.MORPH_CLOSE, kernel)
    tophat = closed - corrected
    tophat = np.clip(tophat, 0.0, None).astype(np.float32)
    if mask is not None:
        tophat[~mask] = 0.0
    return tophat


def threshold_mask(
    tophat: np.ndarray,
    threshold: float,
    min_area: int = DEFAULT_MIN_DETECTION_AREA,
) -> np.ndarray:
    """Binary sunspot mask from the top-hat contrast image, with speckle cleanup."""
    mask = tophat > threshold
    mask = remove_small_holes(mask, max_size=min_area * 4)
    mask = remove_small_objects(mask, max_size=min_area)
    return mask


def solar_disk_mask(shape: tuple[int, int], center_y: float, center_x: float, radius: float,
                     margin: float = DEFAULT_LIMB_MARGIN) -> np.ndarray:
    """Boolean mask of the solar disk (slightly shrunk to avoid limb artefacts)."""
    yy, xx = np.indices(shape, dtype=np.float32)
    rnorm = np.hypot(yy - center_y, xx - center_x) / radius
    return rnorm <= margin


# --------------------------------------------------------------------------- #
# Top-level one-stop function (filepath -> filepath)
# --------------------------------------------------------------------------- #
def generate_pseudolabel(
    input_path: Path | str,
    output_path: Path | str | None = None,
    threshold: float = 375.0,
    footprint_radius: int = DEFAULT_FOOTPRINT_RADIUS,
    min_area: int = DEFAULT_MIN_DETECTION_AREA,
    overwrite: bool = False,
    return_array: bool = False,
) -> np.ndarray | None:
    """Generate a binary pseudo-label mask for one white-light FITS image.

    filepath in -> filepath out (uint8 PNG, 0/255), or return the bool array when
    ``return_array`` is True and no output_path is given.
    """
    from cv2 import imwrite  # local import so importing this module stays cheap

    input_path = Path(input_path)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            log.info("skip %s (exists)", output_path.name)
            return None

    image = load_wl_image(input_path)
    cy, cx, radius = read_solar_disk_geometry(input_path, image)
    disk_m = solar_disk_mask(image.shape, cy, cx, radius)

    corrected = limb_darkening_correct(image, cy, cx, radius)
    tophat = contrast_tophat(corrected, footprint_radius, mask=disk_m)
    mask = threshold_mask(tophat, threshold, min_area)
    mask = binary_fill_holes(mask)  # solid umbrae sometimes crack

    if return_array or output_path is None:
        return mask
    imwrite(str(output_path), (mask.astype(np.uint8) * 255))
    return None


def threshold_sweep(minimum: int = THRESHOLD_MIN, maximum: int = THRESHOLD_MAX,
                    step: int = THRESHOLD_STEP) -> list[int]:
    """All integer thresholds the repo used for experiments (250..500 step 25)."""
    return list(range(minimum, maximum + 1, step))
