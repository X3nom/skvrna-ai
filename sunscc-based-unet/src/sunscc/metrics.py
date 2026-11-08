"""Detection-level metrics (paper sec. 5.1, eq. 2, table. 2).

The paper evaluates segmentation as sunspot DETECTION, not per-pixel accuracy:
connected components in the ground truth and in the prediction are matched; each
matched GT spot is a true positive, GT spots with no match are false negatives
(= missed detections) and predicted spots with no GT counterpart are false positives.
Overdetection appears as *fusions*: several GT spots covered by a single predicted
component (see App. E): if one predicted component is the best match for >1 GT spot,
the first is TP and extras count as fusions (in the original repo the same predicted
component can serve multiple times).

F1 = 2TP / (2TP + FN + FP)

Sizes are reported in 5 bins of sunspot pixel area (table 2):
  bin0 <50, bin1 50-100, bin2 100-400, bin3 400-800, bin4 >800
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import label as connected_components

SIZE_BINS = (50, 100, 400, 800)
BIN_NAMES = ("<50", "50-100", "100-400", "400-800", ">800")


def bin_index(area: int) -> int:
    for i, edge in enumerate(SIZE_BINS):
        if area < edge:
            return i
    return len(SIZE_BINS)


def center_distance(labA: np.ndarray, i: int, labB: np.ndarray, j: int) -> float:
    """Euclidean distance between centroids of component i in labA and j in labB."""
    ya, xa = np.nonzero(labA == i)
    yb, xb = np.nonzero(labB == j)
    dy = ya.mean() - yb.mean()
    dx = xa.mean() - xb.mean()
    return float(np.hypot(dy, dx))


@dataclass
class DetectionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    fusions: int = 0
    per_bin: dict = field(default_factory=lambda: {
        name: {"tp": 0, "fp": 0, "fn": 0, "fusions": 0} for name in BIN_NAMES
    })

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fn + self.fp
        return 2 * self.tp / denom if denom else 0.0

    def summary(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "fusions": self.fusions, "f1": self.f1,
            "per_bin": dict(self.per_bin),
        }


def match_components(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    claim_radius: float = 10.0,
) -> DetectionCounts:
    """Associate predicted components to GT components by nearest centroid.

    A GT spot counts as:
    - **TP** if its nearest predicted centroid is within `claim_radius` px
      (a real detection must overlap the spot closely; a far-away blob is NOT
      credited); if several GT spots claim the same predicted component, one is TP
      and the rest are **fusions** (App. E: merged detections).
    - **FN** (missed detection) otherwise.
    - predicted components claimed by nobody are **FP**.
    """
    counts = DetectionCounts()
    lab_gt, n_gt = connected_components(gt_mask)
    lab_pred, n_pred = connected_components(pred_mask)

    if n_gt:
        gt_sizes = np.bincount(lab_gt.ravel())[1:]
        desired = {i: bin_index(int(s)) for i, s in enumerate(gt_sizes, start=1)}
    else:
        desired = {}

    if n_gt == 0 and n_pred == 0:
        return counts
    if n_gt == 0:
        counts.fp = n_pred
        return counts

    # distance matrix GT x Pred
    dist = np.full((n_gt, max(n_pred, 1)), np.inf)
    for j in range(1, n_pred + 1):
        for i in range(1, n_gt + 1):
            d = center_distance(lab_gt, i, lab_pred, j)
            if d <= claim_radius:
                dist[i - 1, j - 1] = d

    claims: dict[int, list[tuple[float, int]]] = {j: [] for j in range(1, n_pred + 1)}
    for i in range(1, n_gt + 1):
        if not np.isfinite(dist[i - 1]).any():
            counts.fn += 1
            counts.per_bin[BIN_NAMES[desired[i]]]["fn"] += 1
            continue
        first = int(np.argmin(dist[i - 1])) + 1
        claims[first].append((dist[i - 1, first - 1], i))

    claimed: set[int] = set()
    for j, cands in claims.items():
        if not cands:
            continue
        cands.sort()
        (_d0, i_best) = cands[0]
        counts.tp += 1
        counts.per_bin[BIN_NAMES[desired[i_best]]]["tp"] += 1
        counts.fusions += len(cands) - 1
        counts.per_bin[BIN_NAMES[desired[i_best]]]["fusions"] += len(cands) - 1
        claimed.add(j)
    counts.fp = n_pred - len(claimed)
    return counts


class DetectionF1Accumulator:
    """Accumulate DetectionCounts over a validation loop; report per-bin F1."""

    def __init__(self, claim_radius: float = 10.0) -> None:
        self.claim_radius = claim_radius
        self.total = DetectionCounts()

    def update(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> None:
        counts = match_components(gt_mask.astype(bool), pred_mask.astype(bool), self.claim_radius)
        self.total.tp += counts.tp
        self.total.fp += counts.fp
        self.total.fn += counts.fn
        self.total.fusions += counts.fusions
        for name, c in counts.per_bin.items():
            self.total.per_bin[name]["tp"] += c["tp"]
            self.total.per_bin[name]["fp"] += c["fp"]
            self.total.per_bin[name]["fn"] += c["fn"]
            self.total.per_bin[name]["fusions"] += c["fusions"]

    def compute(self) -> dict:
        return self.total.summary()

    def reset(self) -> None:
        self.total = DetectionCounts()
