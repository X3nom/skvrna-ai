"""Hydra entrypoint for training the segmentation model.

    sunscc-train data.image_dir=/data/images data.mask_dir=/data/masks
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from sunscc.dataset import PseudoLabelPatchDataset, image_id_from_filename

log = logging.getLogger(__name__)


def parse_timestamp(image_id: str):
    """UPH20130101133418 -> datetime(2013-01-01 13:34:18) (-ish; seconds truncated)."""
    import datetime as dt

    if len(image_id) < 14:
        return None
    try:
        return dt.datetime(
            int(image_id[0:4]), int(image_id[4:6]), int(image_id[6:8]),
            int(image_id[8:10]), int(image_id[10:12]), int(image_id[12:14]),
        )
    except ValueError:
        return None


def exclude_neighbours(images, val_ids, days: int = 1):
    """Paper: snapshots +/- 1 day around validation snapshots are excluded."""
    val_dates = [d for d in (parse_timestamp(i) for i in val_ids) if d]
    keep = []
    for p in images:
        ts = parse_timestamp(image_id_from_filename(p))
        if ts is None:
            keep.append(p)
            continue
        if any(abs((ts - d).total_seconds()) <= days * 86400 for d in val_dates):
            continue
        keep.append(p)
    n_dropped = len(images) - len(keep)
    if n_dropped:
        log.info("temporal exclusion: dropped %d training images within +/-1 day of val", n_dropped)
    return keep


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transform(p: float):
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.RandomRotate90(p=p),
        ]
    )


def build_datasets(cfg: DictConfig) -> tuple[PseudoLabelPatchDataset, PseudoLabelPatchDataset]:
    image_dir = Path(cfg.data.image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image_dir not found: {image_dir} (set SUNSCC_DATA_ROOT / data.image_dir)")
    images = sorted(
        [*image_dir.glob("**/*.FTS"), *image_dir.glob("**/*.fts"),
         *image_dir.glob("**/*.PNG"), *image_dir.glob("**/*.png")]
    )

    mask_dirs: dict[float, str | Path] = {
        float(t): Path(cfg.data.mask_dir) / f"T{int(t)}_fgbg" for t in cfg.data.thresholds
    }
    transform = build_transform(cfg.train.aug.p)

    # --- validation set (pseudo-manual masks, single shared threshold) -------
    val_mask_dir = Path(cfg.data.validation_mask_dir)
    val_mask_files = sorted(val_mask_dir.glob("*.PNG")) + sorted(val_mask_dir.glob("*.png")) if val_mask_dir.is_dir() else []
    val_stems = {f.stem for f in val_mask_files}
    val_images = [p for p in images if p.stem in val_stems]

    # --- training set: exclude +/-1 day around any validation image ---------
    val_ids = {image_id_from_filename(p) for p in val_images}
    train_images = exclude_neighbours(images, val_ids)

    train_ds = PseudoLabelPatchDataset(
        train_images,
        mask_dirs=mask_dirs,
        thresholds=[float(t) for t in cfg.data.thresholds],
        transform=transform,
    )

    val_dataset: PseudoLabelPatchDataset
    if val_images:
        # per-image hand-corrected mask in ONE folder; use each image's OWN mask:
        val_dataset = PseudoLabelPatchDataset(
            val_images,
            mask_dirs={float(t): val_mask_dir for t in cfg.data.thresholds},
            thresholds=[float(t) for t in cfg.data.thresholds[:1]],
            transform=None,
        )
        if cfg.data.validation_max_images:
            val_dataset.samples = val_dataset.samples[: cfg.data.validation_max_images * 16]
    else:
        log.warning("No validation masks in %s; validation disabled", val_mask_dir)
        val_dataset = None  # type: ignore[assignment]
    return train_ds, val_dataset


CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def apply_thread_caps(torch_threads: int) -> None:
    """Cap intra/inter-op torch threads (CPU server training; 0 = torch default).

    Must run before any tensor math parallelizes; inter-op only settable once.
    Dataloader workers additionally inherit cv2.setNumThreads(0) from dataset.py
    so each worker stays single-threaded (no oversubscription of the caps above).
    """
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
        log.info("torch intra-op threads: %d", torch.get_num_threads())
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized (e.g. fast_dev_run re-import); keep going


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Working directory: %s", os.getcwd())
    apply_thread_caps(int(cfg.train.get("torch_threads", 0)))
    seed_everything(cfg.train.seed)

    train_ds, val_ds = build_datasets(cfg)
    if len(train_ds) == 0:
        raise RuntimeError(
            f"No training patches. Checked mask dirs {list(Path(cfg.data.mask_dir).glob('T*'))} "
            f"for {len(sorted(Path(cfg.data.image_dir).glob('**/*.FTS')))} images"
        )
    log.info("train patches: %d (from %d images)", len(train_ds), len({s[0].stem for s in train_ds.samples}))

    n_workers = int(cfg.data.num_workers)
    loader_kwargs = {
        "num_workers": n_workers,
        "pin_memory": True,
        "persistent_workers": n_workers > 0,  # don't re-fork workers every epoch
        "prefetch_factor": 2 if n_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.data.batch_size),
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1, int(cfg.data.batch_size) * 2),
            shuffle=False,
            **loader_kwargs,
        )

    trainer_from_cfg = instantiate(cfg.trainer, _convert_="all")
    model = instantiate(cfg.model, _convert_="all")

    # callbacks / loggers that can't be cleanly expressed in yaml
    tb = TensorBoardLogger(save_dir=str(Path(cfg.paths.log_dir)), name="lightning")
    checkpoint = ModelCheckpoint(
        dirpath=str(Path(cfg.paths.ckpt_dir)),
        filename="sunscc-epoch{epoch:03d}-val_iou{val/iou:.4f}",
        monitor="val/iou" if val_loader else "train/loss",
        mode="max" if val_loader else "min",
        save_top_k=2,
        save_last=True,
        auto_insert_metric_name=False,
    )
    trainer_from_cfg.callbacks = [checkpoint]
    trainer_from_cfg.logger = tb

    trainer_from_cfg.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    log.info("Best checkpoint: %s", checkpoint.best_model_path)


if __name__ == "__main__":
    main()
