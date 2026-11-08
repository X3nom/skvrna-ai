# SunSCC Segmentation Reimplementation — Knowledge Base

> Purpose: keep the implementation faithful to the paper and don't drift off-track.
> Source of truth: Sayez et al. 2023, "SunSCC: Segmenting, Grouping and Classifying
> Sunspots From Ground-Based Observations Using Deep Learning",
> JGR Space Physics, doi:10.1029/2023JA031548 (plaintext copy in ~/Downloads).
> Original (partial) repo: `sayez-SunSCC-4cc85af/` — depends on the **private `bioblue`**
> package which we replace with plain PyTorch + Lightning + Hydra.

---

## 1. Scope

**Backend**: PyTorch (user OK with it; chosen over Keras for Lightning parity with
orig bioblue/Hydra stack and first-class ONNX export later).

**Focus**: Segmentation stage only (binary sunspot mask / foreground-background "fgbg").
Grouping (mean-shift) and classification (ResNet34+MLPs) are out of scope **for now**.

**Target hardware**: local MX550 (2 GB VRAM). Scale batch size / accumulate grads,
**but keep 512x512 patch size true to the paper**. May move to RTX 3060 later.

**Pipeline (paper Fig. 1)**:
```
2048x2048 USET WL image ──4x4 grid──> 16x 512x512 segments ──> UNet ──> binary masks
──> reassemble 2048x2048 mask ──(later: mean-shift grouping ──> McIntosh classification)
```

---

## 2. Data

- **Source**: USET White Light, SIDC. Correct mirror source:
  `https://www.sidc.be/uset/data/cameras/output_img/FTS/L1c/USET_White_Light/YYYY/MM/UPH*.FTS`
  (user is mirroring SIDC L1c/USET_White_Light — CONFIRMED correct: naming convention
  `UPH20130113150018.FTS` matches repo expectation exactly).
- **Format**: FITS, 8 MB each. Images 2048x2048 (pre-June-2008 are 1024x1024, upsampled
  NN to match; paper does this).
- **Training set WLSA**: 2013-2015, 5953 images, annotated AUTOMATICALLY with
  pseudo-labels (top-hat thresholding).
- **Validation set WLSM**: 36 images (1/month over 2013-15) with MANUAL annotation.
  **We do NOT have these** (SOLARNET-registered only).
  → Our approach: **pseudo-manual validation** — run the pseudo-label generator on the
  held-out images, then manually edit the masks where wrong. Use these edited masks for
  validation.
- **Split rule (paper)**: validation images exclude snapshots taken the SAME day or the
  day before/after from training set. Keep the temporal split; exact files configurable later.
- **Later data** can be downloaded for clustering/classification
  (2002-2019), but segmentation only needs 2013-15.

## 3. Pseudo-label generation (paper §4.2.1, Fig. 2) — the "paper method"

For each RAW WL image:
1. **Limb darkening correction**:
   - polar projection, fit a polynomial of mean radial intensity (center→limb)
   - build a radial-progression mask, divide raw image by it → flattened image
2. **Black top-hat transform** (`skimage.morphology.black_tophat`): finds dark spots
   (sunspots) on brighter surround. Structuring element: disk (radius ~5 px as starting
   guess; original exact value is in the lost bioblue code — TUNE on examples).
3. **Threshold τ** → binary mask. Mask = tophat image > τ.
   - Paper's FINAL model thresholds: **τ1=325, τ2=375, τ3=425** (3 = T).
   - Repo also holds a sweep 250..500 step 25 (11 thresholds) for experiments.
     **User decision: pre-generate ALL thresholds, use just the 3 for training.**
4. Output PNG, same basename as FITS (`UPH...FTS` → `UPH....png`).
5. **OPEN PROBLEM (threshold units) — now RESOLVED empirically (2026-11-05)**:
   USET L1c WL images have quiet-sun intensity ~2400-3300 counts (16-bit source,
   BZERO=32768 handled by astropy; NOT 0-65535). On the limb-corrected raw scale,
   the paper's threshold values (250-500) produce sensible masks (calibrated on
   sample `UPH20130101133418.FTS`): T250 catches spots + a few faint specks,
   T325-T500 progressively stricter while keeping the AR group -> the paper's
   integer thresholds work directly on OUR scale. Defaults: footprint (closing SE)
   radius 25 px, min spot area 8 px. Preview:
   /tmp/opencode/preview_uph20130101.png.
6. Implementation note: skimage black_tophat with disk(8) CANNOT fill spots larger
   than the SE (closing fails inside big sunspot groups) - we use
   cv2.morphologyEx(MORPH_CLOSE, elliptical SE) instead: fast + handles big SEs.

**IMPORTANT**: pseudo-labels are made on the LIMB-CORRECTED image, but the **CNN input
is the RAW image** (paper §4.2.2: "limb darkening effect is NOT corrected on the images
given as input to the CNN; raw white light images are used"). The CNN is expected to
regularize against limb darkening + clouds.

Rationale for multiple thresholds (paper §4.2.2): finding ONE good threshold is
non-trivial (clouds, seeing, per-image illumination). Training on one fixed threshold
teaches the CNN systematic labeling errors. Selecting among 3 stochastic pseudo-labels
increases error-pattern variety and prevents the CNN from learning them.

## 4. Training (paper §4.2.2) — STOCHASTIC PSEUDO-LABELS (not multi-head!)

Earlier plan assumed 3-channel output — **WRONG**. Corrected:
- **Architecture**: single UNet, single **binary output** (1 logit channel).
- **Per image I, pseudo-label set** `M_I = {m_τ1, m_τ2, m_τ3}`
- **Each training step**: pick one mask from M_I at random (equal probability),
  compute binary cross-entropy against it.
- Loss: **sum of binary cross-entropy + log-cosh Dice loss**.
- Optimizer: **AdamW, initial lr = 1e-4**.
- Epochs: **40**, batch size **16** (we downscale for 2 GB VRAM with accum).

## 5. Architecture (paper §4.1 / Fig. 1)

- UNet with **5 levels**: filters 32, 64, 128, 256, 512 (encoder, root→deep);
  4 downsampling steps in encoder; reverse (512...32) in decoder, 4 upsampling steps.
- Each level = **2 convolutional blocks** (lum = double-conv).
- Patch size 512x512, full original resolution (small spots disappear when downsampled).
- Provide a `width_scale` (e.g. 0.5 → 16/32/64/128/256) ONLY as escape hatch for
  2 GB VRAM; default 1.0 = paper-faithful.

## 6. Evaluation (paper §5.1)

- Are treated as **DETECTION** problems, not pixel metrics: connected components in GT
  vs predicted mask; TP / FP / FN per sunspot-object basis → **F1** (Eq. 2):
  `F1 = 2TP / (2TP + FN + FP)`.
- **Size bins (Table 2)**: bin0 <50 px, bin1 50-100, bin2 100-400, bin3 400-800, bin4 >800.
- Second metric: **fusions** (how many sunspots got merged in one detection) vs missed.
- Paper results for reference: best threshold-only seg F1(bin0) ≈ 55%; CNN-1 up to 71%;
  CNN-3 (stochastic) comparable to best CNN-1, fewer fusions.

## 7. Validation (our adapted approach)

User decision: **pseudo-manual validation**
1. Pick ~36 held-out images (1/month of 2013-15, ±1 day exclusions like paper).
2. Run the pseudo-label generator (e.g. central threshold 375) on them.
3. Manually EDIT the masks where wrong (e.g. in napari) → near-GT masks.
4. Validate detection-F1 on these edited masks.
No W&B — TensorBoard local only (FOSS).

## 8. Project layout (implemented in this repo)

```
skvrna-ai/                      (uv project root; Dockerfile/compose pre-existing — DO NOT TOUCH)
├── pyproject.toml              (deps: torch, lightning, hydra-core, albumentations, astropy, sunpy, scikit-image, opencv-headless, tensorboard)
├── sunscc-based-unet/
│   ├── knowledge.md            (this file)
│   ├── src/sunscc/
│   │   ├── pseudolabels.py     (limb correction, black tophat, thresholds — standalone utility)
│   │   ├── dataset.py          (PseudoLabelDataset: raw img + mask set; random mask per step)
│   │   ├── unet.py             (paper UNet, width_scale param)
│   │   ├── lightning_module.py (BCE + log-cosh Dice, AdamW 1e-4)
│   │   ├── metrics.py          (detection F1 per size bin, fusion counts)
│   │   └── cli/
│   │       ├── generate_masks.py  (argparse: single FITS in → mask PNGs out)
│   │       ├── train.py           (hydra entrypoint)
│   │       └── predict.py         (2048 → 4x4 patches → reassemble)
│   └── configs/                (hydra yaml: model, data, trainer per VRAM)
├── dataset-stuff/              (user's bulk download scripts)
└── sayez-SunSCC-4cc85af/       (original repo for reference — read-only)
```

Data root via env var `SUNSCC_DATA_ROOT` / hydra config (never hardcode).

## 9. Gotchas gathered from the original repo / data

- FITS headers contain `SOLAR_R` (solar radius in px) — solar disk center at image center.
- pre-2005 images are flipped vertically (see `process_one_image` flip_time 2003-03-08)
  — irrelevant for 2013-15 data but note for later years.
- Original masks dirs named like `2023_T425_fgbg` / `T425-T375-T325_fgbg`
  (T-prefix = "Threshold").
- Some images are corrupted ("rotten_list" in repo, indices); needs a robust skip /
  try-except when scanning.
- `tqdm.notebook` / jupyter deps in utilities — ignore, we use plain tqdm.
- The original bioblue exp expects hydra cfg with dataset.dtypes e.g.
  `['image', 'T425-T375-T325_fgbg']` — informs naming of our pseudo-label dirs.

## 10. Environment notes

- uv project at repo root; `uv sync --extra dev` installs `.venv` (host-local).
- **torch pinned cu124 wheels** (`[[tool.uv.index]] pytorch-cu124`): host NVIDIA
  driver supports CUDA 12.2; default PyPI torch 2.15 ships CUDA 13 which cannot
  run on it ("NVIDIA driver too old"). torch<2.7 keeps cu12x.
- MX550 currently DISABLED for battery saving on this laptop: nvidia-smi does not
  work locally; train with CPU for dry-runs, or run GPU training in docker/on the
  RTX 3060 server. Driver update may be needed whenever GPU training resumes.
- docker-compose GPUs commented out (pre-existing, left untouched).

## 11. Deferred (explicitly OUT of scope now)

- MeanShift grouping (repo's `MeanShift.py`, `clustering_utilities.py` are complete
  reference implementations when we get there; kernel params h=0.1, wmax=0.4 rad,
  amax=200 µHem from paper App. A/B).
- Classification (ResNet34 + 3 MLPs, hierarchical training, ensemble voting).
- ONNX export + integration.
- Docker moves (container already exists at repo root for jupyter; user handles it).
