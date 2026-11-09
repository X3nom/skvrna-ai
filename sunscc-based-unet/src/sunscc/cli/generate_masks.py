"""Standalone pseudo-label mask generator.

Top-level function generate_pseudolabel(input_path, output_path, threshold) lives in
sunscc.pseudolabels; this CLI wraps it for directory sweeps:

    sunscc-generate-masks IN [IN ...] --output-dir DIR [--thresholds 325 375 425]
    sunscc-generate-masks /data/images_png --meta-csv /data/meta/images.csv -o /data/masks
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from sunscc.pseudolabels import generate_pseudolabel, threshold_sweep

log = logging.getLogger("sunscc.generate_masks")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path, help="FITS/PNG file(s) or directory of FITS/PNG")
    p.add_argument("-o", "--output-dir", type=Path, required=True,
                   help="where to write '<stem>.png' masks (one subfolder per threshold "
                        "named T<int>_fgbg, mirroring the original repo layout)")
    p.add_argument("--meta-csv", type=Path, default=None,
                   help="sidecar CSV for PNG inputs (required if inputs are PNG)")
    p.add_argument("-t", "--thresholds", type=str, nargs="*", default=None,
                   help="threshold values (default: paper 325 375 425; "
                        "'--thresholds all' sweeps 250..500 step 25)")
    p.add_argument("--footprint", type=int,
                   help="top-hat structuring element radius (pixels)")
    p.add_argument("--min-area", type=int, help="minimum detected spot area (px)")
    p.add_argument("-j", "--jobs", type=int, default=4, help="parallel workers")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def collect_inputs(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            files += sorted(item.glob("**/*.FTS")) + sorted(item.glob("**/*.fts"))
            files += sorted(item.glob("**/*.PNG")) + sorted(item.glob("**/*.png"))
        elif item.is_file():
            files.append(item)
        else:
            log.warning("input %s not found", item)
    return files


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    thresholds = args.thresholds
    if thresholds is None:
        thresholds = [325.0, 375.0, 425.0]
    elif len(thresholds) == 1 and str(thresholds[0]).lower() == "all":
        thresholds = [float(t) for t in threshold_sweep()]
    else:
        thresholds = [float(t) for t in thresholds]

    files = collect_inputs([*args.inputs])
    if not files:
        log.error("no input files found")
        return 1
    log.info("%d files -> %s  (thresholds: %s)", len(files), args.output_dir, thresholds)

    out_dirs = {t: args.output_dir / f"T{int(t)}_fgbg" for t in thresholds}
    kwargs = {}
    if args.footprint is not None:
        kwargs["footprint_radius"] = int(args.footprint)
    if args.min_area is not None:
        kwargs["min_area"] = int(args.min_area)
    if args.meta_csv is not None:
        kwargs["meta_csv"] = args.meta_csv

    tasks = []
    for path in files:
        for t, out_dir in out_dirs.items():
            tasks.append((path, out_dir / f"{path.stem}.png", t))

    if args.jobs <= 1:
        for path, out_path, t in tasks:
            try:
                generate_pseudolabel(path, out_path, threshold=t, overwrite=False, **kwargs)
            except Exception:
                log.exception("FAILED %s T%s", path.name, t)
        return 0

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(generate_pseudolabel, path, out_path, threshold=t, overwrite=False, **kwargs): (path, t)
            for (path, out_path, t) in tasks
        }
        n_failed = 0
        for fut in as_completed(futures):
            path, t = futures[fut]
            try:
                fut.result()
            except Exception:
                n_failed += 1
                log.exception("FAILED %s T%s", path.name, t)
        if n_failed:
            log.warning("%d/%d tasks failed", n_failed, len(tasks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
