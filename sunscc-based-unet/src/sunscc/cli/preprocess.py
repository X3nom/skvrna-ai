"""Batch FITS -> 16-bit PNG preprocessor with pure-gain stretch and sidecar CSV.

    sunscc-preprocess /data/images -o /data/images_png --meta-csv /data/meta/images.csv -j 32
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from sunscc.preprocess import preprocess_batch

log = logging.getLogger("sunscc.preprocess")


def build_parser():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_dir", type=Path, help="root dir containing FITS (recursive)")
    p.add_argument("-o", "--output-dir", type=Path, required=True,
                   help="output dir for PNGs (flat, created if needed)")
    p.add_argument("--meta-csv", type=Path, required=True,
                   help="sidecar CSV (stem, solar_r, stretch_hi, ...)")
    p.add_argument("-j", "--jobs", type=int, default=1, help="parallel workers")
    p.add_argument("--overwrite", action="store_true", help="reprocess existing PNGs")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.input_dir.is_dir():
        log.error("input_dir not found: %s", args.input_dir)
        return 1

    log.info("scanning %s ...", args.input_dir)
    ok, rotten = preprocess_batch(
        args.input_dir, args.output_dir, args.meta_csv,
        jobs=args.jobs, overwrite=args.overwrite,
    )
    print(f"{ok} ok, {rotten} rotten")
    return 0 if rotten == 0 else 1


if __name__ == "__main__":
    sys.exit(main())