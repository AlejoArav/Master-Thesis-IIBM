"""Convert RDIM binary GFP/RFP snapshots into 8-bit RGB images.

The RDIM C simulator writes files called `gfp_rfp_<k>.bin` that contain:

    uint64_t total_cells
    float64  total_cells values for GFP
    float64  total_cells values for RFP

This script normalizes each channel independently to [0, 255] using the
min/max present in that file, places GFP in the green channel, RFP in the red
channel, and zeros out the blue channel. Images are saved as PNG files.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def read_snapshot(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return GFP and RFP arrays read from a binary snapshot file."""
    with path.open("rb") as fh:
        total = np.fromfile(fh, dtype=np.uint64, count=1)
        if total.size != 1:
            raise ValueError(f"{path}: missing lattice size header")
        count = int(total[0])

        gfp = np.fromfile(fh, dtype=np.float64, count=count)
        if gfp.size != count:
            raise ValueError(f"{path}: truncated GFP payload")
        rfp = np.fromfile(fh, dtype=np.float64, count=count)
        if rfp.size != count:
            raise ValueError(f"{path}: truncated RFP payload")

    side = int(round(math.sqrt(count)))
    if side * side != count:
        raise ValueError(f"{path}: lattice is not a perfect square (count={count})")

    return gfp.reshape(side, side), rfp.reshape(side, side)


def normalize_to_uint8(field: np.ndarray) -> np.ndarray:
    """Normalize a floating-point field to [0, 255] and return uint8."""
    fmin = float(field.min())
    fmax = float(field.max())
    if fmax <= fmin:
        return np.zeros_like(field, dtype=np.uint8)
    scaled = (field - fmin) / (fmax - fmin)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def convert_snapshot(path: Path, out_dir: Path, suffix: str) -> Path:
    gfp, rfp = read_snapshot(path)
    gfp_u8 = normalize_to_uint8(gfp)
    rfp_u8 = normalize_to_uint8(rfp)

    rgb = np.zeros((gfp.shape[0], gfp.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = rfp_u8  # Red channel
    rgb[..., 1] = gfp_u8  # Green channel

    stem = path.stem
    out_path = out_dir / f"{stem}{suffix}.png"
    Image.fromarray(rgb, mode="RGB").save(out_path)
    return out_path


def find_snapshots(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".bin":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.glob("gfp_rfp_*.bin")))
        else:
            raise FileNotFoundError(f"{p} does not exist")
    if not files:
        raise FileNotFoundError("No snapshot .bin files found in the provided paths")
    return sorted(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Snapshot .bin files or directories that contain them.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("png_exports"),
        help="Directory to place the PNG images (created if needed).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix inserted before .png (e.g., '_norm').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshots = find_snapshots(args.inputs)
    cwd = Path.cwd()
    for snap in snapshots:
        out_path = convert_snapshot(snap, args.output, args.suffix)
        try:
            rel = out_path.relative_to(cwd)
        except ValueError:
            rel = out_path
        print(f"Saved {rel}")


if __name__ == "__main__":
    main()
