"""Convert RDIM binary GFP/RFP outputs into 8-bit RGB images.

Supported input formats:

1) Legacy per-snapshot files: `gfp_rfp_<k>.bin`
   Layout:
       uint64_t total_cells
       float64  total_cells values for GFP
       float64  total_cells values for RFP

2) New append-only per-run file: `fields_f32.bin`
   Header:
       char[8]  magic = "RDIMF32\0"
       uint32   version
       uint32   grid_n
       float64  dt
       uint64   total_cells
   Repeated records:
       uint64   step
       float32  total_cells values for GFP
       float32  total_cells values for RFP
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def read_legacy_snapshot(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return GFP and RFP arrays read from a legacy binary snapshot file."""
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


def iter_fields_f32(path: Path):
    """Yield (step, gfp_2d, rfp_2d) records from fields_f32.bin."""
    with path.open("rb") as fh:
        magic = fh.read(8)
        if magic != b"RDIMF32\x00":
            raise ValueError(f"{path}: invalid magic header")

        version = np.fromfile(fh, dtype=np.uint32, count=1)
        grid_n = np.fromfile(fh, dtype=np.uint32, count=1)
        _dt = np.fromfile(fh, dtype=np.float64, count=1)
        total = np.fromfile(fh, dtype=np.uint64, count=1)

        if version.size != 1 or grid_n.size != 1 or total.size != 1:
            raise ValueError(f"{path}: truncated header")
        if int(version[0]) != 1:
            raise ValueError(f"{path}: unsupported version {int(version[0])}")

        side = int(grid_n[0])
        count = int(total[0])
        if side * side != count:
            raise ValueError(f"{path}: header mismatch grid_n={side}, total={count}")

        while True:
            step = np.fromfile(fh, dtype=np.uint64, count=1)
            if step.size == 0:
                break
            gfp = np.fromfile(fh, dtype=np.float32, count=count)
            rfp = np.fromfile(fh, dtype=np.float32, count=count)
            if gfp.size != count or rfp.size != count:
                raise ValueError(f"{path}: truncated record at step {int(step[0])}")
            yield int(step[0]), gfp.reshape(side, side), rfp.reshape(side, side)


def normalize_to_uint8(field: np.ndarray) -> np.ndarray:
    """Normalize a floating-point field to [0, 255] and return uint8."""
    fmin = float(field.min())
    fmax = float(field.max())
    if fmax <= fmin:
        return np.zeros_like(field, dtype=np.uint8)
    scaled = (field - fmin) / (fmax - fmin)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def save_rgb(gfp: np.ndarray, rfp: np.ndarray, out_path: Path) -> None:
    gfp_u8 = normalize_to_uint8(gfp)
    rfp_u8 = normalize_to_uint8(rfp)

    rgb = np.zeros((gfp.shape[0], gfp.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = rfp_u8
    rgb[..., 1] = gfp_u8
    Image.fromarray(rgb, mode="RGB").save(out_path)


def convert_legacy_snapshot(path: Path, out_dir: Path, suffix: str) -> list[Path]:
    gfp, rfp = read_legacy_snapshot(path)
    out_path = out_dir / f"{path.stem}{suffix}.png"
    save_rgb(gfp, rfp, out_path)
    return [out_path]


def convert_fields_file(path: Path, out_dir: Path, suffix: str) -> list[Path]:
    saved: list[Path] = []
    for idx, (step, gfp, rfp) in enumerate(iter_fields_f32(path)):
        out_path = out_dir / f"{path.stem}_step{step:08d}_idx{idx:05d}{suffix}.png"
        save_rgb(gfp, rfp, out_path)
        saved.append(out_path)
    if not saved:
        raise ValueError(f"{path}: no field records found")
    return saved


def find_binary_inputs(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".bin":
            files.append(p)
        elif p.is_dir():
            legacy = sorted(p.glob("gfp_rfp_*.bin"))
            fields = sorted(p.glob("fields_f32.bin"))
            files.extend(legacy)
            files.extend(fields)
        else:
            raise FileNotFoundError(f"{p} does not exist")
    if not files:
        raise FileNotFoundError("No .bin files found in the provided paths")
    return sorted(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Legacy snapshot .bin files, fields_f32.bin files, or directories containing them.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("png_exports"),
        help="Directory to place PNG images (created if needed).",
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
    binaries = find_binary_inputs(args.inputs)

    cwd = Path.cwd()
    for bin_path in binaries:
        if bin_path.name == "fields_f32.bin":
            saved_paths = convert_fields_file(bin_path, args.output, args.suffix)
            print(f"Saved {len(saved_paths)} images from {bin_path}")
            continue

        saved_paths = convert_legacy_snapshot(bin_path, args.output, args.suffix)
        out_path = saved_paths[0]
        try:
            rel = out_path.relative_to(cwd)
        except ValueError:
            rel = out_path
        print(f"Saved {rel}")


if __name__ == "__main__":
    main()
