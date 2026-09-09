import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MSE/PSNR for all matched images in two folders."
    )
    parser.add_argument(
        "--original_dir",
        default="scripts_with_bug/before-gray",
        help="Folder containing original cover images.",
    )
    parser.add_argument(
        "--stego_dir",
        default="scripts_with_bug/stego",
        help="Folder containing stego images to compare.",
    )
    parser.add_argument(
        "--csv_path",
        default="",
        help="Optional CSV output path. If omitted, a file is created in stego_dir.",
    )
    return parser.parse_args()


def normalized_key(name: str) -> str:
    stem = Path(name).stem.lower()
    for suffix in ("-gray", "-stego", "-recovered"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def calculate_mse_psnr(original: np.ndarray, stego: np.ndarray) -> Tuple[float, float]:
    if original.shape != stego.shape:
        stego = cv2.resize(stego, (original.shape[1], original.shape[0]))

    original_f = original.astype(np.float64)
    stego_f = stego.astype(np.float64)
    mse = float(np.mean((original_f - stego_f) ** 2))

    if mse == 0.0:
        return 0.0, float("inf")

    psnr = float(10.0 * np.log10((255.0**2) / mse))
    return mse, psnr


def build_index(folder: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
            index[normalized_key(path.name)] = path
    return index


def safe_psnr_text(psnr: float) -> str:
    if np.isinf(psnr):
        return "inf"
    return f"{psnr:.4f}"


def write_csv(csv_path: Path, rows: List[List[str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "original", "stego", "mse", "psnr_db"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    original_dir = Path(args.original_dir)
    stego_dir = Path(args.stego_dir)

    if not original_dir.exists() or not original_dir.is_dir():
        raise ValueError(f"Original folder does not exist: {original_dir}")
    if not stego_dir.exists() or not stego_dir.is_dir():
        raise ValueError(f"Stego folder does not exist: {stego_dir}")

    original_index = build_index(original_dir)
    stego_index = build_index(stego_dir)
    common_keys = sorted(set(original_index.keys()) & set(stego_index.keys()))

    if not common_keys:
        raise ValueError("No matching image names between original_dir and stego_dir.")

    rows: List[List[str]] = []
    mse_values: List[float] = []
    psnr_values: List[float] = []

    for key in common_keys:
        original_path = original_index[key]
        stego_path = stego_index[key]

        original = read_gray(original_path)
        stego = read_gray(stego_path)

        mse, psnr = calculate_mse_psnr(original, stego)
        mse_values.append(mse)
        if not np.isinf(psnr):
            psnr_values.append(psnr)

        rows.append(
            [
                key,
                str(original_path),
                str(stego_path),
                f"{mse:.6f}",
                safe_psnr_text(psnr),
            ]
        )

        print(f"{key:20s} MSE={mse:.6f} PSNR={safe_psnr_text(psnr)} dB")

    avg_mse = float(np.mean(mse_values)) if mse_values else 0.0
    avg_psnr: Optional[float] = float(np.mean(psnr_values)) if psnr_values else None

    print("-" * 64)
    if avg_psnr is None:
        print(f"Average on {len(common_keys)} images -> MSE={avg_mse:.6f}, PSNR=inf")
    else:
        print(
            f"Average on {len(common_keys)} images -> "
            f"MSE={avg_mse:.6f}, PSNR={avg_psnr:.4f} dB"
        )

    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = stego_dir / f"metrics_{timestamp}.csv"

    write_csv(csv_path, rows)
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
