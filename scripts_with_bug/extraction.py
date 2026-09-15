# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 00:10:15 2018

@author: YQ
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Batch extraction over a directory of stego images."
    )
    ap.add_argument(
        "--stego_dir",
        default="scripts_with_bug/stego",
        help="Path to folder with stego images",
    )
    ap.add_argument(
        "--recover_dir",
        default="scripts_with_bug/recovered",
        help="Path to folder to save recovered images",
    )
    return ap.parse_args()


def output_name(input_name: str) -> str:
    stem = Path(input_name).stem
    ext = Path(input_name).suffix or ".png"
    if stem.endswith("-stego"):
        stem = stem[:-6]
    return f"{stem}-recovered{ext}"


def extract_single(stego_img: np.ndarray) -> np.ndarray:
    # step 2: change pixel value to binary
    stego_flatten = stego_img.flatten()

    out = []
    for x in stego_flatten:
        x = np.binary_repr(x, width=8)

        # step 3: perform XOR on 7th and 6th bits
        xor_a = int(x[1]) ^ int(x[2])

        # step 4: perform XOR operation on 8th bit with xor_a
        xor_b = int(x[0]) ^ xor_a

        # step 5: perform XOR operations on message bits with 3 MSB
        xor_c = int(x[-1]) ^ xor_b

        out.append(int(xor_c))

    recover_img = np.reshape(np.array(out, dtype=np.uint8), (256, 256))
    recover_img[recover_img == 1] = 255
    return recover_img


def main() -> None:
    args = parse_args()

    stego_dir = Path(args.stego_dir)
    recover_dir = Path(args.recover_dir)

    if not stego_dir.exists() or not stego_dir.is_dir():
        raise ValueError(f"Stego folder does not exist: {stego_dir}")

    recover_dir.mkdir(parents=True, exist_ok=True)

    stego_paths = sorted(
        p for p in stego_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )
    if not stego_paths:
        raise ValueError(f"No stego images found in: {stego_dir}")

    print(f"Processing {len(stego_paths)} stego images...")

    for stego_path in stego_paths:
        # step 1: read stego image
        stego_img = cv2.imread(str(stego_path), 0)
        if stego_img is None:
            print(f"Skipping unreadable stego image: {stego_path}")
            continue

        stego_img = cv2.resize(stego_img, (256, 256))
        recover_img = extract_single(stego_img)

        out_path = recover_dir / output_name(stego_path.name)
        cv2.imwrite(str(out_path), recover_img)
        print(f"Saved recovered image: {out_path}")


if __name__ == "__main__":
    main()