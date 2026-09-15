import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch K-Triple-XOR-LSB extraction from all stego images."
    )
    parser.add_argument("--stego_dir", required=True)
    parser.add_argument("--output_base", default="images/recovered")
    parser.add_argument("-k", "--k_bits", type=int, default=2)

    parser.add_argument("--secret_width", type=int, required=True)
    parser.add_argument("--secret_height", type=int, required=True)

    return parser.parse_args()


def extract_name(stego_name: str) -> str:
    if "-stego" in stego_name:
        return stego_name.replace("-stego", "-recovered")
    stem = Path(stego_name).stem
    return f"{stem}-recovered.png"


def extract_k_triple_xor_lsb(stego: np.ndarray, k_bits: int) -> np.ndarray:
    mask = (1 << k_bits) - 1

    embedded_bits = stego & mask

    bit7 = (stego >> 7) & 1
    bit6 = (stego >> 6) & 1
    bit5 = (stego >> 5) & 1
    xor_key = bit7 ^ bit6 ^ bit5

    recovered_k_bits = embedded_bits ^ (xor_key * mask)

    recovered = recovered_k_bits << (8 - k_bits)

    return recovered.astype(np.uint8)


def main() -> None:
    args = parse_args()

    if args.k_bits < 1 or args.k_bits > 5:
        raise ValueError("k_bits must be between 1 and 5 for Triple-XOR LSB.")

    if args.secret_width <= 0 or args.secret_height <= 0:
        raise ValueError("secret_width and secret_height must be positive.")

    stego_dir = Path(args.stego_dir)
    if not stego_dir.exists() or not stego_dir.is_dir():
        raise ValueError(f"Stego folder does not exist: {stego_dir}")

    stego_paths = sorted(
        p for p in stego_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )

    if not stego_paths:
        raise ValueError(f"No images found in: {stego_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_base) / f"kxorlsb_k{args.k_bits}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting with k={args.k_bits} from {len(stego_paths)} images...")
    print(f"Secret size: {args.secret_width}x{args.secret_height}")
    print(f"Output folder: {output_dir}")

    written = 0

    for stego_path in stego_paths:
        stego = cv2.imread(str(stego_path), cv2.IMREAD_GRAYSCALE)
        if stego is None:
            raise ValueError(f"Could not read stego image: {stego_path}")

        # if args.secret_width > stego.shape[1] or args.secret_height > stego.shape[0]:
        #     raise ValueError(
        #         f"Secret size {args.secret_width}x{args.secret_height} "
        #         f"cannot be larger than stego size {stego.shape[1]}x{stego.shape[0]} "
        #         f"for image {stego_path.name}"
        #     )

        recovered_full = extract_k_triple_xor_lsb(stego, args.k_bits)

        recovered = recovered_full[:args.secret_height, :args.secret_width]

        out_path = output_dir / extract_name(stego_path.name)
        if not cv2.imwrite(str(out_path), recovered):
            raise RuntimeError(f"Failed to write recovered image: {out_path}")

        written += 1

    print(f"Done. Wrote {written} recovered images.")


if __name__ == "__main__":
    main()