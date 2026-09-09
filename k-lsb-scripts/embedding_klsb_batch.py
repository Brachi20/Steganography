import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch K-Triple-XOR-LSB embedding for all images in a folder."
    )
    parser.add_argument("--cover_dir", default="images/before-gray")
    parser.add_argument("--secret_image", default="images/secrets/secret.png")
    parser.add_argument("--output_base", default="images/stego")
    parser.add_argument("-k", "--k_bits", type=int, default=2)

    parser.add_argument("--secret_width", type=int, required=True)
    parser.add_argument("--secret_height", type=int, required=True)

    return parser.parse_args()


def load_gray_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def stego_name(cover_name: str) -> str:
    if "-gray" in cover_name:
        return cover_name.replace("-gray", "-stego")
    stem = Path(cover_name).stem
    return f"{stem}-stego.png"


def embed_k_triple_xor_lsb(
    cover: np.ndarray,
    secret_canvas: np.ndarray,
    k_bits: int
) -> np.ndarray:
    mask = (1 << k_bits) - 1

    secret_k_bits = secret_canvas >> (8 - k_bits)

    bit7 = (cover >> 7) & 1
    bit6 = (cover >> 6) & 1
    bit5 = (cover >> 5) & 1
    xor_key = bit7 ^ bit6 ^ bit5

    embedded_bits = secret_k_bits ^ (xor_key * mask)

    cover_cleared = cover & np.uint8(0xFF ^ mask)
    stego = cover_cleared | embedded_bits

    return stego.astype(np.uint8)


def main() -> None:
    args = parse_args()

    if args.k_bits < 1 or args.k_bits > 5:
        raise ValueError("k_bits must be between 1 and 5 for Triple-XOR LSB.")

    if args.secret_width <= 0 or args.secret_height <= 0:
        raise ValueError("secret_width and secret_height must be positive.")

    cover_dir = Path(args.cover_dir)
    if not cover_dir.exists() or not cover_dir.is_dir():
        raise ValueError(f"Cover folder does not exist: {cover_dir}")

    secret_image = load_gray_image(Path(args.secret_image))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_base) / f"kxorlsb_k{args.k_bits}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cover_paths = sorted(
        p for p in cover_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )

    if not cover_paths:
        raise ValueError(f"No images found in: {cover_dir}")

    print(f"Embedding with k={args.k_bits} into {len(cover_paths)} images...")
    print(f"Secret size: {args.secret_width}x{args.secret_height}")
    print(f"Output folder: {output_dir}")

    written = 0

    for cover_path in cover_paths:
        cover = load_gray_image(cover_path)

        # if args.secret_width > cover.shape[1] or args.secret_height > cover.shape[0]:
        #     raise ValueError(
        #         f"Secret size {args.secret_width}x{args.secret_height} "
        #         f"cannot be larger than cover size {cover.shape[1]}x{cover.shape[0]} "
        #         f"for image {cover_path.name}"
        #     )

        resized_secret = cv2.resize(
            secret_image,
            (args.secret_width, args.secret_height),
            interpolation=cv2.INTER_AREA,
        )

        secret_canvas = np.zeros_like(cover, dtype=np.uint8)
        secret_canvas[:args.secret_height, :args.secret_width] = resized_secret

        stego = embed_k_triple_xor_lsb(cover, secret_canvas, args.k_bits)

        out_path = output_dir / stego_name(cover_path.name)
        if not cv2.imwrite(str(out_path), stego):
            raise RuntimeError(f"Failed to write stego image: {out_path}")

        written += 1

    print(f"Done. Wrote {written} stego images.")


if __name__ == "__main__":
    main()