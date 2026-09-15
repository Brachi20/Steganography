# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path

from embedding_rgb import embed_rgb


def build_output_name(cover_path: Path) -> str:
    return f"{cover_path.stem}-stego-color{cover_path.suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed the same Panda secret image into all RGB cover images in a folder."
    )
    parser.add_argument(
        "--cover_dir",
        default="images/colored_resized",
        help="folder containing RGB cover images (default: images/colored_resized)",
    )
    parser.add_argument(
        "--secret_image",
        default="images/secrets/panda.png",
        help="path to the secret Panda image (default: images/secrets/panda.png)",
    )
    parser.add_argument(
        "--output_dir",
        default="images/stego_color",
        help="folder to save stego images (default: images/stego_color)",
    )
    args = parser.parse_args()

    cover_dir = Path(args.cover_dir)
    secret_image = Path(args.secret_image)
    output_dir = Path(args.output_dir)

    if not cover_dir.is_dir():
        raise FileNotFoundError(f"Could not find cover directory: {cover_dir}")

    if not secret_image.is_file():
        raise FileNotFoundError(f"Could not find secret image: {secret_image}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cover_images = sorted(
        path for path in cover_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    if not cover_images:
        raise FileNotFoundError(f"No image files found in {cover_dir}")

    for cover_image in cover_images:
        output_path = output_dir / build_output_name(cover_image)
        embed_rgb(str(cover_image), str(secret_image), str(output_path))


if __name__ == "__main__":
    main()