# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import cv2


def resize_folder(input_dir: Path, output_dir: Path, size=(256, 256)):
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    for p in image_files:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Skipping unreadable file: {p}")
            continue
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        out_path = output_dir / p.name
        cv2.imwrite(str(out_path), resized)
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Resize all images in a folder to 256x256 and save to a new folder.")
    parser.add_argument("--input_dir", default="images/colored", help="input folder (default: images/colored)")
    parser.add_argument("--output_dir", default="images/colored_resized", help="output folder (default: images/colored_resized)")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return

    resize_folder(input_dir, output_dir, size=(args.width, args.height))


if __name__ == "__main__":
    main()
