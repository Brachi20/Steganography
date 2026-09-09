# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import cv2
import numpy as np


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Batch embedding over a directory of cover images."
    )
    ap.add_argument(
        "--cover_dir",
        default="scripts_with_bug/before-gray",
        help="Path to folder with cover images",
    )
    ap.add_argument(
        "-m",
        "--message_image",
        default="images/secrets/panda.png",
        help="Path to one message image used for all cover images",
    )
    ap.add_argument(
        "--stego_dir",
        default="scripts_with_bug/stego",
        help="Folder to save stego images",
    )
    ap.add_argument(
        "--original_gray_dir",
        default="scripts_with_bug/before-gray",
        help="Folder to save resized grayscale cover images",
    )
    # ארגומנט חדש המאפשר לך לבחור האם לשחזר את הבאג של המאמר
    ap.add_argument(
        "--replicate_bug",
        action="store_true",
        help="Toggle to replicate the paper's Bitwise OR bug",
    )
    return ap.parse_args()


def output_name(input_name: str, suffix: str) -> str:
    stem = Path(input_name).stem
    ext = Path(input_name).suffix or ".png"
    return f"{stem}{suffix}{ext}"


def embed_single(
    c_img: np.ndarray, m_flatten: np.ndarray, replicate_bug: bool
) -> np.ndarray:
    c_flatten = c_img.flatten()
    out = []

    # שלב 3: תהליך הסטגנוגרפיה וההצפנה
    for a, b in zip(c_flatten, m_flatten):
        # המרת ערך הפיקסל למחרוזת בינארית באורך 8 ביטים
        bin_a = np.binary_repr(a, width=8)

        # שליפת הביטים לפי הגדרת המאמר (אינדקס 0 הוא הביט השמיני MSB, אינדקס 7 הוא ה-LSB)
        b8 = int(bin_a[0])
        b7 = int(bin_a[1])
        b6 = int(bin_a[2])

        # ביצוע פעולות ה-XOR על שלושת הביטים המשמעותיים (MSB)
        xor_a = b7 ^ b6
        xor_b = b8 ^ xor_a

        # הזרקת ביט ההודעה (b) בתהליך ה-XOR המשולש
        xor_c = int(b) ^ xor_b

        # בדיקה האם המשתמש ביקש להפעיל את הבאג של כותבי המאמר
        if replicate_bug:
            # השגיאה מהמאמר: ביצוע פעולת OR בין הפיקסל המקורי לביט החדש
            stego_pixel = int(a) | xor_c
        else:
            # הקוד התקין שלך: החלפה נקייה של ביט ה-LSB האחרון בלבד
            save_bin = bin_a[:-1] + str(xor_c)
            stego_pixel = int(save_bin, 2)

        out.append(stego_pixel)

    stego_img_flat = np.array(out, dtype=np.uint8)
    return np.reshape(stego_img_flat, (256, 256))


def main() -> None:
    args = parse_args()

    cover_dir = Path(args.cover_dir)
    stego_dir = Path(args.stego_dir)
    original_gray_dir = Path(args.original_gray_dir)

    if not cover_dir.exists() or not cover_dir.is_dir():
        raise ValueError(f"Cover folder does not exist: {cover_dir}")

    stego_dir.mkdir(parents=True, exist_ok=True)
    original_gray_dir.mkdir(parents=True, exist_ok=True)

    # שלב 2: קריאה ועיבוד תמונת ההודעה (Message) והפיכתה לבינארית נקייה (0 ו-1)
    m_img = cv2.imread(args.message_image, 0)
    if m_img is None:
        raise ValueError(f"Error: Could not read message image: {args.message_image}")
    m_img = cv2.resize(m_img, (256, 256))
    m_img = (m_img > 127).astype(np.uint8)
    m_flatten = m_img.flatten()

    cover_paths = sorted(
        p for p in cover_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )
    if not cover_paths:
        raise ValueError(f"No cover images found in: {cover_dir}")

    print(f"Processing {len(cover_paths)} cover images...")

    for cover_path in cover_paths:
        # שלב 1: קריאה ועיבוד תמונת המקור (Cover) לגודל 256x256 בגווני אפור
        c_img = cv2.imread(str(cover_path), 0)
        if c_img is None:
            print(f"Skipping unreadable cover image: {cover_path}")
            continue

        c_img = cv2.resize(c_img, (256, 256))

        original_path = original_gray_dir / output_name(cover_path.name, "-gray")
        cv2.imwrite(str(original_path), c_img)

        stego_img = embed_single(c_img, m_flatten, args.replicate_bug)

        stego_path = stego_dir / output_name(cover_path.name, "-stego")
        cv2.imwrite(str(stego_path), stego_img)
        print(f"Saved stego image: {stego_path}")

    if args.replicate_bug:
        print("Replicate-bug mode is ON (Bitwise OR).")
    else:
        print("Replicate-bug mode is OFF (clean LSB replacement).")


if __name__ == "__main__":
    main()