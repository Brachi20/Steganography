import cv2
import numpy as np
from pathlib import Path

def calculate_metrics(original_path, stego_path):
    # טעינת התמונות
    original = cv2.imread(original_path, 0)
    stego = cv2.imread(stego_path, 0)
    
    if original is None or stego is None:
        return "Error: One of the images was not found."

    # הבטחת גודל זהה
    original = cv2.resize(original, (256, 256))
    stego = cv2.resize(stego, (256, 256))

    # --- התיקון הקריטי: המרה ל-float64 ---
    original = original.astype(np.float64)
    stego = stego.astype(np.float64)

    # חישוב MSE
    # כעת החיסור ייתן מספרים שליליים נכונים (כמו 1-) והריבוע יהיה תקין (1)
    mse = np.mean((original - stego) ** 2)

    if mse == 0:
        print("The images are identical (MSE=0)")
        return

    # חישוב PSNR
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    
    print(f"MSE: {mse:.4f}")
    print(f"PSNR: {psnr:.2f} dB")
    return mse, psnr


def build_stego_name(original_path: Path) -> str:
    base_name = original_path.stem
    if base_name.endswith("-gray"):
        base_name = base_name[:-5]
    return f"{base_name}-stego{original_path.suffix}"


def calculate_metrics_for_folders(original_folder, stego_folder):
    original_folder = Path(original_folder)
    stego_folder = Path(stego_folder)

    if not original_folder.is_dir():
        print(f"Error: original folder not found: {original_folder}")
        return

    if not stego_folder.is_dir():
        print(f"Error: stego folder not found: {stego_folder}")
        return

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    original_images = sorted(
        path for path in original_folder.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )

    if not original_images:
        print(f"No image files found in {original_folder}")
        return

    results = []

    for original_path in original_images:
        stego_path = stego_folder / build_stego_name(original_path)

        if not stego_path.is_file():
            print(f"Skipping missing stego image: {stego_path}")
            continue

        print(f"\n{original_path.name} vs {stego_path.name}")
        metrics = calculate_metrics(str(original_path), str(stego_path))
        if metrics is not None:
            mse, psnr = metrics
            results.append((original_path.name, mse, psnr))

    if results:
        avg_mse = sum(item[1] for item in results) / len(results)
        avg_psnr = sum(item[2] for item in results) / len(results)
        print("\nSummary")
        print(f"Images processed: {len(results)}")
        print(f"Average MSE: {avg_mse:.4f}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate MSE/PSNR for two image folders (original vs stego).")
    parser.add_argument("--original_folder", default="images/before-gray",
                        help="folder with original images (default: images/before-gray)")
    parser.add_argument("--stego_folder", default="images/stego",
                        help="folder with stego images (default: images/stego)")
    args = parser.parse_args()

    calculate_metrics_for_folders(args.original_folder, args.stego_folder)

# זה הנוסחא המקורית שהשתמשו במאמר על סמך ה MSE ובמאמר הם כתבו סתם נוסחא שגויה
# psnr = 10 * np.log10((255.0 ** 2) / 0.225)
# print(f"PSNR: {psnr:.2f} dB")