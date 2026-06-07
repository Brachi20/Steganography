import cv2
import numpy as np

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

# הרצה
calculate_metrics("peppers-stego.png", "peppers-gray.png")

# זה הנוסחא המקורית שהשתמשו במאמר על סמך ה MSE ובמאמר הם כתבו סתם נוסחא שגויה
# psnr = 10 * np.log10((255.0 ** 2) / 0.225)
# print(f"PSNR: {psnr:.2f} dB")