import cv2
import numpy as np

def generate_difference_image(original_path, stego_path, output_path="difference_result.png"):
    # טעינת התמונות
    original = cv2.imread(original_path)
    stego = cv2.imread(stego_path)

    if original is None or stego is None:
        print("Error: One of the images was not found.")
        return

    # וידוא שהתמונות באותו גודל (המאמר מציין שהן צריכות להיות באותו גודל) [cite: 69, 129]
    if original.shape != stego.shape:
        stego = cv2.resize(stego, (original.shape[1], original.shape[0]))

    # חישוב ההפרש המוחלט בין התמונות
    # הפונקציה מחשבת |A - B| עבור כל פיקסל
    diff = cv2.absdiff(original, stego)

    # בגלל שההבדלים בסטגנוגרפיה (כמו LSB) הם מזעריים (ערך של 1 או 0), 
    # התמונה תיראה שחורה לגמרי. כדי שנראה משהו בעיניים, נבצע "נרמול":
    # נכפיל את ההבדלים כדי להפוך אותם לבולטים (למשל פי 50)
    enhanced_diff = diff * 50

    # שמירת התוצאות
    cv2.imwrite(output_path, diff)
    cv2.imwrite("enhanced_difference.png", enhanced_diff)

    # הצגת התמונות
    cv2.imshow('Absolute Difference (Original)', diff)
    cv2.imshow('Enhanced Difference (Visible)', enhanced_diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Difference images saved as {output_path} and enhanced_difference.png")

# הרצה
generate_difference_image("lena-gray.png", "lena-stego.png", "lena-difference.png")