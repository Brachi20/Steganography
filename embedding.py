# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cover_image", required=True, help="path to cover image")
ap.add_argument("-m", "--message_image", required=True, help="path to message image")
ap.add_argument("-s", "--stego_image", required=True, help="path to save stego image")
# ארגומנט חדש לשמירת תמונת המקור בשחור-לבן
ap.add_argument("-o", "--original_gray", required=True, help="path to save original grayscale image")
args = vars(ap.parse_args())

# שלב 1: קריאה ועיבוד תמונת המקור (Cover)
c_img = cv2.imread(args["cover_image"], 0) # קריאה כגווני אפור
if c_img is None:
    print("Error: Could not read cover image")
    exit()

c_img = cv2.resize(c_img, (256, 256))

# שמירת תמונת המקור בשחור-לבן לפני השינויים
cv2.imwrite(args["original_gray"], c_img)
print(f"Original grayscale image saved at: {args['original_gray']}")

# עיבוד תמונת ההודעה
m_img = cv2.imread(args["message_image"], 0)
if m_img is None:
    print("Error: Could not read message image")
    exit()
m_img = cv2.resize(m_img, (256, 256))
m_img[m_img > 0] = 1 

# שלב 2: תחילת תהליך הסטגנוגרפיה (ללא שינוי מהקוד הקודם)
c_flatten = c_img.flatten()
m_flatten = m_img.flatten()

out = []
for a, b in zip(c_flatten, m_flatten):
    bin_a = np.binary_repr(a, width=8)
    
    # לוגיקת ה-XOR שלך
    xor_a = int(bin_a[1]) ^ int(bin_a[2])
    xor_b = int(bin_a[0]) ^ xor_a
    xor_c = int(b) ^ xor_b 
    
    save_bin = bin_a[:-1] + str(xor_c)
    out.append(int(save_bin, 2))

# שלב 3: המרה חזרה למערך תמונה ושמירת תמונת הסטגנו
stego_img_flat = np.array(out, dtype=np.uint8)
stego_img = np.reshape(stego_img_flat, (256, 256))

cv2.imwrite(args["stego_image"], stego_img)
print(f"Stego image saved at: {args['stego_image']}")