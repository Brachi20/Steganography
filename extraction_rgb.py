# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np


def extract_rgb(stego_image_path, recover_image_path):
    stego_image = cv2.imread(stego_image_path, cv2.IMREAD_COLOR)
    if stego_image is None:
        raise FileNotFoundError(f"Could not read stego image: {stego_image_path}")

    stego_flatten = stego_image.flatten()

    out = []
    for pixel_value in stego_flatten:
        bin_pixel = np.binary_repr(pixel_value, width=8)

        xor_a = int(bin_pixel[1]) ^ int(bin_pixel[2])
        xor_b = int(bin_pixel[0]) ^ xor_a
        xor_c = int(bin_pixel[-1]) ^ xor_b

        out.append(int(xor_c))

    recover_image = np.reshape(np.array(out, dtype=np.uint8), (256, 256, 3))
    recover_image[recover_image == 1] = 255
    cv2.imwrite(recover_image_path, recover_image)
    print(f"Recovered image saved at: {recover_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract an RGB secret image from an RGB stego image.")
    parser.add_argument(
        "-s",
        "--stego_image",
        default="images/stego/lena-stego-color.png",
        help="path to stego image (default: images/stego/lena-stego-color.png)",
    )
    parser.add_argument(
        "-r",
        "--recover_image",
        default="images/secrets/recovered-color.png",
        help="path to save recovered image (default: images/secrets/recovered-color.png)",
    )
    args = parser.parse_args()

    extract_rgb(args.stego_image, args.recover_image)
