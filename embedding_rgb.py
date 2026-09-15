# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np


def embed_rgb(cover_image_path, message_image_path, stego_image_path):
    cover_image = cv2.imread(cover_image_path, cv2.IMREAD_COLOR)
    if cover_image is None:
        raise FileNotFoundError(f"Could not read cover image: {cover_image_path}")

    message_image = cv2.imread(message_image_path, cv2.IMREAD_COLOR)
    if message_image is None:
        raise FileNotFoundError(f"Could not read message image: {message_image_path}")

    cover_image = cv2.resize(cover_image, (256, 256))
    message_image = cv2.resize(message_image, (256, 256))
    message_image[message_image > 0] = 1

    cover_flatten = cover_image.flatten()
    message_flatten = message_image.flatten()

    out = []
    for cover_pixel, message_bit in zip(cover_flatten, message_flatten):
        bin_cover = np.binary_repr(cover_pixel, width=8)

        xor_a = int(bin_cover[1]) ^ int(bin_cover[2])
        xor_b = int(bin_cover[0]) ^ xor_a
        xor_c = int(message_bit) ^ xor_b

        save_bin = bin_cover[:-1] + str(xor_c)
        out.append(int(save_bin, 2))

    stego_image = np.reshape(np.array(out, dtype=np.uint8), (256, 256, 3))
    cv2.imwrite(stego_image_path, stego_image)
    print(f"Stego image saved at: {stego_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed an RGB secret image into an RGB cover image.")
    parser.add_argument(
        "-c",
        "--cover_image",
        default="images/colored/lena.png",
        help="path to cover image (default: images/colored/lena.png)",
    )
    parser.add_argument(
        "-m",
        "--message_image",
        default="images/colored/peppers.png",
        help="path to message image (default: images/colored/peppers.png)",
    )
    parser.add_argument(
        "-s",
        "--stego_image",
        default="images/stego/lena-stego-color.png",
        help="path to save stego image (default: images/stego/lena-stego-color.png)",
    )
    args = parser.parse_args()

    embed_rgb(args.cover_image, args.message_image, args.stego_image)
