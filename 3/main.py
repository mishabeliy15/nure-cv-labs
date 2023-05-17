import argparse

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser(description="Impulse view")
    args.add_argument("-i", "--img", type=str, help="Path to img", default="./img.png", required=False)
    args.add_argument("-k", type=float, help="K", default=1.2, required=False)

    args = args.parse_args()
    return args


def to_impulse(img: np.ndarray, coef: float) -> np.ndarray:
    bmp = img.mean(axis=-1)
    sum_pixels = 0
    max_pixel = np.amax(bmp)
    final = np.zeros_like(bmp)

    for i in range(bmp.shape[0]):
        for j in range(bmp.shape[1]):
            sum_pixels += bmp[i][j]
            if sum_pixels >= coef * max_pixel:
                final[i][j] = 1
                sum_pixels -= coef * max_pixel

    return final


def main():
    args = parse_args()
    img = cv2.imread(args.img)
    b = to_impulse(img, args.k)
    cv2.imshow("Image", b)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
