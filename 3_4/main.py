import argparse

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser(description="Impulse view")
    args.add_argument("-i", "--img", type=str, help="Path to img", default="./data/img_1.png", required=False)
    args.add_argument("-k", type=float, help="K", default=1, required=False)

    args = args.parse_args()
    return args


def to_impulse(img: np.ndarray, coef: float) -> np.ndarray:
    bmp = img.sum(axis=-1)
    sum_pixels = 0
    max_pixel = np.amax(bmp)
    final = np.zeros_like(bmp, dtype=np.uint8)

    for i in range(bmp.shape[0]):
        for j in range(bmp.shape[1]):
            if sum_pixels >= coef * max_pixel:
                final[i][j] = 1
                sum_pixels -= coef * max_pixel
            else:
                sum_pixels += bmp[i][j]


    final *= 255
    backtorgb = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
    return backtorgb


def find_contours(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.blur(image, (5, 5))
    image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)[1]
    image = image.astype(np.uint8)
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)


def main():
    args = parse_args()
    img = cv2.imread(args.img)
    imp = to_impulse(img, args.k)

    cv2.imshow("Impulse Image", imp)
    contours = find_contours(imp)

    found = cv2.drawContours(imp.copy(), contours, 1, (255, 0, 0), 1)
    cv2.imshow(f"Detect on Impulse Image", found)

    found = cv2.drawContours(img, contours, 1, (255, 0, 0), 1)
    cv2.imshow(f"Detect on Original Image", found)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
