from typing import List, Tuple, Any

import numpy as np
import cv2, joblib
from imutils.object_detection import non_max_suppression
from skimage.feature import hog
from skimage import color
from skimage.transform import pyramid_gaussian

import argparse


def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y : y + window_size[1], x : x + window_size[0]])


def predict(model, image, size=(64, 128), step_size=(9, 9), downscale=1.25) -> List[Tuple[int, int, Any, int, int]]:
    detections = []
    scale = 0
    for im_scaled in pyramid_gaussian(image, downscale=downscale):
        # The list contains detections at the current scale
        if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
            break
        for (x, y, window) in sliding_window(im_scaled, size, step_size):
            if window.shape[0] != size[1] or window.shape[1] != size[0]:
                continue
            window = color.rgb2gray(window)

            fd = hog(window, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
            fd = fd.reshape(1, -1)
            pred = model.predict(fd)
            if pred == 1 and model.decision_function(fd) > 0.75:
                detections.append(
                    (
                        int(x * (downscale ** scale)),
                        int(y * (downscale ** scale)),
                        model.decision_function(fd),
                        int(size[0] * (downscale ** scale)),
                        int(size[1] * (downscale ** scale)),
                    )
                )

        scale += 1

    return detections


def parse_args():
    parser = argparse.ArgumentParser(description="Demo for human detection from image")
    parser.add_argument("-i", "--input", type=str, help="input file", required=True)
    parser.add_argument("-o", "--output", type=str, default="./output/output.png", help="output file")
    parser.add_argument("-m", "--model", type=str, default="./models/model.dat", help="the trained model file")
    args = parser.parse_args()
    return args


def draw_detections(image, detections):
    clone = image.copy()
    clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print("sc: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.5)
    for (x1, y1, x2, y2) in pick:
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.putText(clone, "Human", (x1 - 2, y1 - 2), 1, 0.75, (255, 255, 0), 1)

    return clone


def main():
    args = parse_args()

    image = cv2.imread(args.input)
    image = cv2.resize(image, (400, 256))
    model = joblib.load(args.model)

    detections = predict(model, image)
    clone = draw_detections(image, detections)

    cv2.imwrite(args.output, clone)
    cv2.imshow("Detections", clone)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

