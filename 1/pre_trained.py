import argparse
import glob
import os
from time import sleep

import numpy as np

import cv2
import imutils
from imutils.object_detection import non_max_suppression


def read_images(image_path):
    for filename in glob.glob(os.path.join(image_path, "*.png")):
        image = cv2.imread(filename)
        yield image


def suppression_image(image, regions):
    image = image.copy()

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in regions])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    return image


def detect_pedestrian(image, hog):
    # Resizing the Image
    image = imutils.resize(image, width=min(400, image.shape[1]))

    # Detecting all the regions in the
    # Image that has a pedestrians inside it
    (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)

    # Drawing the regions in the Image
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)


def parse_args():
    args = argparse.ArgumentParser(description="Train human detection model")
    args.add_argument("-i", "--input", type=str, help="Path to images", default="./data/1", required=False)
    args = args.parse_args()
    return args


def main():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    for image in read_images(parse_args().input):
        detect_pedestrian(image, hog)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
