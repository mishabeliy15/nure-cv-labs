from typing import List, Tuple

import cv2
import glob
import joblib
import numpy as np
import os
import argparse
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import metrics


def load_images(image_path: str, label: int) -> Tuple[List[np.ndarray], List[int]]:
    images = []
    for filename in glob.glob(os.path.join(image_path, "*.png")):
        image = cv2.imread(filename, 0)
        image = cv2.resize(image, (64, 128))
        features = hog(image, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
        images.append(features)
    labels = [label] * len(images)
    return images, labels


def train_and_evaluate_model(X_train, y_train, X_test, y_test) -> LinearSVC:
    model = LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(
        f"Classification report for classifier {model}:", f"{metrics.classification_report(y_test, y_pred)}", sep="\n"
    )
    return model


def parse_args():
    args = argparse.ArgumentParser(description="Train human detection model")
    args.add_argument("-d", "--data", type=str, help="Path to train data", default="./data/", required=False)
    args.add_argument("-m", "--model", type=str, help="Path to save model", default="./models/model.dat", required=False)

    args = args.parse_args()
    return args

def main():
    args = parse_args()

    pos_im_path = os.path.join(args.data, "1")
    neg_im_path = os.path.join(args.data, "0")

    positive_images, positive_labels = load_images(pos_im_path, 1)
    negative_images, negative_labels = load_images(neg_im_path, 0)

    X = np.float32(positive_images + negative_images)
    Y = np.array(positive_labels + negative_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    print("Train Data:", len(X_train))

    model = train_and_evaluate_model(X_train, y_train, X_test, y_test)

    joblib.dump(model, args.model)
    print("Model saved : {}".format(args.model))


if __name__ == "__main__":
    main()
