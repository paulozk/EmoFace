import pandas as pd
import numpy as np
import cv2

def read_images(path):
    data = pd.read_csv(path)
    pixels = [image.split(' ') for image in data['pixels']]
    images = np.reshape(pixels, (len(pixels), 48, 48, 1)).astype(float)
    return images, data['emotion'], data['Usage']

def get_splits(images, labels, splits):
    idx_train = splits == 'Training'
    idx_val = splits == 'PublicTest'
    idx_test = splits == 'PrivateTest'

    return images[idx_train], labels[idx_train], images[idx_val], labels[idx_val],  images[idx_test], labels[idx_test]


def preprocess(filepath):
    images, labels, splits = read_images(filepath)
    labels = pd.get_dummies(labels)
    X_train, y_train, X_val, y_val, X_test, y_test = get_splits(images, labels, splits)

    X_train /= 255.0
    X_val /= 255.0
    X_test /= 255.0

    return X_train, y_train, X_val, y_val, X_test, y_test


class FaceExtractor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, img):
        return self.face_cascade.detectMultiScale(img, 1.3, 5)

    def crop_face(self, img):
        faces = self.detect(img)
        x,y,width,height = faces[0]
        cropped_image = img[y:y+height, x:x+width]
        return cropped_image







