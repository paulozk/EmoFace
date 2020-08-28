from src.processing import *
from src.model import *
import cv2

def train_model(path):
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess(path)

    # create model
    model = CNN(height=48, width=48, n_classes=7, learning_rate=0.02)

    # train model
    model.fit(X_train,
              y_train,
              X_val,
              y_val,
              batch_size=512,
              epochs=30)

    model.store_weights("weights/weights.h5")


if __name__ == "__main__":
    path = 'data/fer2013.csv'
    train_model(path)


