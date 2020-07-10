import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import sklearn.preprocessing import StandardScaler

def read_images(path):
    data = pd.read_csv(path)
    data = data.sample(n=1000, random_state=42)
    pixels = [image.split(' ') for image in data['pixels']]
    #print(pixels)
    images = np.reshape(pixels, (len(pixels), 48, 48)).astype(float)
    #for i in range(5):
    #    plt.imshow(images[i], cmap='gray')
    #    plt.show()

    return images, data['emotion'], data['Usage']

def get_splits(images, labels, splits):
    idx_train = splits == 'Training'
    idx_val = splits == 'PublicTest'
    idx_test = splits == 'PrivateTest'

    return images[idx_train], labels[idx_train], images[idx_val], labels[idx_val],  images[idx_test], labels[idx_test]


def preprocess(filepath):
    images, labels, splits = read_images(filepath)
    X_train, y_train, X_val, y_val, X_test, y_test = get_splits(images, labels, splits)

    #normalize
    X_train /= 255.0
    X_val /= 255.0
    X_test /= 255.0

    return X_train, y_train, X_val, y_val, X_test, y_test












