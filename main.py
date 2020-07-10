from src.processing import *
from src.model import *

def train_model(path):
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess(path)

    # create model
    model = CNN(height=48, width=48, n_classes=7)
    print(model.model)
    # train model



if __name__ == "__main__":
    path = 'data/fer2013.csv'
    train_model(path)


