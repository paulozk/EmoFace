import tensorflow.keras as keras

class CNN:
    def __init__(self, height, width, n_classes):
        self.model = self.build_model(height, width, n_classes)

    def build_model(self, height, width, n_classes):
        input_layer = keras.Input(shape=(height, width))

        conv1 = keras.layers.Conv2D(filters = 25, kernel_size = (3,3), activation="relu", padding='same')(input_layer)
        max1 = keras.layers.MaxPooling2D(pool_size=(3, 3))(conv1)
        bnorm = keras.layers.BatchNormalization()(max1)

        conv2 = keras.layers.Conv2D(filters=50, kernel_size=(3, 3), activation="relu", padding='same')(bnorm)
        max2 = keras.layers.MaxPooling2D(pool_size=(3, 3))(conv2)
        bnorm2 = keras.layers.BatchNormalization()(max2)

        conv3 = keras.layers.Conv2D(filters=125, kernel_size=(3, 3), activation="relu", padding='same')(bnorm2)
        max3 = keras.layers.MaxPooling2D(pool_size=(3, 3))(conv3)
        bnorm3 = keras.layers.BatchNormalization()(max3)

        drop = keras.layers.Dropout(0.2)(bnorm3)

        bnorm3_flat = keras.layers.Flatten()(drop)

        output_layer = keras.layers.Dense(n_classes, activation="softmax")(bnorm3_flat)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model


    def fit(self, X_train, y_train, X_val, y_val):
        pass


    def predict(self, X):
        pass


