import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering("th")


def dense_cnn_training(X_train, y_train, X_test, y_test, num_classes):
    """
    Trains an OCR using keras. The model is a somewhat bigger convolutional neural network.

    :param x_train: the training data in a NxP format where N is the amount of images, and P the 1D data vector
    :param y_train: the labels for the training data in a binary matrix class format
    :param x_test: the training data in a KxL format where K is the amount of images, and L the 1D data vector
    :param y_test: the labels for the test data in a binary matrix class format
    :param num_classes: the amount of classes for the dataset

    """
    seed = 7
    np.random.seed(seed)

    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype("float32")
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype("float32")

    # Normalizing our data between 0 and 1
    X_train = X_train / 255
    X_test = X_test / 255
    num_pixels = X_train.shape[1]

    model = initialize_model(num_pixels, num_classes)
    model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=20, batch_size=200, verbose=2)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Error: %.2f%%" % (100 - scores[1] * 100))


def initialize_model(num_pixels, num_classes):
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return model
