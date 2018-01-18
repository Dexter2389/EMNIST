import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


def baseline_training(X_train, y_train, X_test, y_test, num_classes):
    """
    Trains an OCR using keras. The model is a simple multi-layer perceptron model
    to act as a baseline.

    :param x_train: the training data in a NxP format where N is the amount of images, and P the 1D data vector
    :param y_train: the labels for the training data in a binary matrix class format
    :param x_test: the training data in a KxL format where K is the amount of images, and L the 1D data vector
    :param y_test: the labels for the test data in a binary matrix class format
    :param num_classes: the amount of classes for the dataset

    """
    seed = 7
    np.random.seed(seed)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalizing our data to be between 0 and 1
    X_train = X_train / 255
    X_test = X_test / 255
    num_pixels = X_train.shape[1]

    model = initialize_model(num_pixels, num_classes)
    model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=10, batch_size=200, verbose=2)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))


def initialize_model(num_pixels, num_classes):
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels,
                    kernel_initializer='normal', activation="relu"))
    model.add(Dense(num_classes, kernel_initializer='normal', activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return model
