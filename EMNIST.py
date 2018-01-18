import data_extracter as de
import baseline as perceptron
import dense_cnn_training as adcnn
import numpy as np
import vect_to_mat as conv

x_train, y_train = de.load_binaries(r"C:\Stuff\Nextech Labs\Machine Learning\Learning part\EMNIST\emnist-mnist-train-images-idx3-ubyte",
                                    r"C:\Stuff\Nextech Labs\Machine Learning\Learning part\EMNIST\emnist-mnist-train-labels-idx1-ubyte")
x_test, y_test = de.load_binaries(r"C:\Stuff\Nextech Labs\Machine Learning\Learning part\EMNIST\emnist-mnist-test-images-idx3-ubyte",
                                  r"C:\Stuff\Nextech Labs\Machine Learning\Learning part\EMNIST\emnist-mnist-test-labels-idx1-ubyte")

mappings = de.load_char_mapping(
    r"C:\Stuff\Nextech Labs\Machine Learning\Learning part\EMNIST\emnist-mnist-mapping.txt")

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

y_train_matrix = conv.classvec_to_binary_matrix(y_train)
y_test_matrix = conv.classvec_to_binary_matrix(y_test)

dim = len(mappings)

adcnn.dense_cnn_training(x_train, y_train_matrix, x_test, y_test_matrix, dim)
