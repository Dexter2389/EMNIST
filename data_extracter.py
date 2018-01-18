import struct
from array import array
from scipy.io import loadmat
import numpy as np


def load_binaries(labels_path, data_path):
    """
    load EMNIST binary data, which follows the MNIST file format.

    :param data_path: the path to the binary file containing the images
    :param labels_path: the path to the labels for the images

    :return: the images and labels as list

    """
    with open(labels_path, 'rb') as file:
        magic_number, size = struct.unpack(">II", file.read(8))
        labels = array("B", file.read())

    with open(data_path, "rb") as file:
        magic_number, size, rows, cols = struct.unpack(">IIII", file.read(16))
        image_data = array("B", file.read())

    images = []
    for i in range(size):
        images.append([0] * rows * cols)

    for i in range(size):
        images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

    return images, labels


def load_char_mapping(mapping_path):
    """
    load EMNIST character mappings. This maps a label to the correspondent 
    byte value of the given character

    :param mapping_path: the path to the text file containing the mappings    

    :return: the dictionary of label mappings

    """
    mappings = {}
    with open(mapping_path) as pair:
        for line in pair:
            (key, val) = line.split()
            mappings[int(key)] = int(val)

    return mappings
