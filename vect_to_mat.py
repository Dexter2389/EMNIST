import numpy as np
import tensorflow as tf

def classvec_to_binary_matrix(vector):
    permut = np.zeros((vector.shape[0],vector.max()+1))
    permut[np.arange(vector.shape[0]),vector] = 1
    return permut