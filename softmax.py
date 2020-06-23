import numpy as np

def softmax(x, axis = -1):
    e_x = np.exp(x-np.max(x))
    return e_x/e_x.sum(axis=axis, keepdims=True)
