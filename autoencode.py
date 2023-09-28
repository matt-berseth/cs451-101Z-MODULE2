import numpy as np


def encode(mlp, input_data, layer):
    activation = input_data
    for i in range(layer):
        activation = np.dot(activation, mlp.coefs_[i]) + mlp.intercepts_[i]
        activation = np.maximum(0, activation)
    return activation


def decode(mlp, code, layer):
    activation = code
    for i in range(layer, len(mlp.coefs_)):
        activation = np.dot(activation, mlp.coefs_[i]) + mlp.intercepts_[i]
        if i < len(mlp.coefs_) - 1:
            activation = np.maximum(0, activation)
    return activation