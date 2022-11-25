import numpy as np
from typing import *

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)

def relu_grad(x: np.ndarray) -> np.ndarray:
    grad = x.copy()
    grad[x > 0] = 1
    grad[x <= 0] = 0
    return grad

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1 - tanh(x) ** 2

def identity(x: np.ndarray) -> np.ndarray:
    return x

def identity_grad(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)

def softmax(V):
    e_V = np.exp(V - np.max(V, axis=1, keepdims=True))
    Z = e_V / e_V.sum(axis=1, keepdims=True)
    return Z