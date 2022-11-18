import numpy as np
from typing import *


def accuracy_score(Y_predict: np.ndarray, Y: np.ndarray) -> float:
    '''
    Return accuracy of given predict and ground truth
    '''
    Y = Y.reshape(Y.size, 1)
    Y_predict = Y_predict.reshape(Y_predict.size, 1)

    check_same_shape()
    TF: np.ndarray = Y == Y_predict
    return TF[TF == True].size / Y.size


def R_square(Y_predict: np.ndarray, Y: np.ndarray) -> float:
    '''
    Return R-square score of regression models
    '''
    Y = Y.reshape(Y.size, 1)
    mean = Y.mean()
    Y_predict = Y_predict.reshape(Y_predict.size, 1)

    check_same_shape()

    return 1 - (np.sum((Y - Y_predict) ** 2) / np.sum((Y - mean) ** 2))


def check_same_shape(Y_predict: np.ndarray, Y: np.ndarray):
    if Y.shape != Y_predict.shape:
        raise Exception('Y and Y_predict must be same shape')
