import numpy as np
from typing import *

def train_test_split(X: np.ndarray, Y: np.ndarray, test_size: float = 0.2, train_size: float = None, shuffle: bool = False) -> Tuple[np.ndarray]:
    N = X.shape[0]
    indexes = np.arange(N)
    if shuffle:
        indexes = np.random.shuffle(indexes)

    if train_size == None and test_size == None:
        raise Exception('train_size or test_size must be not None')
    if train_size != None and test_size == None:
        N_train = int(N * train_size)
        N_test = N - N_train
    elif train_size == None and test_size != None:
        N_test = int(N * test_size)
        N_train = N - N_test
    elif train_size != None and test_size != None:
        if train_size + test_size > 1:
            raise Exception('train_size + test_size must be smaller than 1')

        N_train = int(N * train_size)
        N_test = min(int(N * test_size), N - N_train)

    return X[indexes[:N_train]], X[indexes[N_train:N_train+N_test]], Y[indexes[:N_train]], Y[indexes[N_train:N_train+N_test]]
    