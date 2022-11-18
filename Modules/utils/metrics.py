import numpy as np
from typing import *

def accuracy_score(Y_predict : np.ndarray, Y : np.ndarray) -> float:
    '''
    Return accuracy of given predict and ground truth
    '''
    if Y.shape != Y_predict.shape:
        raise Exception('Y and Y_predict must be same shape')
    TF : np.ndarray = Y == Y_predict
    return TF[TF == True].size / Y.size