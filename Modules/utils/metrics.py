import numpy as np
from typing import *


def accuracy_score(Y_predict: np.ndarray, Y: np.ndarray) -> float:
    '''
    Return accuracy of given predict and ground truth
    '''
    Y = Y.reshape(Y.size, 1)
    Y_predict = Y_predict.reshape(Y_predict.size, 1)

    check_same_shape(Y_predict, Y)
    TF: np.ndarray = Y == Y_predict
    return TF[TF == True].size / Y.size

def f1_score(Y_predict: np.ndarray, Y: np.ndarray, average = 'binary') -> float:
    '''
    Calculate f1-score

    @Parameter
    -----------
    - Y_predict: predicted values
    - Y: true values
    - average: 'binary', 'micro', 'macro'. Default = 'binary'
    '''
    Pr, Rc, score = precision_recall_f1_score(Y_predict, Y, average)
    return score

def precision_score(Y_predict: np.ndarray, Y: np.ndarray, average = 'binary') -> float:
    '''
    Calculate precision score

    @Parameter
    -----------
    - Y_predict: predicted values
    - Y: true values
    - average: 'binary', 'micro', 'macro'. Default = 'binary'
    '''
    Pr, Rc, score = precision_recall_f1_score(Y_predict, Y, average)
    return Pr

def recall_score(Y_predict: np.ndarray, Y: np.ndarray, average = 'binary') -> float:
    '''
    Calculate recall score

    @Parameter
    -----------
    - Y_predict: predicted values
    - Y: true values
    - average: 'binary', 'micro', 'macro'. Default = 'binary'
    '''
    Pr, Rc, score = precision_recall_f1_score(Y_predict, Y, average)
    return Rc

def precision_recall_f1_score(Y_predict: np.ndarray, Y: np.ndarray, average = 'binary') -> Tuple[float]:
    '''
    Calculate precision, recall and f1-score

    @Parameter
    -----------
    - Y_predict: predicted values
    - Y: true values
    - average: 'binary', 'micro', 'macro'. Default = 'binary'

    @Return 
    -----------
    precision, recall, f1-score
    '''
    Y = Y.reshape(Y.size, 1)
    Y_predict = Y_predict.reshape(Y_predict.size, 1)

    check_same_shape(Y_predict, Y)

    # precision = (TP1 + TP2 + .... + TPn) / (TP1 + TP2 + ... + TPn + FP1 + FP2 + ... + FPn)
    # recall = (TP1 + TP2 + .... + TPn) / (TP1 + TP2 + ... + TPn + FN1 + FN2 + ... + FNn)
    if average == 'micro':
        TP_sum = 0
        FP_sum = 0
        FN_sum = 0
        for c in np.unique(Y):
            TP = Y_predict[Y_predict == c][Y[Y_predict == c] == c].size
            if TP == 0:
                continue
            FP = Y_predict[Y_predict == c][Y[Y_predict == c] != c].size
            FN = Y_predict[Y_predict != c][Y[Y_predict != c] == c].size

            TP_sum += TP
            FP_sum += FP
            FN_sum += FN
        
        Pr = TP_sum / (TP_sum + FP_sum)
        Rc = TP_sum / (TP_sum + FN_sum)
        
    # precision = (Pr1 + Pr2 + ... + Prn) / (number of classes)
    # recall = (Rc1 + Rc2 + ... + Rcn) / (number of classes)
    elif average == 'macro':
        PR = []
        RC = []
        for c in np.unique(Y):
            TP = Y_predict[Y_predict == c][Y[Y_predict == c] == c].size
            if TP == 0:
                PR.append(0)
                RC.append(0)
                continue
            FP = Y_predict[Y_predict == c][Y[Y_predict == c] != c].size
            FN = Y_predict[Y_predict != c][Y[Y_predict != c] == c].size

            Pr = TP / (TP + FP)
            Rc = TP / (TP + FN)
            PR.append(Pr)
            RC.append(Rc)
        
        Pr = np.sum(PR) / np.unique(Y).size
        Rc = np.sum(RC) / np.unique(Y).size

    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    else:
        TP = Y_predict[Y_predict == 1][Y[Y_predict == 1] == 1].size
        if TP == 0:
            return 0
        FP = Y_predict[Y_predict == 1][Y[Y_predict == 1] != 1].size
        FN = Y_predict[Y_predict == 0][Y[Y_predict == 0] != 0].size
        Pr = TP / (TP + FP)
        Rc = TP / (TP + FN)

    # f1-score = 2 * precision * recall / (precision + recall)
    score = 2 * Pr * Rc / (Pr + Rc)

    return Pr, Rc, score

def R_square(Y_predict: np.ndarray, Y: np.ndarray) -> float:
    '''
    Return R-square score of regression models
    '''
    Y = Y.reshape(Y.size, 1)
    mean = Y.mean()
    Y_predict = Y_predict.reshape(Y_predict.size, 1)

    check_same_shape(Y_predict, Y)

    return 1 - (np.sum((Y - Y_predict) ** 2) / np.sum((Y - mean) ** 2))


def check_same_shape(Y_predict: np.ndarray, Y: np.ndarray):
    '''
    Check if predict and true value have same shape
    '''
    if Y.shape != Y_predict.shape:
        raise Exception('Y and Y_predict must be same shape')

def entropy(X: np.ndarray) -> np.ndarray:
    '''
    Calculate entropy loss of X
    '''
    X_remove0 = X[X.nonzero()[0]]
    p = X_remove0/float(np.sum(X_remove0))
    return -np.sum(p * np.log(p))
