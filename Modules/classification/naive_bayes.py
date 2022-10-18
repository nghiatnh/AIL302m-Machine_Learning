import numpy as np
from typing import *


class MultinomialNB():
    '''
    Linear Regression Model
    ----------

    Ordinary least squares (OLS) Linear Regression.
    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets __predicted by the linear approximation.

    Parameters
    ----------

    '''

    def __init__(self) -> None:
        self.X : np.ndarray = None
        self.Y : np.ndarray = None
        self.classes : np.ndarray = None
        self.p_coef : np.ndarray = None

    def fit(self, X: Type[np.ndarray], Y: Type[np.ndarray]) -> None:
        '''
        Fit and train model from given data.

        Parameter
        ----------

            - l : regularization coefficient
            - method : "GradientDescent" or "Equation"
        '''
        self.__parameter(X, Y)

        self.__train()

    def score(self, X: Type[np.ndarray], Y: Type[np.ndarray]) -> float:
        '''
        Return Mean Square Error of given data
        '''
        return self.__cost(np.concatenate((np.ones((X.shape[0],1)), X), axis = 1), Y, 0)

    def predict_proba(self, X : np.ndarray):
        
        p = np.zeros((X.shape[0], self.classes.size))

        for i in range(self.classes.size):
            pi = self.Y[self.Y == self.classes[i]].size / self.Y.size
            p[:,i] = (np.log(pi) + np.sum(np.log(self.p_coef[i]) * X, axis=1))
        
        D = 10 ** int(np.log10(np.abs(p[0][0])) - 2)
        return (np.exp(p / D).T / np.sum(np.exp(p / D), axis=1)).T

    def predict(self, X : np.ndarray) -> Type[np.ndarray]:
        '''
        Return predict output of given data (private function)
        '''
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]

    def __parameter(self, X : np.ndarray, Y : np.ndarray) -> None:
        '''
        Concatenate ones column to X data
        '''
        self.X = X
        self.Y = Y.reshape(Y.shape[0])
        self.classes = np.unique(self.Y)

    def __train(self) -> None:
        '''
        Use gradient descent for training
        '''

        sum_train = np.zeros((self.classes.size, self.X.shape[1]))
        alpha = 1
        d = self.X.shape[1]
        for i in range(self.classes.size):
            Xi = self.X[self.Y == self.classes[i]]
            sum_train[i] = np.sum(Xi, axis=0)
            sum_train[i] = (alpha + sum_train[i])  / (np.sum(sum_train[i]) + d * alpha)

        self.p_coef = sum_train

