import numpy as np
from typing import *


class LogisticRegression():
    '''
    Logistic Regression Model
    ----------

    Parameters
    ----------

    '''

    def __init__(self) -> None:
        self.__theta : np.ndarray = None
        self.X : np.ndarray = None
        self.Y : np.ndarray = None
        self.l : float = 1000
        self.threshold : float = 0.5
        self.costs : list = []

    def fit(self, X: Type[np.ndarray], Y: Type[np.ndarray], l : float = 0.01, method : str = "GradientDescent") -> None:
        '''
        Fit and train model from given data.

        Parameter
        ----------

            - l : regularization coefficient
            - method : "GradientDescent" or "StochasticGradientDescent"
        '''
        self.__parameter(X, Y, l)

        if method == "Equation":
            self.__equation()
        else:
            self.__GD()


    def predict(self, X: Type[np.ndarray]) -> Type[np.ndarray]:
        '''
        Return predict output of given data
        '''
        result = self.__sigmoid(self.intercept_() + X @ self.coefficient_())
        result[result <= self.threshold] = 0
        result[result > self.threshold] = 1
        return result

    def score(self, X: Type[np.ndarray], Y: Type[np.ndarray]) -> float:
        '''
        Return true classification score
        '''
        predict = self.predict(X)
        error = predict.reshape((predict.size, 1)) == Y.reshape((Y.size, 1))
        return error[error == True].size / Y.size

    def coefficient_(self) -> np.ndarray:
        '''
        Return coefficient of model
        '''
        return self.__theta[1:]

    def intercept_(self) -> np.ndarray:
        '''
        Return intercept (bias) of model
        '''
        return self.__theta[0]

    def __sigmoid(self, X : np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def __predict(self, X : np.ndarray) -> Type[np.ndarray]:
        '''
        Return predict output of given data (private function)
        '''
        return self.__sigmoid(X @ self.__theta)

    def __parameter(self, X : np.ndarray, Y : np.ndarray, l : float) -> None:
        '''
        Concatenate ones column to X data
        '''
        self.X = np.concatenate((np.ones((Y.shape[0],1)), X), axis = 1)
        self.Y = Y.reshape((Y.shape[0], 1))
        self.l = l

    def __cost(self, X: Type[np.ndarray], Y: Type[np.ndarray], l: float) -> np.ndarray:
        '''
        Return cost values of X, Y
        '''
        N = Y.size
        h = self.__predict(X).reshape((Y.size, 1))
        return -np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h)) + self.l * np.sum(self.__theta.T @ self.__theta)

    def __cost_gradient(self, X: Type[np.ndarray], Y: Type[np.ndarray], l: float) -> np.ndarray:
        '''
        Return gradient of cost values
        '''
        N = Y.size
        h = self.__predict(X).reshape((Y.size, 1))
        error = h - Y
        # print(X.T @ error)
        return X.T @ error + self.l * self.__theta


    def __equation(self) -> None:
        '''
        Use equation for training
        '''
        temp = self.X.T @ self.X + self.l * np.identity(self.X.shape[1])
        self.__theta = np.linalg.pinv(temp) @ self.X.T @ self.Y

    def __GD(self) -> None:
        '''
        Use gradient descent for training
        '''
        itor = 1000000
        init_lr = 1e-5
        lr = init_lr
        C = 1e-5
        self.costs.clear()
        self.__theta = np.zeros((self.X.shape[1], 1))
        for i in range(itor):
            grad = self.__cost_gradient(self.X, self.Y, self.l)
            self.__theta = self.__theta - lr * grad
            self.costs.append(self.__cost(self.X, self.Y, self.l))
            if len(self.costs) > 1 and abs(self.costs[-1] - self.costs[-2]) < C:
                break
            elif len(self.costs) > 1 and 100 * C < abs(self.costs[-1] - self.costs[-2]) < 1000 * C:
                lr += init_lr

