import numpy as np
from typing import *


class LinearRegression():
    '''
    Linear Regression Model
    ----------

    Ordinary least squares (OLS) Linear Regression.
    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------

    '''

    def __init__(self) -> None:
        self.intercept_ : np.ndarray = None
        self.coefficient_ : np.ndarray = None

    def fit(self, X: Type[np.ndarray], Y: Type[np.ndarray]) -> None:
        intercept_ = np.zeros(1)
        coefficient_ = np.zeros(X.shape[1] - 1)
        (gamma, r, num_itors, C) = (0.01, 0, 1e5, 1e-5)
        for i in range(num_itors):
            grad = self.cost_gradient(X, Y, intercept_, coefficient_, r)
            intercept_ -= gamma * grad[0]
            coefficient_ -= gamma * grad[1:]
            cost = self.cost(X, Y, intercept_, coefficient_, r)
            if cost < C:
                break
        self.intercept_ = intercept_
        self.coefficient_ = coefficient_

    def predict(self, X: Type[np.ndarray]) -> Type[np.ndarray]:
        return self.intercept_ + X @ self.coefficient_

    def score(self, X: Type[np.ndarray], Y: Type[np.ndarray]) -> float:
        return self.cost(X, Y, self.intercept_, self.coefficient_, 0)

    def cost(self, X: Type[np.ndarray], Y: Type[np.ndarray], intercept: Type[np.ndarray], coefficient: Type[np.ndarray], r: float) -> float:
        N : int = Y.size
        h: np.ndarray = self.predict(X)
        error = h - Y
        return (error.T @ error +
                       r * (coefficient.T @ coefficient)) / (2 * N)

    def cost_gradient(self, X: Type[np.ndarray], Y: Type[np.ndarray], intercept: Type[np.ndarray], coefficient: Type[np.ndarray], r: float) -> Type[np.ndarray]:
        N = Y.size
        h: np.ndarray = self.predict(X)
        error = h - Y
        grad = np.zeros(X.size[1])
        grad[0] = (X[:, 0].T @ error) / N
        grad[1:] = ((X[:, 1:].T @ error) + r * coefficient) / N
        return grad
