import numpy as np
from typing import *

class LinearRegression():
    '''
    Linear Regression Model
    ----------

    Ordinary least squares Linear Regression.
    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    
    '''
    def __init__(self) -> None:
        pass

    def fit(self, X: Type[np.ndarray], Y: Type[np.ndarray]) -> None:
        pass

    def predict(self, X : Type[np.ndarray]) -> Type[np.ndarray]:
        pass

    def score(self, X : Type[np.ndarray], Y : Type[np.ndarray]) -> float:
        pass
