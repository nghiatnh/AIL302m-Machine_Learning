import numpy as np
from typing import *
from scipy.spatial.distance import cdist
from ..utils.metrics import accuracy_score


class KNearestNeighbors():
    '''
    K-Nearest Neighbors Model
    ----------

    Parameters
    ----------

    '''

    def __init__(self, K: int = 3) -> None:
        self._X: np.ndarray = None
        self._Y: np.ndarray = None
        self.weights = self.__uniform
        self.K = K

    def fit(self, X: Type[np.ndarray], Y: Type[np.ndarray], weights: str = 'uniform') -> None:
        '''
        Fit and train model from given data.
        '''
        self.__parameter(X, Y, weights)

    def score(self, X: Type[np.ndarray], Y: Type[np.ndarray]) -> float:
        '''
        Return true classification score
        '''
        return accuracy_score(self.predict(X), Y)

    def predict(self, X: np.ndarray) -> Type[np.ndarray]:
        '''
        Return predict output of given data 
        '''
        return self.__voting(X)

    def __parameter(self, X: np.ndarray, Y: np.ndarray, weights: str) -> None:
        '''
        Change value of properties
        '''
        self._X = X
        self._Y = Y.reshape(Y.shape[0])

        if type(weights) == str:
            if weights == 'distance':
                self.weights = self.__distance
            else:
                self.weights = self.__uniform
        else:
            self.weights = weights

    def __uniform(self, distance: np.ndarray) -> None:
        '''
        Use uniform voting (return 1 for all)
        '''

        return np.ones(distance.shape)

    def __distance(self, distance: np.ndarray) -> np.ndarray:
        '''
        Use distance voting (return inverse of distance)
        '''

        return 1 / (distance + 1e-4)

    def __voting(self, X: np.ndarray) -> None:
        '''
        Return voting base on weights function
        '''

        distance = (cdist(self._X, X))
        K_nearest = distance.argpartition(kth=self.K, axis=0)[:self.K]
        classes = np.unique(self._Y[K_nearest])
        result = [[] for x in range(len(classes))]
        for i in range(len(classes)):
            for j in range(K_nearest.shape[1]):
                nearest = distance[K_nearest[:, j],
                                   j][self._Y[K_nearest[:, j]] == classes[i]]
                if nearest.size == 0:
                    result[i].append(-np.inf)
                else:
                    result[i].append(np.sum(self.weights(nearest)))

        arr = np.array(result)
        ind = np.argmax(arr, axis=0)
        return classes[ind]
