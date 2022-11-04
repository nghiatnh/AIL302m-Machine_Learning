import numpy as np
from typing import *
from scipy.spatial.distance import cdist


class KMeanClustering():
    '''
    K-Means Clustering  Model
    ----------
    

    Parameters
    ----------

    '''

    def __init__(self) -> None:
        self.X : np.ndarray = None
        self.K : int = 0

    def fit(self, X: Type[np.ndarray], K : int, max_iter = 1000) -> None:
        '''
        Fit and train model from given data.

        Parameter
        ----------
        '''
        self.__parameter(X, K)
        self.__k_means(max_iter)


    def predict(self, X: Type[np.ndarray]) -> Type[np.ndarray]:
        '''
        Return predict output of given data
        '''
        return self.__labels[self.__assign_labels(X, self.__centers)]

    def score(self, X: Type[np.ndarray]) -> float:
        '''
        Return Mean Square Error of given data
        '''
        return -self.__cost(X, self.centers)

    def __cost(self, X : np.ndarray, center : np.ndarray) -> float:
        labels = self.__assign_labels(X, center)
        return np.linalg.norm(X - center[labels]) ** 2

    def __parameter(self, X : np.ndarray, K : int) -> None:
        '''
        '''
        self.X = X
        self.K = K

    def __init_centers(self, X : np.ndarray, K : int, init_number : int = 50) -> np.ndarray:
        # randomly pick K rows of X as initial centers
        centers = [X[np.random.choice(X.shape[0], K, replace=False)] for i in range(init_number)]
        # Choose center with lowest cost
        return min(centers, key=lambda center: self.__cost(X, center))

    def __assign_labels(self, X : np.ndarray, centers : np.ndarray) -> np.ndarray:
        # calculate pairwise distances btw data and centers
        D = cdist(X, centers)
        # return index of the closest center
        return np.argmin(D, axis = 1)

    def __update_centers(self, X : np.ndarray, labels : np.ndarray, K : int) -> np.ndarray:
        centers = np.zeros((K, X.shape[1]))
        for k in range(K):
            # collect all points assigned to the k-th cluster 
            Xk = X[labels == k, :]
            # take average
            centers[k,:] = np.mean(Xk, axis = 0)
        return centers

    def __is_converged(self, centers : np.ndarray, new_centers : np.ndarray) -> bool:
        # return True if two sets of centers are the same
        return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))


    def __k_means(self, max_iter : int) -> None:
        centers = self.__init_centers(self.X, self.K)
        for i in range(max_iter):
            labels = self.__assign_labels(self.X, centers)
            new_centers = self.__update_centers(self.X, labels, self.K)
            if self.__is_converged(centers, new_centers):
                break
            centers = new_centers

        self.centers = centers