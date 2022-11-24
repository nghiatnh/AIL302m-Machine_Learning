from __future__ import division, print_function, unicode_literals
import numpy as np
from typing import *
from scipy import sparse
from ..utils.metrics import accuracy_score


class MultilayerPerceptron():
    '''
    Logistic Regression Model
    ----------

    Parameters
    ----------

    '''

    def __init__(self, hidden_layer_sizes: tuple = (100, ), 
                        activation: Literal['identity', 'logistic', 'tanh', 'relu'] = 'relu',
                        alpha: float = 1e-4,
                        learning_rate: float = 1e-4,
                        tol: float = 1e-5,
                        solver: Literal['gd'] = 'gd',
                        validation_fraction: float = 0.1,
                        early_stopping: bool = False,
                        max_iter: int = 200,
                        batch_size: Literal['auto'] | int = 'auto'
                ) -> None:
        self.__hidden_layer_sizes = hidden_layer_sizes
        self.__activation: Literal['identity',
                                 'logistic', 'tanh', 'relu'] = activation
        self.__alpha = alpha
        self.__lr = learning_rate
        self.__tol = tol
        self.__solver = solver
        self.__validation_fraction = validation_fraction
        self.__early_stopping = early_stopping
        self.__max_iter = max_iter
        self.__batch_size = batch_size

        self.__L = len(self.__hidden_layer_sizes) + 1
        self.__layer_sizes = self.__hidden_layer_sizes
        self.__labels: np.ndarray = None
        self.__X: np.ndarray = None
        self.__Y: np.ndarray = None
        
        self.N_train_: int = 0
        self.loss_: float = -1
        self.best_loss_: float = -1
        self.coefs_: List[np.ndarray]
        self.intercepts_: List[np.ndarray]
        self.loss_curve_: list = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        '''
        Fit and train model from given data.

        Parameter
        ----------

            - l : regularization coefficient
            - method : "GradientDescent" or "StochasticGradientDescent"
        '''
        self.__parameter(X, Y)

        self.__GD(self.__X, self.__Y)

    def predict(self, X: Type[np.ndarray]) -> Type[np.ndarray]:
        '''
        Return predict output of given data
        '''
        (_, __, Y_predict) = self.__forward(X, self.coefs_, self.intercepts_)
        return self.__labels[np.argmax(Y_predict, axis=1)]

    def score(self, X: Type[np.ndarray], Y: Type[np.ndarray]) -> float:
        '''
        Return true classification score
        '''
        pass
        return accuracy_score(self.predict(X), Y)

    def __parameter(self, X: np.ndarray, Y: np.ndarray) -> None:
        '''
        Concatenate ones column to X data
        '''
        self.__N_train = min(X.shape[0], Y.shape[0])
        self.__labels = np.unique(Y[:self.__N_train])
        self.__X = X[:self.__N_train]
        self.__Y = self.__convert_labels(Y[:self.__N_train], self.__labels.size)
        self.__layer_sizes = (self.__X.shape[1],) + self.__hidden_layer_sizes + (self.__labels.size,)

    def __convert_labels(self, y: np.ndarray, C: int) -> np.ndarray:
        '''
        Convert labels to one-hot encoding
        '''
        print(y.shape)
        Y = sparse.coo_matrix((np.ones_like(y),
            (np.arange(len(y)), y)), shape = (len(y), C)).toarray()

        return Y

    def __softmax(self, V):
        e_V = np.exp(V - np.max(V, axis=1, keepdims=True))
        Z = e_V / e_V.sum(axis=1, keepdims=True)
        return Z

    def __cost(self, Y: Type[np.ndarray], Y_predict: Type[np.ndarray]) -> np.ndarray:
        '''
        Return cost values of X, Y
        '''
        N = Y.shape[0]
        return -np.sum(Y*np.log(Y_predict)) / N

    def __forward(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        A = [None for i in range(self.__L + 1)]
        Z = [None for i in range(self.__L + 1)]
        A[0] = X
        for i in range(1, self.__L + 1):
            Z[i] = np.dot(A[i-1], W[i]) + b[i]

            # Relu function
            A[i] = np.maximum(Z[i], 0)

        Y_predict = self.__softmax(Z[self.__L])
        return (Z, A, Y_predict)

    def __backward(self, X: np.ndarray, Y: np.ndarray, Y_predict: np.ndarray, W: np.ndarray, A: np.ndarray, Z: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        N = X.shape[1]
        dW = [None for i in range(self.__L + 1)]
        db = [None for i in range(self.__L + 1)]
        Ei_1 = (Y_predict - Y) / N
        for i in range(self.__L, 0, -1):
            Ei = Ei_1
            dW[i] = np.dot(A[i-1].T, Ei)
            db[i] = np.sum(Ei, axis=0, keepdims=True)
            if i == 1:
                continue
            Ei_1 = np.dot(Ei, W[i].T)
            Ei_1[Z[i-1] <= 0] = 0  # gradient of ReLU

        return (dW, db)

    def __GD(self, X: np.ndarray, Y: np.ndarray) -> None:
        '''
        Use gradient descent for training
        '''
        W = [None] + [0.01*np.random.randn(self.__layer_sizes[i], self.__layer_sizes[i+1])
                      for i in range(self.__L)]
        b = [None] + [np.zeros((1, self.__layer_sizes[i+1])) for i in range(self.__L)]
        
        for i in range(self.__max_iter):
            (Z, A, Y_predict) = self.__forward(X, W, b)

            (dW, db) = self.__backward(X, Y, Y_predict, W, A, Z)

            # Gradient Descent update
            for j in range(1, self.__L + 1):
                W[j] += -self.__lr * dW[j]
                b[j] += -self.__lr * db[j]

            if (i + 1) % 1000 == 0:
                loss = self.__cost(Y, Y_predict)
                self.loss_curve_.append(loss)
                print("iter %d, loss: %f" % (i+1, loss))
                if len(self.loss_curve_) >= 2 and abs(self.loss_curve_[-1] - self.loss_curve_[-2]) <= self.__tol:
                    break

        self.coefs_ = W
        self.intercepts_ = b
