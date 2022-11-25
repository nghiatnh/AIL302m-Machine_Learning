from __future__ import division, print_function, unicode_literals
import numpy as np
from typing import *
from scipy import sparse
from ..utils.metrics import accuracy_score
from ..utils.model_selection import train_test_split
from ..utils.matrix_functions import *
import math


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
                        show_loss: bool = True,
                        max_iter: int = 200,
                        batch_size: Literal['auto'] | int = 'auto',
                        n_iter_no_change: int = 10,
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
        self.__show_loss = show_loss
        self.__max_iter = max_iter
        self.__batch_size = batch_size
        self.__n_iter_no_change = n_iter_no_change

        self.__L = len(self.__hidden_layer_sizes) + 1
        self.__layer_sizes = self.__hidden_layer_sizes
        self.__labels: np.ndarray = None
        self.__X: np.ndarray = None
        self.__Y: np.ndarray = None
        self.__X_val: np.ndarray = None
        self.__Y_val: np.ndarray = None
        self.__activations = {
            "relu" : {
                "function": relu,
                "gradient": relu_grad
            },
            "logistic" : {
                "function": sigmoid,
                "gradient": sigmoid_grad
            },
            "identity" : {
                "function": identity,
                "gradient": identity_grad
            },
            "tanh" : {
                "function": tanh,
                "gradient": tanh_grad
            }
        }
        self.__activate_function = self.__activations[self.__activation]
        self.__val_loss_curve: list = []
        self.__val_score: list = []
        
        self.N_train_: int = 0
        self.N_val_: int = 0
        self.best_loss_: float = np.inf
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
        return accuracy_score(self.predict(X), Y)

    def __parameter(self, X: np.ndarray, Y: np.ndarray) -> None:
        '''
        Concatenate ones column to X data
        '''
        self.N_train_ = min(X.shape[0], Y.shape[0])
        self.__labels = np.unique(Y[:self.N_train_])
        self.__X = X[:self.N_train_]
        self.__Y = self.__convert_labels(Y[:self.N_train_], self.__labels.size)
        if self.__early_stopping:
            self.__X, self.__X_val, self.__Y, self.__Y_val = train_test_split(self.__X, self.__Y, test_size=self.__validation_fraction)

        self.__layer_sizes = (self.__X.shape[1],) + self.__hidden_layer_sizes + (self.__labels.size,)
        if self.__batch_size == 'auto':
            self.__batch_size = min(200, self.N_train_)

    def __convert_labels(self, y: np.ndarray, C: int) -> np.ndarray:
        '''
        Convert labels to one-hot encoding
        '''
        Y = sparse.coo_matrix((np.ones_like(y),
            (np.arange(len(y)), y)), shape = (len(y), C)).toarray()

        return Y

    def __cost(self, Y: Type[np.ndarray], Y_predict: Type[np.ndarray], W: List[np.ndarray]) -> np.ndarray:
        '''
        Return cost values of X, Y
        '''
        N = Y.shape[0]
        return -np.sum(Y*np.log(Y_predict)) / N + self.__alpha * np.sum(np.linalg.norm(w) for w in W)

    def __forward(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        A = [None for i in range(self.__L + 1)]
        Z = [None for i in range(self.__L + 1)]
        A[0] = X
        for i in range(1, self.__L + 1):
            Z[i] = np.dot(A[i-1], W[i]) + b[i]

            # Relu function
            A[i] = self.__activate_function["function"](Z[i])

        Y_predict = softmax(Z[self.__L])
        return (Z, A, Y_predict)

    def __backward(self, X: np.ndarray, Y: np.ndarray, Y_predict: np.ndarray, W: np.ndarray, A: np.ndarray, Z: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        N = X.shape[1]
        dW = [None for i in range(self.__L + 1)]
        db = [None for i in range(self.__L + 1)]
        Ei_1 = (Y_predict - Y) / N
        for i in range(self.__L, 0, -1):
            Ei = Ei_1
            dW[i] = np.dot(A[i-1].T, Ei) + self.__alpha * W[i]
            db[i] = np.sum(Ei, axis=0, keepdims=True)
            if i == 1:
                continue
            Ei_1 = np.dot(Ei, W[i].T) * self.__activate_function["gradient"](Z[i-1])

        return (dW, db)

    def __GD(self, X: np.ndarray, Y: np.ndarray) -> None:
        '''
        Use gradient descent for training
        '''
        W = [None] + [0.01*np.random.randn(self.__layer_sizes[i], self.__layer_sizes[i+1])
                      for i in range(self.__L)]
        b = [None] + [np.zeros((1, self.__layer_sizes[i+1])) for i in range(self.__L)]
        
        num_iter = math.ceil(self.N_train_ / self.__batch_size)
        same_count = 1

        for i in range(self.__max_iter):

            for k in range(num_iter):
                X_ = X[k * self.__batch_size: (k + 1) * self.__batch_size,:]
                Y_ = Y[k * self.__batch_size: (k + 1) * self.__batch_size,:]
                (Z, A, Y_predict) = self.__forward(X_, W, b)

                (dW, db) = self.__backward(X, Y_, Y_predict, W, A, Z)

                # Gradient Descent update
                for j in range(1, self.__L + 1):
                    W[j] += -self.__lr * dW[j]
                    b[j] += -self.__lr * db[j]

            (Z, A, Y_predict) = self.__forward(X, W, b)
            if (i + 1) % 1000 == 0:
                loss = self.__cost(Y, Y_predict, W[1:])
                self.loss_curve_.append(loss)
                if self.__early_stopping:
                    (Z, A, Y_predict) = self.__forward(self.__X_val, W, b)
                    val_loss = self.__cost(self.__Y_val, Y_predict, W[1:])
                    self.best_loss_ = min(val_loss, self.best_loss_)
                    val_score = accuracy_score(np.argmax(Y_predict, axis=1), np.argmax(self.__Y_val, axis=1))
                    self.__val_loss_curve.append(val_loss)
                    self.__val_score.append(val_score)

                    if len(self.__val_loss_curve) >= 2:
                        if abs(self.__val_loss_curve[-1] - self.__val_loss_curve[-2]) <= self.__tol or abs(self.__val_score[-1] - self.__val_score[-2]) <= self.__tol:
                            same_count += 1
                        else:
                            same_count = 1
                
                if self.__show_loss: 
                    print('')
                    if self.__early_stopping:
                        print("epoch %d, train loss: %f, validation loss: %f, validation accuracy: %f" % (i+1, loss, val_loss, val_score))
                    else:
                        print("epoch %d, loss: %f" % (i+1, loss))
                        
                if same_count >= self.__n_iter_no_change:
                    break
            
            rate = (((i + 1) % 1000) / 1000)
            length = int(rate * 50) + 1
            bar = '█' * length + '-' * (50 - length)
            print(f'\repoch {i+1}:\t|{bar}|\t{int(rate * 100) + 1}%', end='\r')

        self.coefs_ = W
        self.intercepts_ = b
