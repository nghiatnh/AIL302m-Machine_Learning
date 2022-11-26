from __future__ import division, print_function, unicode_literals
import numpy as np
from typing import *
from scipy import sparse
from ..utils.metrics import accuracy_score
from ..utils.model_selection import train_test_split
from ..utils.matrix_functions import *
from ..utils.common import print_progress_bar
import math


class MultilayerPerceptron():
    '''
    Logistic Regression Model
    ----------

    Parameters
    ----------

    '''

    def __init__(self, hidden_layer_sizes: tuple = (100, ),
                 activation: Literal['identity',
                                     'logistic', 'tanh', 'relu'] = 'relu',
                 alpha: float = 1e-4,
                 learning_rate: float = 0.001,
                 shuffle: bool = True,
                 tol: float = 1e-4,
                 solver: Literal['lbfgs', 'sgd', 'adam'] = 'adam',
                 validation_fraction: float = 0.1,
                 early_stopping: bool = False,
                 verbose: bool = True,
                 max_iter: int = 200,
                 batch_size: Literal['auto'] | int = 'auto',
                 n_iter_no_change: int = 10,
                 n_iter_check_loss: Literal['auto'] | int = 'auto',
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-8,
                 ) -> None:
        self.__hidden_layer_sizes = hidden_layer_sizes
        self.__activation: Literal['identity',
                                   'logistic', 'tanh', 'relu'] = activation
        self.__alpha = alpha
        self.__lr = learning_rate
        self.__shuffle = shuffle
        self.__tol = tol
        self.__solver = solver
        self.__validation_fraction = validation_fraction if early_stopping else 0
        self.__early_stopping = early_stopping
        self.__verbose = verbose
        self.__max_iter = max_iter
        self.__batch_size = batch_size
        self.__n_iter_no_change = n_iter_no_change
        self.__n_iter_check_loss = max(10**(len(str(self.__max_iter)) - 3), 1) if n_iter_check_loss == 'auto' else n_iter_check_loss
        self.__beta_1 = beta_1
        self.__beta_2 = beta_2
        self.__epsilon = epsilon

        self.__L = len(self.__hidden_layer_sizes) + 1
        self.__layer_sizes = self.__hidden_layer_sizes
        self.__labels: np.ndarray = None
        self.__X: np.ndarray = None
        self.__Y: np.ndarray = None
        self.__activations = {
            "relu": {
                "function": relu,
                "gradient": relu_grad
            },
            "logistic": {
                "function": sigmoid,
                "gradient": sigmoid_grad
            },
            "identity": {
                "function": identity,
                "gradient": identity_grad
            },
            "tanh": {
                "function": tanh,
                "gradient": tanh_grad
            }
        }
        self.__solvers = {
            "lbfgs": self.__lbfgs,
            "sgd": self.__sgd,
            "adam": self.__adam,
        }
        self.__activate_function = self.__activations[self.__activation]
        self.__solver_function = self.__solvers[self.__solver]

        self.N_train_: int = 0
        self.best_loss_: float = np.inf
        self.coefs_: List[np.ndarray]
        self.intercepts_: List[np.ndarray]
        self.loss_curve_: list = []
        self.val_loss_curve_: list = []
        self.val_score_: list = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        '''
        Fit and train model from given data.

        Parameter
        ----------

            - l : regularization coefficient
            - method : "GradientDescent" or "StochasticGradientDescent"
        '''
        self.__parameter(X, Y)

        self.__solver_function(self.__X, self.__Y)

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

        self.__layer_sizes = (
            self.__X.shape[1],) + self.__hidden_layer_sizes + (self.__labels.size,)
        if self.__batch_size == 'auto':
            self.__batch_size = min(200, int(self.N_train_ * self.__validation_fraction))

    def __convert_labels(self, y: np.ndarray, C: int) -> np.ndarray:
        '''
        Convert labels to one-hot encoding
        '''
        Y = sparse.coo_matrix((np.ones_like(y),
                               (np.arange(len(y)), y)), shape=(len(y), C)).toarray()

        return Y

    def __cost(self, Y: Type[np.ndarray], Y_predict: Type[np.ndarray], W: List[np.ndarray]) -> np.ndarray:
        '''
        Return cost values of X, Y
        '''
        N = Y.shape[0]
        indexes = Y_predict != 0
        return -np.sum(Y[indexes]*np.log(Y_predict[indexes])) / N + self.__alpha * np.sum(np.linalg.norm(w) for w in W)

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
            Ei_1 = np.dot(Ei, W[i].T) * \
                self.__activate_function["gradient"](Z[i-1])

        return (dW, db)

    def __sgd(self, X: np.ndarray, Y: np.ndarray) -> None:
        '''
        Use gradient descent for training
        '''
        W = [None] + [np.random.randn(self.__layer_sizes[i], self.__layer_sizes[i+1])
                      for i in range(self.__L)]
        b = [None] + [np.zeros((1, self.__layer_sizes[i+1]))
                      for i in range(self.__L)]

        num_iter = math.ceil(int(self.N_train_ * (1 - self.__validation_fraction)) / self.__batch_size)

        same_count = 1
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=self.__validation_fraction, shuffle=False)

        for i in range(self.__max_iter):
            if self.__shuffle:
                indexes = np.arange(X_train.shape[0])
                np.random.shuffle(indexes)
                X_train = X_train[indexes,:]
                Y_train = Y_train[indexes,:]

            alpha_t = self.__lr

            for k in range(num_iter):
                X_ = X_train[k * self.__batch_size: (k + 1) * self.__batch_size, :]
                Y_ = Y_train[k * self.__batch_size: (k + 1) * self.__batch_size, :]
                (Z, A, Y_predict) = self.__forward(X_, W, b)

                (dW, db) = self.__backward(X_, Y_, Y_predict, W, A, Z)

                # Gradient Descent update
                for j in range(1, self.__L + 1):
                    W[j] += -alpha_t * dW[j]
                    b[j] += -alpha_t * db[j]

            (Z, A, Y_predict) = self.__forward(X, W, b)

            same_count = self.__print_train_progress(
                i, X_train, Y_train, X_val, Y_val, W, b, same_count)
            
            if same_count >= self.__n_iter_no_change:
                break

        self.coefs_ = W
        self.intercepts_ = b

    def __lbfgs(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass

    def __adam(self, X: np.ndarray, Y: np.ndarray) -> None:
        '''
        Use gradient descent for training
        '''
        W = [None] + [np.random.randn(self.__layer_sizes[i], self.__layer_sizes[i+1])
                      for i in range(self.__L)]
        b = [None] + [np.zeros((1, self.__layer_sizes[i+1]))
                      for i in range(self.__L)]

        num_iter = math.ceil(int(self.N_train_ * (1 - self.__validation_fraction)) / self.__batch_size)

        same_count = 1
        beta1 = self.__beta_1
        beta2 = self.__beta_2
        epsilon = self.__epsilon
        mt = [[0] * len(b), [0] * len(W)]
        vt = [[0] * len(b), [0] * len(W)]
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=self.__validation_fraction, shuffle=False)

        for i in range(self.__max_iter):
            if self.__shuffle:
                indexes = np.arange(X_train.shape[0])
                np.random.shuffle(indexes)
                X_train = X_train[indexes,:]
                Y_train = Y_train[indexes,:]

            beta1_t = beta1**(i+1)
            beta2_t = beta2**(i+1)
            alpha_t = self.__lr

            for k in range(num_iter):
                X_ = X_train[k * self.__batch_size: (k + 1) * self.__batch_size, :]
                Y_ = Y_train[k * self.__batch_size: (k + 1) * self.__batch_size, :]
                (Z, A, Y_predict) = self.__forward(X_, W, b)

                (dW, db) = self.__backward(X_, Y_, Y_predict, W, A, Z)

                # Gradient Descent update
                for j in range(1, self.__L + 1):
                    mt[1][j] = beta1 * mt[1][j] + (1 - beta1) * dW[j]
                    vt[1][j] = beta2 * vt[1][j] + (1 - beta2) * (dW[j] ** 2)
                    mt_hat = mt[1][j] / (1 - beta1_t)
                    vt_hat = vt[1][j] / (1 - beta2_t)
                    W[j] += -alpha_t * mt_hat / (np.sqrt(vt_hat) + epsilon)

                    mt[0][j] = beta1 * mt[0][j] + (1 - beta1) * db[j]
                    vt[0][j] = beta2 * vt[0][j] + (1 - beta2) * (db[j] ** 2)
                    mt_hat = mt[0][j] / (1 - beta1_t)
                    vt_hat = vt[0][j] / (1 - beta2_t)
                    b[j] += -alpha_t * mt_hat / (np.sqrt(vt_hat) + epsilon)

            (Z, A, Y_predict) = self.__forward(X, W, b)

            same_count = self.__print_train_progress(
                i, X_train, Y_train, X_val, Y_val, W, b, same_count)
            
            if same_count >= self.__n_iter_no_change:
                break

        self.coefs_ = W
        self.intercepts_ = b

    def __print_train_progress(self, epoch: int, 
                                    X: np.ndarray, 
                                    Y: np.ndarray, 
                                    X_val: np.ndarray, 
                                    Y_val: np.ndarray, 
                                    W: np.ndarray, 
                                    b: np.ndarray, 
                                    same_count: int
                                ) -> int:
        '''
        Print progress of training steps.

        Parameter
        -----------
        - X, Y, X_val, Y_val: Array, data for calculate train loss, validation loss, and validation score
        - W, b: List, Weights and biases
        - same_count: int, current count of consecutive same validation scores

        Return
        ----------
        The integer the `same_count` for early stopping

        Example
        ----------
        >>> self.__print_train_progress(i, X, Y, X_val, Y_val, W, b, same_count, min(1000, self.__max_iter))
        # epoch 9000:	|██████████████████████████████████████████████████|	100%
        # epoch 9000, train loss: 0.0490555408805367
        '''
        epoch_rate = self.__n_iter_check_loss
        if (epoch + 1) % epoch_rate == 0:
            # Loss for training
            (_, _, Y_predict) = self.__forward(X, W, b)
            loss = self.__cost(Y, Y_predict, W[1:])
            self.loss_curve_.append(loss)

            if self.__early_stopping:
                # Loss, score for validation
                (_, _, Y_predict) = self.__forward(X_val, W, b)
                val_loss = self.__cost(Y_val, Y_predict, W[1:])
                self.best_loss_ = min(val_loss, self.best_loss_)
                val_score = accuracy_score(
                    np.argmax(Y_predict, axis=1), np.argmax(Y_val, axis=1))

                self.val_loss_curve_.append(val_loss)
                self.val_score_.append(val_score)

                # Calculate not change validation score or loss for early stopping
                if len(self.val_loss_curve_) >= 2:
                    if self.val_loss_curve_[-2] - self.val_loss_curve_[-1] <= self.__tol or self.val_score_[-1] - self.val_score_[-2] <= self.__tol:
                        same_count += 1
                    else:
                        same_count = 1
            else:
                if len(self.loss_curve_) >= 2:
                    if self.loss_curve_[-2] - self.loss_curve_[-1] <= self.__tol:
                        same_count += 1
                    else:
                        same_count = 1

            # Print loss after epoch_rate
            if self.__verbose:
                if self.__n_iter_check_loss != 1:
                    print('')

                if self.__early_stopping:
                    print("Epoch {},\ttrain loss: {},\nvValidation loss: {},\tvalidation accuracy: {}".format(
                        epoch+1, loss, val_loss, val_score))

                    if same_count >= self.__n_iter_no_change:
                        print(f'Validation score did not improve more than tol={self.__tol} for {self.__n_iter_no_change} consecutive epochs. Stopping.')
                else:
                    print("Epoch {},\ttrain loss: {}".format(epoch+1, loss))

                    if same_count >= self.__n_iter_no_change:
                        print(f'Training loss did not improve more than tol={self.__tol} for {self.__n_iter_no_change} consecutive epochs. Stopping.')
            return same_count

        # Print progress bar
        rate = ((((epoch + 1) % epoch_rate) + 1) / epoch_rate)
        if self.__verbose:
            print_progress_bar(f'Epoch {math.ceil(epoch / epoch_rate) * epoch_rate}',rate, f'{int(rate * 100)}%')

        return same_count