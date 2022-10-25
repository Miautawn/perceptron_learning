from typing import Callable
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from utils.perceptron_utils import step_activation, sigmoid_activation


class Perceptron:
    """
    Perceptron implementation that uses ADALINE training technique, which
    is to use pure stochastic gradient descent (i.e. batch_size = 1) and 
    do not use activation function inside loss function computation.
    """
    def __init__(self, n_features: int, learning_rate: float, activation_function: Callable):
        self.w = np.random.randn(n_features) * sqrt(2.0 / n_features)
        self.learning_rate = learning_rate
        self.activation_function = step_activation if activation_function == "Step Activatation" else sigmoid_activation

    def _loss_function(self, y_true: float, prob: float) -> float:
        return 0.5 * ((y_true - prob) ** 2)

    def _update_weights(self, x: np.ndarray, y_true: float, prob: float):
        w_gradient = (y_true - prob) * -x # Loss function derivative with respect to whole weight vector
        self.w -= self.learning_rate * w_gradient

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> float:
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # train loop
        y_train_pred = []
        for i in range(x_train.shape[0]):
            # update weights
            prob = self.predict_proba(x_train[i])
            train_loss += self._loss_function(y_train[i], prob)
            y_train_pred.append(round(prob))

            self._update_weights(x_train[i], y_train[i], prob)
        train_acc = accuracy_score(y_train, y_train_pred)

        # eval loop
        y_val_pred = []
        for i in range(x_val.shape[0]):
            prob = self.predict_proba(x_val[i])
            val_loss += self._loss_function(y_val[i], prob)
            y_val_pred.append(round(prob))

        val_acc = accuracy_score(y_val, y_val_pred)  

        return (train_loss, train_acc, val_loss, val_acc)

    def predict(self, x: np.ndarray) -> int:
        prob = self.predict_proba(x)
        return round(prob)

    def predict_proba(self, x: np.ndarray) -> float:
        a = np.dot(self.w, x)
        return self.activation_function(a)

        