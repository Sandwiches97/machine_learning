# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from utils import accuracy_score

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError

    def gradient(self, y, y_pred):
        return NotImplementedError

    def acc(self, y, y_pred):
        return 0

class SquareLoss(Loss):
    def __init__(self):pass

    def loss(self, y_true, y_pred):
        return 0.5 * np.power((y_pred-y_true), 2)

    def gradient(self, y, y_pred):
        return -(y-y_pred)

class CrossEntropy(Loss):
    def __init__(self):pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1-1e-15)
        return -y*np.log(p) - (1-y)*np.log(1-p)

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1-1e-15)
        return -y/p - (1-y)/(1-p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

class SoftmaxLoss(Loss):
    """
     s_i = the i-th value of the output of softmax
     y_i 是一个 one-hot vector
     L = - (y_1*log(s_1) + y_2*log(s_2) + ... + y_T*log(s_T)
    """
    def gradient(self, y, p):
        return y-p

if __name__=="__main__":
    pass