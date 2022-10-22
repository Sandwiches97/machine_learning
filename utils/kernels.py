# -*- coding: utf-8 -*-
import numpy as np


def linear_kernel(**kwargs):
    def f(x1, x2):
        return np.inner(x1, x2)
    return f

def polynomial_kernel(power: int, coef, **kwargs):
    def f(x1, x2):
        return (np.inner(x1, x2) + coef)**power

def rbf_kernel(gamma, **kwargs):
    def f(x1, x2):
        # linalg = linear 线性 + algebra 代数，norm 默认求二范数
        distance = np.linalg.norm(x1-x2)**2
        return np.exp(- gamma * distance)
    return f
