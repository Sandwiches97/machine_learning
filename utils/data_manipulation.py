# -*- coding: utf-8 -*-

from __future__ import division
from itertools import combinations_with_replacement
import numpy as np
import math
import sys
from typing import List, Optional

def shuffle_data(X, y, seed=None):
    """
     Random shuffle of the samples in X and y
    :param X:
    :param y:
    :param seed:
    :return:
    """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def batch_iterator(X, y=None, batch_size=64):
    """
     Simple batch generator
    :param X:
    :param y:
    :param batch_size:
    :return:
    """
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin: end], y[begin: end]
        else:
            yield X[begin: end]

def divide_on_feature(X: np.ndarray, feature_i: int, threshold: Optional[int]):
    """
    Divide dataset based on if sample value on feature index is larger than the given threshold.
    :param X:
    :param feature_i:
    :param threshold:
    :return:
    """
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])
    return np.array([X_1, X_2])

def polynomial_features(X: np.ndarray, degree: int):
    """
    如果有a，b两个特征，那么它的2次多项式为（1,a,b,a^2,ab, b^2）
    :param X:
    :param degree: 控制多项式的度
    :return:
    """
    n_samples, n_features = np.shape(X)

    def index_combinations():
        # combinations_with_replacement 相比于 combinations，特点是可以重复选择元素
        combs = [combinations_with_replacement(range(n_features), i) for i in range(degree+1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs # 例如对于【combinations_with_replacement(range(2), i) for i in range(4)】, 可得
        # [(), (0,), (1,), (0, 0), (0, 1), (1, 1), (0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]

    combinations = index_combinations()
    n_output_features = len(combinations)
    x_new = np.empty((n_samples, n_output_features))

    for i, index_comb in enumerate(combinations):
        x_new[:, i] = np.prod(X[:, index_comb], axis=1)

    return x_new

def get_random_subsets(X: np.ndarray, y: np.ndarray, n_subsets: int, replacements: bool=True)->List[list]:
    """ 随机返回数据集中的一个子集

    :param X:
    :param y:
    :param n_subsets:
    :param replcements: 是否完全替换
    :return:
    """
    n_samples = np.shape(X)[0]
    # Concatenate x and y and do a random shuffle
    X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(X_y)
    subsets = []

    # Uses 50% of training samples without replacements
    subsample_size = int(n_samples//2)
    if replacements:
        subsample_size = n_samples          # 100% with replacements

    for _ in range(n_subsets):
        idx = np.random.choice(
            range(n_samples),
            size=np.shape(range(subsample_size)),
            replace=replacements
        )
        X = X_y[idx][:, :, -1]
        y = X_y[idx][:, -1]
        subsets.append([X, y])
    return subsets

def normalize(X: np.ndarray, axis=-1, order=2)->np.ndarray:
    """ Normalize the dataset X """
    # np.atleast_1d: 将输入转化为：至少一维的数组
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2==0] = 1
    return X / np.expand_dims(l2, axis)

def standardize(X: np.ndarray):
    """ Standardize the dataset X """
    X_std = X.copy()
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    return X_std

def train_test_split(X:np.ndarray, y, test_size:int=0.5, shuffle:bool=True, seed=None):
    """ 参考 sklearn 的函数"""
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in test_size
    split_idx = len(y) - int(len(y)*test_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test

def k_fold_cross_validation_sets(X: np.ndarray, y: np.ndarray, k: int, shuffle: bool=True):
    if shuffle:
        X, y = shuffle_data(X, y)

    n_samples = len(y)
    left_overs = {}
    n_left_overs = n_samples%k
    if n_left_overs != 0:
        left_overs["X"] = X[-n_left_overs:]
        left_overs["y"] = y[-n_left_overs:]

    X_split = np.split(X, k) # return list，包含k个 ndarray
    y_split = np.split(y, k)
    sets = []
    for i in range(k):
        X_test, y_test = X_split[i], y_split[i]
        X_train = np.concatenate(X_split[:i] + X_split[i+1:], axis=0)
        y_train = np.concatenate(y_split[:i] + y_split[i + 1:], axis=0)
        sets.append([X_train, X_test, y_train, y_test])

    # Add left over samples to last set as training samples
    if n_left_overs != 0:
        np.append(sets[-1][0], left_overs["X"], axis=0)
        np.append(sets[-1][2], left_overs["y"], axis=0)

    return np.array(sets)

def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x)+1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

def to_nominal(x):
    """ Conversion from one-hot encoding to nominal """
    return np.argmax(x, axis=1)

def make_diagonal(x):
    """ Converts a vector into an diagonal matrix """
    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])):
        m[i, i] = x[i]
    return m


if __name__=="__main__":

    tmp = [combinations_with_replacement(range(2), i) for i in range(5)]
    flat_combs = [item for sublist in tmp for item in sublist]

    # print(flat_combs)

    data = np.arange(50).reshape(10, 5)
    data[data==0] = -5

    print(np.amax(data), '\n', standardize(data))