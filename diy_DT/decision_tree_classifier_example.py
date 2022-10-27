# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import sys
import os

# Import helper functions
from utils import train_test_split, standardize, accuracy_score
from utils import mean_squared_error, calculate_variance, Plot
from decision_tree_model import ClassificationTree

def main():
    print("-- Classification Tree --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test = 0, 0