# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np

from utils import divide_on_feature, train_test_split, standardize, mean_squared_error
from utils import calculate_entropy, accuracy_score, calculate_variance

class DecisionNode():
    """Class that represents a decision node or leaf in the decision tree
    Parameters:
    -----------
    feature_i: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """
    def __init__(self, feature_i=None, threshold: int=None,
                 value: int=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i              # 作为评估的特征索引
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch          # 左子树
        self.false_branch = false_branch        # 右子树

# Super class of Regression Tree and Classification Tree
class DecisionTree(object):
    """Super class of RegressionTree and ClassificationTree.
    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.
    """
    def __init__(self, min_samples_split: int=2, min_impurity: float=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None
        # 分割的最小样本数
        self.min_samples_split = min_samples_split
        # 分割的最小非纯度
        self.min_impurity = min_impurity
        # 生成一棵树的最大深度
        self.max_depth = max_depth
        # 计算非纯度的函数，切割树的方法，gini系数、方差等
        self._impurity_calculation = None
        # 决定叶子节点 y 的预测值的函数，
        # 分类树：选取出现最多次数的值；回归树：取所有值的平均值
        self._leaf_value_calculation = None
        # 决定 y 是否为 one_hot 编码（multi-dim） 或者 不是独热编码（one-dim）
        self.one_dim = None
        # if Gradient Boost
        self.loss = loss

    def fit(self, X: np.ndarray, y, loss=None):
        # Build decision tree
        self.one_dim = len(np.shape(y))==1
        self.root = self._build_tree(X, y)
        self.loss = None

    def _build_tree(self, X: np.ndarray, y: np.ndarray, current_depth: int=0):
        """
        递归方法
        :param X:
        :param y:
        :param current_depth:   树的深度
        :return:                a Decision Node
        """
        largest_impurity = 0
        best_criteria = None            # Feature index and Threshold
        best_sets = None                # Subsets of the data

        # Check if expansion of y is needed
        if len(np.shape(y))==1:
            y = np.expand_dims(y, axis=1)
        # Add y as last column of X
        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each features
            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)       # 类似 set 去重的作用

                # Iterate through all unique values of feature column i
                # and calculate the impurity
                for threshold in unique_values:
                    # Divide X and y depending on if the feature value of X
                    # at index feature_i meets the threshold
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Select the y-values of the two sets
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # Calculate impurity
                        impurity = self._impurity_calculation(y, y1, y2)

                        # If this threshold resulted in a higher information gain than
                        # previously recorded save the threshold value and the feature index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i,
                                             "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],  # X of left subtree
                                "lefty": Xy1[:, n_features:],  # y of left subtree
                                "rightX": Xy2[:, :n_features], # X of right subtree
                                "righty": Xy2[:, n_features:]  # y of right subtree
                            }
        if largest_impurity > self.min_impurity:
            # build subtrees for the right and left branches
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth+1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["lefty"], current_depth+1)
            return DecisionNode(feature_i=best_criteria["feature_i"],
                                threshold=best_criteria["threshold"],
                                true_branch=true_branch,
                                false_branch=false_branch)

        # We're at Leaf Node => determine value
        leaf_value = self._leaf_value_calculation(y)
        return DecisionNode(value=leaf_value)

    def predict_value(self, X, tree: DecisionNode=None):
        """ Do a recursive search down the tree and
        make a predicion of the data sample by the value of the leaf that we end up at
        :param X:
        :param tree:
        :return:
        """
        if tree is None:
            return self.root

        # If we have a value (i.e. leaf node) => return value as the prediction
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = X[tree.feature_i]

        # Determine if we will follow left or right branch
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        # Test subtree
        return self.predict_value(X, branch)

    def predict(self, X: np.ndarray):
        """ Classify samples one by one and return the set of labels """
        y_pred = []
        for x in X:
            y_pred.append(self.predict_value(x))
        return y_pred

    def print_tree(self, tree: DecisionNode=None, indent=" "):
        """ Recursively print the decision tree """
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.value is not None:
            print(tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print("%s:%s? "%(tree.feature_i, tree.threshold))
            # Print the true scenario
            print("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent+indent)
            # Print the false scenario
            print("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent+indent)

class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self, y, y1, y2):
        # Calculate information gain, 二分类
        p = len(y1)/len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p*calculate_entropy(y1) - (1-p)*calculate_entropy(y2)
        return info_gain

    def _majority_vote(self, y: np.ndarray):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurences of samples with label
            count = len(y[y==label])
