import numpy as np

class LeastSquareRegressionTree:
    def __init__(self, train_X, y, epsilon):
        self.x = train_X
        self.y = y
        self.feature_count = train_X.shape[1]         # 特征总数
        self.epsilon = epsilon                                  # 阈值
        self.tree = None

    def _fit(self, x, y, feature_count, epsilon):
        # 选择最优切分点变量 j 与 切分点 s
        (j, s, minval, c1, c2) = self._divede(x, y, feature_count)

    def fit(self):
        self.tree = self._fit(self.x, self.y, self.feature_count, self.epsilon)

    @staticmethod # 外部不需要创建实例，就可以调用该方法
    def _divede(x, y, feature_count):
        # 初始化 loss
        cost = np.zeros((feature_count, len(x)))
        # eq 5.21
        for i in range(feature_count): # i为特征切割点
            for k in range(len(x)):


def main():
    train_X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T
    y = np.array([4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00])

    model_tree = LeastSquareRegressionTree(train_X, y, 0.2)



if __name__=="__main__":
    main()