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
        for i in range(feature_count):
            for k in range(len(x)):
                # k 行 i 列 的特征值
                value = x[k, i]
                y1 = y[np.where(x[:, i] <= value)]
                c1 = np.mean(y1)
                y2 = y[np.where(x[:, i] > value)]
                c2 = np.mean(y2)
                y1[:] = y1[:] - c1
                y2[:] = y2[:] - c2
                cost[i, k] = np.sum(y1**2) + np.sum(y2**2)
        # 选取最优损失误差点
        cost_index = np.where(cost==np.min(cost))
        # 选取第几个特征值
        j = cost_index[0][0]
        # 选取特征值的切分点
        s = cost_index[1][0]
        # 求两个区域的 c1, c2
        c1 = np.mean(y[np.where(x[:, j] <= x[s, j])])
        c2 = np.mean(y[np.where(x[:, j] > x[s, j])])
        return j, s, cost[cost_index], c1, c2


def main():
    train_X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T
    y = np.array([4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00])

    model_tree = LeastSquareRegressionTree(train_X, y, 0.2)



if __name__=="__main__":
    main()