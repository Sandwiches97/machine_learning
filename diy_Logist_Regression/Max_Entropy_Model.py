import math
from copy import deepcopy

class MaxEntropy:
    def __init__(self, EPS=0.005):
        self._samples = []
        self._Y = set()     # 标签集合
        self._numXY = {}    # key: (x, y)，value: 出现次数
        self._N = 0         # 样本数
        self._Ep_ = []      # 样本分布的特征期望值
        self._xyID = {}     # key: 记录(x, y), value: id号
        self._n = 0         # 特征键值 (x, y) 的个数
        self._C = 0         # 最大特征数
        self._IDxy = {}     # key: (x, y), value: id号
        self._w = []
        self._EPS = EPS     # 收敛条件
        self._lastw = []    # 上一次 w参数值

    def loadData(self, dataset):
        self._samples = deepcopy(dataset)
        for items in self._samples:
            y = items[0]
            X = items[1:]
            self._Y.add(y)
            for x in X:
                if (x, y) in self._numXY:
                    self._numXY[(x, y)] += 1
                else:
                    self._numXY[(x, y)] = 1

        self._N = len(self._samples)
        self._n = len(self._numXY)
        self._C = max([len(sample)-1 for sample in self._samples])
        self._w = [0] *self._n
        self._lastw = self._w[:]

        self._EP_ = [0]*self._n
        for i, xy in enumerate(self._numXY): # 计算特征函数 fi 关于经验分布的期望
            self._EP_[i] = self._numXY[xy] / self._N
            self._xyID[xy] = i
            self._IDxy[i] = xy

    def _Zx(self, X):
        """ 计算特征每个 Z(x)的值

        :param X:
        :return:
        """
        zx = 0
        for y in self._Y:
            ss = 0
            for x in X:
                if (x, y) in self._numXY:
                    ss += self._w[self._xyID[(x, y)]]
            zx += math.exp(ss)
        return zx

    def _model_pyx(self, y, X):
        """ 计算每个 p(y|X)

        :param y:
        :param X:
        :return:
        """
        zx = self._Zx(X)
        ss = 0
        for x in X:
            if (x, y) in self._numXY:
                ss += self._w[self._xyID[(x, y)]]
        pyx = math.exp(ss) / zx
        return pyx

    def _model_ep(self, idx):
        x, y = self._IDxy[idx]
        ep = 0
        for sample in self._samples:
            if x not in sample:
                continue
            pyx = self._model_pyx(y, sample)
            ep += pyx / self._N
