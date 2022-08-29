import math
from copy import deepcopy
from typing import List

class MaxEntropy:
    def __init__(self, EPS=0.005):
        self._samples = []
        self._Yset = set()     # 标签集合
        self._freqXY = {}    # key: (x, y)，value: 出现次数
        self._Num_Samples = 0         # 样本数
        self._Ep_ = []      # 样本分布的特征期望值
        self._xy2Idx = {}   # key: 记录(x, y), value: id号
        self._num_xy = 0         # 特征键值 (x, y) 的个数
        self._maxCnt_Feat_Nums = 0         # 最大特征数
        self._Idx2xy = {}
        self._w = []
        self._EPS = EPS     # 收敛条件
        self._lastw = []    # 上一次 w参数值

    def loadData(self, dataset: List[list]):
        self._samples = deepcopy(dataset)
        for items in self._samples:
            y = items[0]
            X = items[1:]
            self._Yset.add(y)
            for x in X:
                if (x, y) in self._freqXY:
                    self._freqXY[(x, y)] += 1
                else:
                    self._freqXY[(x, y)] = 1

        self._Num_Samples = len(self._samples)
        self._num_xy = len(self._freqXY)
        self._maxCnt_Feat_Nums = max([len(sample)-1 for sample in self._samples])
        self._w = [0] *self._num_xy
        self._lastw = self._w[:]

        self._EP_ = [0]*self._num_xy
        for i, xy in enumerate(self._freqXY): # 计算特征函数 fi 关于经验分布的期望
            self._EP_[i] = self._freqXY[xy] / self._Num_Samples
            self._xy2Idx[xy] = i
            self._Idx2xy[i] = xy

    def _Z_x(self, X: List[list]):
        """ 计算特征每个 Z(x)的值, 归一化因子 """
        zx = 0
        for y in self._Yset:
            ss = 0
            for x in X:
                if (x, y) in self._freqXY:
                    ss += self._w[self._xy2Idx[(x, y)]]
            zx += math.exp(ss)
        return zx

    def _model_P_y_given_x(self, y, X: List[list]):
        """ 计算每个 p(y|X) = \sum_x {exp(wx+b)} / Z(X) """
        zx = self._Z_x(X)
        ss = 0
        for x in X:
            if (x, y) in self._freqXY:
                ss += self._w[self._xy2Idx[(x, y)]]
        pyx = math.exp(ss) / zx
        return pyx

    def _model_ep(self, idx):
        """ 计算特征函数 f_i 关于模型的期望 """
        x, y = self._Idx2xy[idx]
        ep = 0
        for sample in self._samples:
            if x not in sample:
                continue
            pyx = self._model_P_y_given_x(y, sample)
            ep += pyx / self._Num_Samples
        return ep

    def _convergence(self):
        for last, now in zip(self._lastw, self._w):
            if abs(last-now) >= self._EPS:
                return False
        return True

    def predict(self, X):   # 计算预测概率
        Z = self._Z_x(X)
        result = {}
        for y in self._Yset:
            pxy = self._model_P_y_given_x(y, X)
            result[y] = pxy
        return result

    def train(self, maxIter=1000):
        for loop in range(maxIter):
            print("iter: %d"%loop)
            self._lastw = self._w[:]
            for i in range(self._num_xy):
                ep = self._model_ep(i)      # 第 i 个特征的模型期望
                self._w[i] += math.log(self._EP_[i]/ep) / self._maxCnt_Feat_Nums
            print(f"w: {self._w}")
            if self._convergence():
                break




if __name__=="__main__":
    dataset = [['no', 'sunny', 'hot', 'high', 'FALSE'],
           ['no', 'sunny', 'hot', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'high', 'FALSE'],
           ['yes', 'rainy', 'mild', 'high', 'FALSE'],
           ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
           ['no', 'rainy', 'cool', 'normal', 'TRUE'],
           ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
           ['no', 'sunny', 'mild', 'high', 'FALSE'],
           ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
           ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
           ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
           ['yes', 'overcast', 'mild', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
           ['no', 'rainy', 'mild', 'high', 'TRUE']]

    maxEnt = MaxEntropy()
    x = ['overcast', 'mild', 'high', 'FALSE']
    maxEnt.loadData(dataset)
    maxEnt.train()