import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(fileName):
    """    加载数据集

    :param fileName: 文件路径
    :return:
        dataMat - 数据
        labelMat -  label
    """
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')  # 源文件使用制表符分割数据
        dataMat.append([float(i) for i in lineArr[0:2]])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def task1():
    """
    task 1: visualize dataset
    :return:
    """
    def showDataSet(dataMat, labeMat):
        """     数据可视化

        :param dataMat: 数据数组, array.size=(bz, 2)
        :param labeMat: 标签数组, array.size=(bz)
        :return:
        """
        data_pos, data_neg = [], [];
        for (x, y) in zip(dataMat, labeMat):
            if (y==-1): data_neg.append(x);
            else: data_pos.append(x);
        data_pos = np.array(data_pos)
        data_neg = np.array(data_neg)
        plt.scatter(np.transpose(data_pos)[0], np.transpose(data_pos)[1])
        plt.scatter(np.transpose(data_neg)[0], np.transpose(data_neg)[1])
        plt.show()

    dataMat, dataLabel = loadDataSet("../ML_master/SVM/testSet.txt")
    showDataSet(dataMat, dataLabel)

def main():
    task1()

if __name__ == "__main__":
    main()