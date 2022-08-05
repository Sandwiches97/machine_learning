from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = next(iter(myTree))       # iter() 将字典转化为 iterator
    secondDict = myTree[firstStr]
    for key in secondDict.key():
        if type(secondDict[key]).__name__ == "dict":
            # 递归，往下寻找叶子节点
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

# def createPlot(inTree):
#     """ 创建绘画板
#
#     :param inTree:
#     :return:
#     """
#     fig = plt.figure(1, facecolor="white")
#     fig.clf()
#     axprops = dict(xticks=[], yticks=[])
#     createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
#     plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
#     plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
#     plotTree.xOff = -0.5 / plotTree.totalW;
#     plotTree.yOff = 1.0;  # x偏移
#     plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
#     plt.show()

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")                                                                  # 定义箭头格式
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)          # 设置中文字体







def main():
    pass

if __name__ == "__main__":
    main()