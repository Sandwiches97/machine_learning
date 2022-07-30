from math import log
from typing import List
from collections import defaultdict
import operator
## the three things DT need to do:
# 1. feature selection; 2. tree building; 3 tree clipping


## 1. feature selection
def calculate_ShannonEnt(dataSet: List[list])->float:
        """ ShannonEnt 是针对 随机变量 X 来说的，假设 dataSet 中有 n 个类别

        :param dataSet:
        :return: H(X) = -\sum_{i=1}^n p_i log(p_i) 对每个类别计算，再求和
        """
        num_samples = len(dataSet)
        y_records = defaultdict(int)
        for sample in dataSet:
                y = sample[-1]
                y_records[y] += 1
        shannonEnt = 0.0
        for it in y_records.values():
                shannonEnt -= it/num_samples * log(it/num_samples, 2)
        return shannonEnt

def feature_DataSet(dataSet: List[list], axis: int, value)->List[List]:
        """ 返回 仅包含轴axis特征=value 的数据集

        :param dataSet:
        :param axis: 选择数据集的特征
        :param value: 特征的值
        :return: 仅包含轴axis特征=value 的 样本集
        """
        res_DataSet = []
        for sample in dataSet:
                if sample[axis] != value: continue
                res_feature = sample[:axis]
                res_feature.extend(sample[axis+1:]) # 指定特征后，样本不再包含该特征
                res_DataSet.append(res_feature)
        return  res_DataSet

def calculate_Condition_ShannonEnt(dataSet, axis: int, features: set):
        condition_ShannonEnt = 0.0
        for feature in features:
                res_DataSet = feature_DataSet(dataSet, axis, feature)
                prob = len(res_DataSet)/len(dataSet)
                condition_ShannonEnt += prob * calculate_ShannonEnt(res_DataSet)
        return condition_ShannonEnt

def chooseBestFeature2splitDataset(dataSet: List[list])->int:
        """

        :param dataSet:
        :return: 最优的特征轴
        """
        num_Features = len(dataSet[-1])-1       # 特征数量
        total_ShannonEnt = calculate_ShannonEnt(dataSet)
        best_Info_Gain = 0.0
        best_Feature_axis = -1                               # 最优特征索引
        for axis in range(num_Features):
                featList = [sample[axis] for sample in dataSet]
                features = set(featList)              # axis特征的种类
                condition_ShannonEnt = calculate_Condition_ShannonEnt(
                        dataSet, axis, features
                )
                info_Gain = total_ShannonEnt - condition_ShannonEnt #  信息增益
                print(f'第 {axis: d} 个特征的信息增益为 {info_Gain: .4f}')
                if best_Info_Gain < info_Gain:
                        best_Info_Gain = info_Gain
                        best_Feature_axis = axis
        return best_Feature_axis

# 2. build a DT
# # 使用dict存储决策树，例如
# {"house": {
#         0: {
#                 "work": {
#                         0: "yes",
#                         1: "no"
#                 }
#         },
#         1: "yes"
# }}
def majorityCnt(y_List: list):
        """

        :param y_List:
        :return: 出现最多的类标签，即为预测标签
        """
        classCount = defaultdict(int)
        for vote in y_List:
                classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
        return sortedClassCount[0][0]

def build_Tree(dataSet, feature_name: List[str], selected_feature: list)->dict:
        """

        :param dataSet:
        :param feature_name:         待选择特征
        :param selected_feature:  记录被选择作为节点的特征
        :return:
        """
        y_List = [sample[-1] for sample in dataSet] # 最终标签，是否放贷
        if y_List.count(y_List[0])==len(y_List):         # 返回条件1：类别完全相同
                return y_List[0]
        if len(dataSet[0])==1:                                  # 返回条件2：没有特征了，只剩标签（这里也可以改成信息增益小于某个值）
                return majorityCnt(y_List)                   # 返回最多的标签类别
        best_Feature_axis = chooseBestFeature2splitDataset(dataSet)                  # 选择 最优特征索引
        best_Feature = feature_name[best_Feature_axis]                                     # str, 最优特征，作为节点
        selected_feature.append(best_Feature)                                                   # 存储被选择的特征
        myTree = {best_Feature:{}}                                                                      # 创建节点
        del(feature_name[best_Feature_axis])                                                      # 删除已选择特征

        best_feature_Values = [sample[best_Feature_axis] for sample in dataSet]
        best_feature_Classes = set(best_feature_Values)
        for value in best_feature_Classes:      # 对已选择的特征的每个取值，建立子节点
                myTree[best_Feature][value] = build_Tree(
                        feature_DataSet(dataSet, best_Feature_axis, value),
                        feature_name,
                        selected_feature
                )
        return myTree

## 使用决策数分类
def classify(inputTree, selected_features, testVec):
        firstNode = next(iter(inputTree))                       # 获取决策树结点
        secondDict = inputTree[firstNode]
        feature_idx = selected_features.index(firstNode)
        classLabel = None
        for key in secondDict.keys():
                if testVec[feature_idx] == key:
                        if type(secondDict[key]).__name__ == "dict":
                                classLabel = classify(secondDict[key], selected_features, testVec)
                        else:
                                classLabel = secondDict[key]
        return classLabel


if __name__ == "__main__":

        dataSet = [[0, 0, 0, 0, 'no'],         #数据集, 最后一个为类别，是否给贷款
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
        feature_names = ["年龄", "work", "house", "信贷情况"]
        selected_feature = []

        myTree = build_Tree(dataSet, feature_names, selected_feature)
        print(myTree, "\n", selected_feature)

        tmp = [1,2,4,2,4]
        print(tmp.index(2))


        testVec = [0,1]
        result = classify(myTree, selected_feature, testVec)
        if result == 'yes':
                print('放贷')
        if result == 'no':
                print('不放贷')