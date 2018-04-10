# -*- coding: utf-8 -*-
"""
knn_task
 
Created on Fri Apr  6 16:07:43 2018

@author: vector
"""

import numpy as np
import operator
import matplotlib.pyplot as plt

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
    

filename = '/home/vector/program/machine_learning/Ch02/datingTestSet.txt'
def file2matrix(filename):
    f = open(filename)
    arrayOlines = f.readlines()
    numberOfLines = len(arrayOlines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0 
    for line in arrayOlines:
        line = line.strip() #-- 截取所有的回车字符   
        listFromLine = line.split('\t') #-- 对字符串进行切片
        returnMat[index, :] = listFromLine[0:3]
        labels = {'didntLike':1, 'smallDoses':2, 'largeDoses':3} 
        classLabelVector.append(labels[listFromLine[-1]])
        index += 1
    return returnMat, classLabelVector
 
file2matrix(filename)    
print(returnMat, classLabelVector)

'''    
#def plotDataSet(feature, label, color):
dataset = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
x = dataset[:, 0]
y = dataset[:, 1]
labels = ['A', 'A', 'B', 'B']
colors = ['r', 'r', 'g', 'g']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Scatter Plot')
ax.scatter(x, y, s=20, color=colors, alpha=1) # s 大小 alpha 透明度
#type1 = ax.scatter(x, y, s=20, c='red')
#type2 = ax.scatter(x, y, s=20, c='g')
#plt.legend((type1, type2), ('A', 'B'), loc=2)
plt.legend()
plt.show()
'''    

#-- knn的核心代码    
def classify0(inX, dataSet, labels, k):
    #-- 计算距离
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    
    #-- 获取k个相邻数据标签并排序
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]    
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #-- get 获取字典中的值
        print(classCount)  
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) 
    #-- sorted 对可迭代的对象进行排序操作，并返回一个新的list
    return sortedClassCount[0][0]


'''  
if __name__ == '__main__':
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    colors = ['r', 'g']
    plotDataSet(group, labels)
'''    
    