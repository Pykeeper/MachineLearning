# -*- coding: utf-8 -*-
"""
knn_task
 
Created on Fri Apr  6 16:07:43 2018

@author: vector
"""

import numpy as np
import operator
import matplotlib.pyplot as plt
from os import listdir


filename = '/home/vector/program/machine_learning/Ch02/datingTestSet.txt'
filename1 = '/home/vector/program/machine_learning/Ch02/datingTestSet2.txt'
filename2 = '/home/vector/program/machine_learning/Ch02/digits/trainingDigits/0_0.txt'
director1 = '/home/vector/program/machine_learning/Ch02/digits/trainingDigits'
director2 = '/home/vector/program/machine_learning/Ch02/digits/testDigits'


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
    

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


#-- 归一化特征值
def autoNorm(dataset):
    dataset, label = file2matrix(filename)
    minVals = dataset.min(0) #-- 获取每列的最小值
    maxVals = dataset.max(0) #-- 获取每列的最大值
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    normDataSet = dataset - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


#-- knn的核心代码--分类器    
def classify0(inX, dataSet, labels, k): # inX 是用于分类的输入向量
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
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #-- items()把字典中的键、值以列表形式给出
    #-- sorted 对可迭代的对象进行排序操作，并返回一个新的list
    return sortedClassCount[0][0]
    

#-- 分类器针对约会网站的测试代码
def datingClassTest():
    k = 3
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestvecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestvecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestvecs:m, :], \
        datingLabels[numTestvecs:m], k)
        print('The classifier came back with: %d, the real answer is: %d' \
        % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print('The total number of errors is: %d' % errorCount)    
    print('The total error rate is: %f' % (errorCount/float(numTestvecs)))
    


#-- 约会网站预测函数
def classifyPerson():
    k = 3
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('pencentage of time spent playing vedio games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, k)
    print('You will probably like this person: ', resultList[classifierResult-1])


def plotDataSet(feature, label, color):
    dataset = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    x = dataset[:, 0]
    y = dataset[:, 1]
    #labels = ['A', 'A', 'B', 'B']
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
    

#-- k-近邻分类器的手写识别系统
#-- 数据准备：将图像转换为测试向量
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect
    
    
#-- 手写数字识别系统的测试代码
def handwritingClassTest():
    k = 3
    hwLabels =[]
    trainingFileList = listdir(director1) #-- 获取目录内容
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024)) 
    
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('/home/vector/program/machine_learning/Ch02/digits/trainingDigits/%s' % fileNameStr)
    
    
    testFileList = listdir(director2)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('/home/vector/program/machine_learning/Ch02/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, k)
        print('The classifier came back with: %d, the real answer is: %d' \
        % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print('\nThe total number of errors is: %d' % errorCount)
    print('\nThe total error rate is: %f' % (errorCount/float(mTest)))
    
    
handwritingClassTest()    
  


  