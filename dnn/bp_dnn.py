#!/bin/env python
#coding=utf8

#================================================================
#   Copyright (C) 2020 company. All rights reserved.
#   
#   file:bp_dnn.py
#   author:yourname
#   date:2020-02-20
#   description:BP神经网络
#
#================================================================

from __future__ import print_function, division
import numpy as np
import math
import copy
import sklearn.datasets
import matplotlib.pyplot as plt

trainingSet, trainingLabels = sklearn.datasets.make_moons(400, noise = 0.2)

plt.scatter(trainingSet[trainingLabels == 1][:,0], trainingSet[trainingLabels == 1][:,1], s = 40, c = 'r', marker = 'x', cmap = plt.cm.Spectral)
plt.scatter(trainingSet[trainingLabels == 0][:,0], trainingSet[trainingLabels == 0][:,1], s = 40, c = 'y', marker = '+', cmap = plt.cm.Spectral)
plt.show()
testSet = trainingSet[320:]
testLabels = trainingLabels[320:]
trainingSet = trainingSet[:320]
trainingLabels = trainingLabels[:320]

#设置网络参数
layer = [2,3,1] #设置网络层数和结点数
Lambda = 0.005  #正则化系数
alpha = 0.2     #学习速率
num_passes = 20000 #迭代次数
m = len(trainingSet) #样本数量

#建立网络
b = [] #偏置 b[0]第一个隐藏层
W = []
for i in range(len(layer) - 1):
    W.append(np.random.random(size = (layer[i + 1], layer[i]))) #W[i]表示第i层到i+1层的参数
    b.append(np.array([0.1] * layer[i+1])) #b[i]是

a = [np.array(0)] * (len(W) + 1) # a[0] = x 是输入层

z = [np.array(0)] * len(W)

W = np.array(W)

def costfunction(predict, labels):
    return sum((predict - labels) ** 2)

def error_rate(predict, labels):
    error = 0.0
    for i in range(len(predict)):
        if predict[i] != labels[i]:
            error += 1
    return error / len(predict)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def diff_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

activation_function = sigmoid
diff_activation_function = diff_sigmoid

#开始训练
a[0] = np.array(trainingSet).T
y = np.array(trainingLabels)

for v in range(num_passes):
    #前向传播
    for i in range(len(W)):
        z[i] = np.dot(W[i], a[i])
        for j in range(m):
            z[i][:,j] += b[i]
        a[i+1] = activation_function(z[i])
    predict = a[-1][0]
    #反向传播
    delta = [np.array(0)] * len(W) #delta[-1]是输出层的残差
    #计算输出层残差
    delta[-1] = -(y - a[-1]) * diff_activation_function(z[-1])
    #计算第二层起除输出层外的残差
    for i in range(len(delta) - 1):
        delta[-i - 2] = np.dot(W[-i - 1].T, delta[-i-1]) * diff_activation_function(z[-i-2])
    #计算偏置倒数
    delta_w = [np.array(0)] * len(W)
    delta_b = [np.array(0)] * len(W)
    for i in range(len(W)):
        delta_w[i] = np.dot(delta[i], a[i].T)
        delta_b[i] = np.sum(delta[i], axis = 1)
    #更新权值
    for i in range(len(W)):
        W[i] -= alpha * (Lambda * W[i] + delta_w[i] / m)
        b[i] -= alpha / m * delta_b[i]
    print("训练样本未正则化的代价函数:{}".format(costfunction(predict, np.array(trainingLabels))))
    print("训练样本错误率:{}".format(error_rate(predict, np.array(trainingLabels))))
        
#使用测试集测试
a[0] = np.array(testSet).T
#前向
m = len(testSet)
for i in range(len(W)):
    z[i] = np.dot(W[i], a[i])
    for j in range(m):
        z[i][:,j] += b[i].T[0]
    a[i+1] = activation_function(z[i])
predict = a[-1][0]
print("测试样本的未正则化代价函数:{}".format(costfunction(predict, np.array(testLabels))))
print("测试样本的错误率:{}".format(error_rate(predict, np.array(testLabels))))

