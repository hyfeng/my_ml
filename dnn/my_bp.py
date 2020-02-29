#!/bin/env python
#coding=utf8

#================================================================
#   Copyright (C) 2020 company. All rights reserved.
#   
#   file:my_bp.py
#   author:yourname
#   date:2020-02-22
#   description:自己写的bp
#
#================================================================
from __future__ import print_function, division
from sklearn.datasets import load_iris
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import func

def data_analysis(iris):
    """分析数据"""
    x = iris[0]
    y = iris[1]
    print(x.shape)
    print(y.shape)
    print(x[0])
    print(y[0])
    st = {}
    for k in y:
        if k in st:
            st[k] += 1
        else:
            st[k] = 1
    for c in st:
        print("class:{}, occure:{}".format(c, st[c]))

def prepare_data(iris):
    y = np.zeros(shape = (len(iris[1]), 3))
    for i, v in enumerate(iris[1]):
        y[i][v] = 1
    print(iris[1][0:3])
    print(y[0:3])
    print(iris[1][-1])
    print(y[-1])
    return [iris[0], y]

#学习异或逻辑
iris_feature = load_iris().data
iris_label = load_iris().target
data_analysis([iris_feature, iris_label])
x,y = prepare_data([iris_feature, iris_label])
train_size = 120
train_x = x[0:train_size]
train_y = y[0:train_size]
test_x = x[train_size:]
test_y = y[train_size:]
print(x.shape)
print(y.shape)

layer = [4,3,3] #0层是输入层，这里2个隐藏层
epoch = 2000 #训练200次
alpha = 0.3 #学习率

epoch_wb = []

train_cost = {}

b = []  #0层是输入层，无偏置
W = []  #0层是输入层，无参数

costfunction = func.entropy_costfunction
diff_costfunction = func.diff_entropy_costfunction
activate = func.sigmoid
diff_activate = func.diff_sigmoid

b.append(np.array([]))
W.append(np.array([]))
for i in range(1, len(layer)):
    b.append(np.random.random(size = (layer[i], 1)))
    W.append(np.random.random(size = (layer[i], layer[i - 1]))) #初始化参数

m = len(y)

#训练阶段
for i in range(epoch):
    #a0是输入层，一列代表一个样本
    #前向传播
    x = train_x.T
    y = train_y.T
    a = [0] * len(layer)
    z = [0] * len(layer)
    delta = [0] * len(layer)
    a[0] = x
    for j in range(1, len(layer)):
        z[j] = np.dot(W[j], a[j - 1]) + b[j]
        a[j] = activate(z[j])
    #反向传播
    predict = a[-1]
    cost_f = costfunction(predict, y)
    train_cost[i] = np.sum(cost_f, axis = 1)[0]
    delta[-1] = diff_costfunction(predict, y)
    for j in range(1, len(layer)):
        j = -j
        '''
        print("layer:{}".format(j))
        print("delta[{}]:{}".format(j, delta[j]))
        print("diff_sigmod[z[{}]]:{}".format(j, diff_sigmoid(z[j])))
        '''
        delta_z = delta[j] * diff_activate(z[j])
        '''
        print("W[{}].T shape:{}".format(j, W[j].T.shape))
        print("delta_z shape:{}".format(delta_z.shape))
        '''
        delta[j - 1] = np.dot(W[j].T, delta_z)
        #更新参数
        delta_w = np.dot(delta_z, a[j - 1].T) / m
        delta_b = np.sum(delta_z, axis = 1, keepdims = True) / m
        W[j] = W[j] - alpha * delta_w
        b[j] = b[j] - alpha * delta_b

def predictfunction(test):
    """一列一个样本"""
    a = [0] * len(layer)
    z = [0] * len(layer)
    a[0] = test 
    for j in range(1, len(layer)):
        z[j] = np.dot(W[j], a[j - 1]) + b[j]
        a[j] = activate(z[j])
#        print("layer:{},\n W:{},\nb:{}\n z:{}\na:{}".format(j,W[j], b[j], z[j], a[j]))
    return a[-1].T

p_x = [x for x in train_cost.keys()]
p_x.sort()
p_y = [train_cost[e] for e in p_x]
plt.figure(12)
plt.subplot(211)
plt.plot(p_x, p_y, "-")
print("test")
predict = predictfunction(test_x.T)

print(predict)
plt.show()

