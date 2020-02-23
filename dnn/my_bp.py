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
import numpy as np
import random
import math

#学习异或逻辑
x = np.array([[1,0],[0,1],[0,0],[1,1]])
y = np.array([1,1,0,0]).reshape((1,4))
#x = np.array([[1,0]])
#y = np.array([1]).reshape((1,1))

layer = [2,1] #0层是输入层，这里2个隐藏层
epoch = 200 #训练200次
alpha = 0.1 #学习率

def costfunction(predict, label):
    #最大似然函数
    #predict 一列是一个样本的预测
    return -label * np.log(predict) - (1-label) * np.log(1.000001 - predict)

def diff_costfunction(predict, label):
    return (1 - label) / (1 - predict + 0.000001) - label / (predict + 0.0000001)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def diff_sigmoid(z):
    return sigmoid(z) * (1-sigmoid(z))

b = []  #0层是输入层，无偏置
W = []  #0层是输入层，无参数

b.append(np.array([]))
W.append(np.array([]))
for i in range(1, len(layer)):
    b.append(np.zeros(shape = (layer[i], 1)))
    W.append(np.random.random(size = (layer[i], layer[i - 1]))) #初始化参数
print("init")
print("b shape:{}\n :{}".format(b[1].shape, b))
print("W shape:{}\n:{}".format(W[1].shape, W))

m = len(y)

#训练阶段
for i in range(epoch):
    #a0是输入层，一列代表一个样本
    #前向传播
    print("epoch:%d" % i)
    a = [0] * len(layer)
    z = [0] * len(layer)
    delta = [0] * len(layer)
    print("a:{}".format(a))
    a[0] = x.T
    for j in range(1, len(layer)):
        print("b[{}] shape:{}".format(j, b[j].shape))
        z[j] = np.dot(W[j], a[j - 1]) + b[j]
        a[j] = sigmoid(z[j])
    #反向传播
    predict = a[-1]
    print("epoch:{} predict:{}".format(i, predict))
    cost_f = costfunction(predict, y)
    print("epoch:{} cost:{}".format(i, np.sum(cost_f, axis=1)))
    delta[-1] = diff_costfunction(predict, y)
    print("echo:{} delta:{}".format(i, delta[-1]))
    for j in range(1, len(layer)):
        j = -j
        '''
        print("layer:{}".format(j))
        print("delta[{}]:{}".format(j, delta[j]))
        print("diff_sigmod[z[{}]]:{}".format(j, diff_sigmoid(z[j])))
        '''
        delta_z = delta[j] * diff_sigmoid(z[j])
        '''
        print("W[{}].T shape:{}".format(j, W[j].T.shape))
        print("delta_z shape:{}".format(delta_z.shape))
        '''
        delta[j - 1] = np.dot(W[j].T, delta_z)
        #更新参数
        delta_w = np.dot(delta_z, a[j].T)
        delta_b = delta_z
        arg_delta_w = 1.0 / m * np.sum(delta_w, axis = 1, keepdims=True)
#        print("arg_delta_w:{}".format(arg_delta_w))
        arg_delta_b = np.sum(delta_b, axis = 1, keepdims=True) / m
        W[j] = W[j] - alpha * arg_delta_w
#        print("b[{}]:{}".format(j,b[j]))
#        print("arg_delta_b:{}".format(arg_delta_b))
        b[j] = b[j] - alpha * arg_delta_b
#        print("b[{}]:{}".format(j,b[j]))

def predictfunction(test):
    """一列一个样本"""
    a = [0] * len(layer)
    z = [0] * len(layer)
    a[0] = test 
    for j in range(1, len(layer)):
        z[j] = np.dot(W[j], a[j - 1]) + b[j]
        a[j] = sigmoid(z[j])
#        print("layer:{},\n W:{},\nb:{}\n z:{}\na:{}".format(j,W[j], b[j], z[j], a[j]))
    return a[-1][0]

print("test")
test = np.array([[1,0],[0,0],[0,1],[1,1]]).T
test_label = np.array([1,0,1,0]).reshape((1,4))
predict = predictfunction(test)
print("test:{}".format(test))
print("test predict:{}".format(predict))
print("test: cost:{}".format(costfunction(predict, test_label)))
