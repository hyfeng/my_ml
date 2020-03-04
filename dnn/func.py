#!/bin/env python
#coding=utf8

#================================================================
#   Copyright (C) 2020 company. All rights reserved.
#   
#   file:func.py
#   author:yourname
#   date:2020-02-29
#   description:
#
#================================================================
import numpy as np

def softmax(z):
    """输出z(m,n)的softmax函数,m个样本,n是softmax的单元"""
    res = np.exp(z)
    total = np.sum(res,axis = 1)
    res = res / total
    return res

def diff_softmax(z):
    """z是(m,n)矩阵，是softmax的输入,m个样本，n个单元"""
    """返回一个二维数组，第i列是第i个输出对所有输入的梯度,多个样本输入的话，返回三维数组"""
    size = z.shape[1]
    cc = z.shape[0]
    res = np.zeros(shape = (cc, size, size))
    soft_res = softmax(z)
    for j in range(cc):
        for i in range(size):
            for k in range(size):
                if i == k:
                    res[j][k][i] = soft_res[j][i] * (1 - soft_res[j][i])
                else:
                    res[j][k][i] = -(soft_res[j][i] * soft_res[j][i])
    return res

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def diff_sigmoid(z):
    return sigmoid(z) * (1-sigmoid(z))

def entropy_costfunction(predict, label):
    #最大似然函数
    #predict 一列是一个样本的预测
    return -label * np.log(predict + 0.0000001) - (1-label) * np.log(1.000001 - predict)

def diff_entropy_costfunction(predict, label):
    return (1 - label) / (1 - predict + 0.000001) - label / (predict + 0.0000001)

if __name__ == "__main__":
    a = np.array([[1,2,3,4]])
    b = softmax(a)
    print(b)
    diff_softmax(b)
