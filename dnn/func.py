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
    """输出z(1,n)的softmax函数"""
    res = np.exp(z)
    total = np.sum(res,axis = 1)
    res = res / total
    return res

def diff_softmax(z):
    """z是(1,n)矩阵，是softmax的输入"""
    size = z.shape[1]
    res = np.zeros(shape = (size, size))
    soft_res = softmax(z)
    for i in range(size):
        for k in range(size):
            if i == k:
                res[k][i] = soft_res[0][i] * (1 - soft_res[0][i])
            else:
                res[k][i] = -(soft_res[0][i] * soft_res[0][i])
    print(res)
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
