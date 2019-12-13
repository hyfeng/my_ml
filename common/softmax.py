#!/bin/env python
#coding=utf8

#================================================================
#   Copyright (C) 2019 All rights reserved.
#   
#   file:softmax.py
#   author:houyufeng
#   date:2019-12-12
#   description:
#
#================================================================
import numpy as np
import sys

def forward(x_arr):
    res = np.exp(x_arr)
    acc = np.sum(res)
    return res / acc

def bprob(x_arr, y_degrade):
    """x_arr是原始输入,y_degrade是总输出对每个输出结果的梯度,都是一维数组"""
    if len(x_arr.shape) != 1 or len(y_degrade.shape) != 1:
        print("softmax input must be one dimention", file = sys.stderr)
        exit(1)
    if len(x_arr) != len(y_degrade):
        print("softmax bprob x_arr and y_degrade must have same dimention", file = sys.stderr)
        exit(1)

    res = forward(x_arr)
    rd = []
    for i in range(len(x_arr)):
        i_bp = 0.0
        for j in range(len(x_arr)):
            f = -(res[i] * res[j])
            if i == j:
                f += res[i]
            print("idx:{} f:{}".format(i, f))
            i_bp += y_degrade[j] * f
        print("idx:{} bp:{}".format(i, i_bp))
        rd.append(i_bp)
    return np.array(rd)

if __name__ == "__main__":
    a = np.array([1,2,3])
    print("forward:{}".format(forward(a)))
    d = np.array([1,1,1])
    print("bp:{}".format(bprob(a, d)))

