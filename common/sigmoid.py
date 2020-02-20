#!/bin/env python
#coding=utf8

#================================================================
#   Copyright (C) 2019 company. All rights reserved.
#   
#   file:sigmoid.py
#   author:yourname
#   date:2019-11-16
#   description:
#
#================================================================
import numpy as np
import sys

def forward(arr):
    return 1.0 / (1.0 + np.exp(-arr))

def bprob(x_arr, y_degrade):
    """对外梯度计算函数, y_degrade是外部对每个输出的梯度"""
    if len(x_arr.shape) != 1 or len(y_degrade.shape) != 1:
        raise Exception("sigmoid bprob x_arr[{}] and y_degrade[{}] must be one dimention".format(x_arr.shape, y_degrade.shape))
    if len(x_arr) != len(y_degrade):
        print("sigmoid bporb x_arr and y_degrade must have same dimention", file = sys.stderr)
        exit(1)

    tmp = forward(x_arr)
    f = tmp * (1.0 - tmp)
    return y_degrade * f

def degrade(arr):
    epxi = 0.0000001
    b = forward(arr + epxi)
    a = forward(arr - epxi)
    return (b - a) / (2 * epxi)

if __name__ == "__main__":
    a = np.array([-1, 0, 1])
    d = np.array([1,1,1])
    print(forward(a))
    print("degrade")
    print(bprob(a, d))
    print("cal degrade")
    print(degrade(a))
