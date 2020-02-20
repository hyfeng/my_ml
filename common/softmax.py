#!/bin/env python
#coding=utf8

#================================================================
#   Copyright (C) 2019 company. All rights reserved.
#   
#   file:softmax.py
#   author:yourname
#   date:2019-11-17
#   description:
#
#================================================================

import numpy as np

def softmax(arr):
    """输入时数组"""
    tmp = np.exp(arr)
    acc = np.sum(tmp)
    res = tmp / acc
    return res

def bprob(arr):
    exp_arr = np.exp(arr)
    s_arr = softmax(arr)
    acc = np.sum(s_arr)
    res = exp_arr - s_arr * acc
    return res

def degrade(arr):
    espi = 0.0000005
    res = np.empty_like(arr)
    for i in range(len(arr)):
        arr[i] += espi
        r = softmax(arr)
        arr[i] -= 2*espi
        l = softmax(arr)
        arr[i] += espi
        res[i] = np.sum((r - l ) / (2 * espi))
    return res

if __name__ =="__main__":
    arr = np.array(range(50))
    arr = (arr - 25) / 25
    print("raw=====")
    print(arr)
    print("bprob++++")
    bp = bprob(arr)
    print(bp)
    
    print("debrade---")
    deg = degrade(arr)
    print(deg)

    print("finished")
