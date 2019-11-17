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

def forward(arr):
    return 1.0 / (1.0 + np.exp(-arr))

def bprob(arr):
    tmp = forward(arr)
    return tmp * (1.0 - tmp)
def degrade(arr):
    epxi = 0.0000001
    b = forward(arr + epxi)
    a = forward(arr - epxi)
    return (b - a) / (2 * epxi)

if __name__ == "__main__":
    a = np.array([-1, 0, 1])
    print(forward(a))
    print("degrade")
    print(bprob(a))
    print("cal degrade")
    print(degrade(a))
