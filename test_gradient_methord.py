#!/bin/env python
#coding=utf8

#================================================================
#   Copyright (C) 2019 oppo. All rights reserved.
#   
#   file:test_gradient_methord.py
#   author:houyufeng
#   date:2019-10-30
#   description:
#
#================================================================
import math
import sys

def x_square(x):
    return x*x

def gradient(f, x):
    espison = 0.000001
    delta_y = f(x + espison) - f(x - espison)
    delta_x = 2 * espison
    return delta_y / delta_x

if __name__ == "__main__":
    print("%.3f", gradient(x_square, 1))

