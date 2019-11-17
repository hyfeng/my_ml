#!/bin/env python
#coding=utf8

#================================================================
#   Copyright (C) 2019 company. All rights reserved.
#   
#   file:iris_dnn.py
#   author:yourname
#   date:2019-11-17
#   description:iris数据使用dnn训练
#
#================================================================
import sys
sys.path.append("../common")
import sigmoid
import numpy as np
import random

class Layer:
    def __init__(self, unit_num, feature_num):
        self.unit_num = unit_num
        self.feature_num = feature_num
        ##增加一个偏置
        self.w = np.random.uniform(-3, 3, (unit_num, feature_num))
        self.activate = sigmoid
        ##学习率
        self.lam = 0.2

    def forward(self, x):
        """x是列向量,x一定是二维的，即是只有一个向量"""
        if len(x.shape) <= 1:
            print("x input is illegal; x must be a matrix, column is feature vector")
            exit(1)
        print("x:{}".format(x))
        print("w:{}".format(self.w))
        dot_mul = np.dot(self.w, x)
        ret = self.activate.forward(dot_mul)
        return ret

    def bprob(self, x, G):
        """G是外部输入的对layer输出的梯度"""
        """将梯度传递给输入层，即对输入的梯度"""
        acv_i = np.dot(self.w, x)
        act_g = self.activate.bprob(acv_i)
        #按元素乘法
        tmp_G = G * act_g
        #点积
        ret_G = np.dot(self.w.T, tmp_G)
        return ret_G

    def update_G(self, x, G):
        """更新自己的参数"""
        acv_i = np.dot(self.w, x)
        print("sig input:\n{}".format(acv_i))
        ##激活函数输出的梯度
        act_g = self.activate.bprob(acv_i)
        print("sig bprob output:\n{}".format(act_g))
        tmp_G = G * act_g
        print("tmp_G:\n{}".format(tmp_G))
        w_g = np.dot(tmp_G, x.T)
        self.w = self.w - (w_g * self.lam)


if __name__ == "__main__":
    x = np.array([[1,2,1]])
    x = x.T
    lay = Layer(2, 3)
    fw = lay.forward(x)
    print("fw:")
    print(fw)
    print("start bprob")
    G = np.array([0.3])
    G = G.T
    bp = lay.bprob(x, G)
    print("bp:")
    print("{}".format(bp))
    print("w:\n{}".format(lay.w))
    lay.update_G(x, G)
    print("up_w:\n{}".format(lay.w))
