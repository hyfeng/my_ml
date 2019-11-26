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
import math
from sklearn import datasets

espi = 0.000000001

class Layer:
    def __init__(self, unit_num, feature_num):
        self.unit_num = unit_num
        self.feature_num = feature_num
        ##增加一个偏置
        self.w = np.random.uniform(-3, 3, (unit_num, feature_num))
        self.activate = sigmoid
        ##学习率
        self.lam = 0.2
        self.x = None

    def forward(self, x):
        """x是列向量,x一定是二维的，即是只有一个向量"""
        if len(x.shape) <= 1:
            print("x input is illegal; x must be a matrix, column is feature vector")
            exit(1)
        print("x:{}".format(x))
        print("w:{}".format(self.w))
        dot_mul = np.dot(self.w, x)
        ret = self.activate.forward(dot_mul)
        self.x = x
        return ret

    def bprob(self, G):
        """G是外部输入的对layer输出的梯度"""
        """将梯度传递给输入层，即对输入的梯度"""
        x = self.x
        acv_i = np.dot(self.w, x)
        act_g = self.activate.bprob(acv_i)
        #按元素乘法
        tmp_G = G * act_g
        #点积
        ret_G = np.dot(self.w.T, tmp_G)
        return ret_G

    def update_G(self, G):
        """更新自己的参数"""
        x = self.x
        acv_i = np.dot(self.w, x)
        print("sig input:\n{}".format(acv_i))
        ##激活函数输出的梯度
        act_g = self.activate.bprob(acv_i)
        print("sig bprob output:\n{}".format(act_g))
        tmp_G = G * act_g
        print("tmp_G:\n{}".format(tmp_G))
        w_g = np.dot(tmp_G, x.T)
        self.w = self.w - (w_g * self.lam)
        print("w:\n{}\n".format(self.w))

class CrossEntropyLoss:

    def loss(self, y_list, x_list):
        """输入时numpy数组"""
        global espi
        sum = 0.0
        for i in range(len(y_list)):
            sum += y_list[i] * math.log(x_list[i] + espi) +\
                    (1 - y_list[i]) * math.log(1 - x_list[i] + espi)
        return -sum

    def jacbi(self, y_list, x_list):
        global espi
        tmp = [-(y_list[i] / (x_list[i] + espi) + (1 - y_list[i])/(1-x_list[i]+espi)) for i in range(len(y_list))]
        return np.array(tmp).T


class DNN:
    def __init__(self):
        self.layers = []
        self.total_unit_num = 0
        self.loss = CrossEntropyLoss()

    def add_layer(self, unit_num, feature_num):
        """每层最增加额外的偏置"""
        layer = Layer(unit_num, feature_num + 1)
        self.layers.append(layer)
        self.total_unit_num += unit_num

    def set_train(self, train, label):
        self.train_data = train
        self.train_label = label

    def forward(self, x):
        tmp_out = x
        for layer in self.layers:
            tmp_out = layer.forward(tmp_out)
        return tmp_out

    def backward(self, loss_vector):
        tmp_out = loss_vector
        la = len(self.layers)
        for layer in self.layers[::-1]:
            print("layer[{}]; bp:{}".format(la, tmp_out))
            la -= 1
            bp = layer.bprob(tmp_out)
            layer.update_G(tmp_out)
            tmp_out = bp
        
    def forward_one(self, idx):
        x = np.array(self.train_data[idx]).reshape((1,4)).T
        x = np.append(x, [[1]], axis = 0)
        fw_ret = self.forward(x)
        loss = self.loss.loss(self.train_data[idx], fw_ret)
        bp = self.loss.jacbi(self.train_label[idx], fw_ret)
        return [loss, bp]

    def fit(self, epoch, batch):
        """默认使用交叉熵损失函数"""
        size = len(self.train_data)
        for i in range(epoch):
            print("======epoch[{}]=======".format(i))
            times = size // batch
            for ts in range(times):
                print("+++++++bach[{}]++++++".format(ts))
                idx = batch * ts
                count = 1
                loss_sum, bp_sum = self.forward_one(idx)
                for at in range(1,batch):
                    idx = idx + at
                    if idx >= size:
                        break
                    loss, bp = self.forward_one(idx)
                    loss_sum += loss
                    bp_sum += bp
                    count += 1
                loss_sum /= count
                bp_sum /= count
                self.backward(bp_sum.T)
                print("bp:{}".format(bp_sum.T))
                print("loss:{}".format(loss_sum))


def load_data():
    train_data = np.load("train_data.npy")
    train_out = np.load("train_label.npy")
    test_data = np.load("test_data.npy")
    test_label = np.load("test_label.npy")
    print("train_data:\n{}".format(train_data))
    print("train_out:\n{}".format(train_out))
    print("test_data:\n{}".format(test_data))
    print("test_label:\n{}".format(test_label))
    return [train_data, train_out, test_data, test_label]

if __name__ == "__main__":
    train_data, train_label, test_data, test_labbel = load_data()
    dnn = DNN()
    dnn.add_layer(4, 4)
    dnn.set_train(train_data, train_label)
    dnn.fit(6, 1)

    print("finished")
