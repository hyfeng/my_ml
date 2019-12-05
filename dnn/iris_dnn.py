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

debug = True

espi = 0.000000001

class Layer:
    def __init__(self, unit_num, feature_num):
        self.unit_num = unit_num
        ##增加一个偏置
        self.feature_num = feature_num + 1
        self.w = np.random.uniform(-0.5, 0.5, (unit_num, self.feature_num))
        self.activate = sigmoid
        ##学习率
        self.lam = 0.01
        self.x = None

    def print(self):
        s = "{}".format(self.w)
        return s

    def forward(self, x):
        """x是列向量,x一定是二维的，即是只有一个向量"""
        """x增加偏置项"""
        if debug:
            print("x:shape[{}]".format(x.shape))
        self.x = np.row_stack((x, np.ones(x[0].shape, dtype=x.dtype)))
        if debug:
            print("self.x:shape[{}]".format(self.x.shape))
        if len(x.shape) <= 1:
            print("x input is illegal; x must be a matrix, column is feature vector")
            exit(1)
        dot_mul = np.dot(self.w, self.x)
        ret = self.activate.forward(dot_mul)
        return ret

    def bprob(self, G):
        """G是外部输入的对layer输出的梯度"""
        """将梯度传递给输入层，即对输入的梯度"""
        """去掉对偏置项的梯度"""
        x = self.x
        acv_i = np.dot(self.w, x)
        act_g = self.activate.bprob(acv_i)
        if debug:
            print("w:shape[{}]".format(self.w.shape))
            print("x:shape[{}]".format(x.shape))
            print("acv_i:shape[{}]".format(acv_i.shape))
            print("act_g:shape[{}]".format(act_g.shape))
            print("G:shape[{}]".format(G.shape))
        #按元素乘法
        tmp_G = G * act_g
        #点积
        ret_G = np.dot(self.w.T, tmp_G)
        return ret_G[:-1]

    def update_G(self, G):
        """更新自己的参数"""
        x = self.x
        acv_i = np.dot(self.w, x)
        ##激活函数输出的梯度
        act_g = self.activate.bprob(acv_i)
        tmp_G = G * act_g
        w_g = np.dot(tmp_G, x.T)
        self.w = self.w - (w_g * self.lam)

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
        tmp = [-(y_list[i] / (x_list[i] + espi) - (1 - y_list[i])/(1-x_list[i]+espi)) for i in range(len(y_list))]
        return np.array(tmp).T


class DNN:
    def __init__(self):
        self.layers = []
        self.total_unit_num = 0
        self.loss = CrossEntropyLoss()

    def add_layer(self, unit_num, feature_num):
        """每层最增加额外的偏置"""
        layer = Layer(unit_num, feature_num)
        self.layers.append(layer)
        self.total_unit_num += unit_num

    def set_train(self, train, label):
        self.train_data = train
        self.train_label = label

    def forward(self, x):
        """x是列向量，需要增加偏置项"""
        global debug
        tmp_out = x
        i = 0
        for layer in self.layers:
            if debug:
                print("fw layer[{0}]\ninput:\n{1}".format(i, tmp_out.ravel()))
                i+=1
            tmp_out = layer.forward(tmp_out)
        return tmp_out

    def backward(self, loss_vector):
        tmp_out = loss_vector
        la = len(self.layers)
        for layer in self.layers[::-1]:
            la -= 1
            print("layer[{}]:bp_degrade[shape:{}]\n{}".format(la,
                tmp_out.shape, tmp_out.ravel()))
            bp = layer.bprob(tmp_out)
            layer.update_G(tmp_out)
            tmp_out = bp

    def print(self):
        i = 0
        for layer in self.layers:
            print("layer[{}]:\n{}".format(i, layer.print()))
            i += 1
        
    def forward_one(self, idx):
        global debug
        x = np.array(self.train_data[idx]).reshape((1,4)).T
        fw_ret = self.forward(x)
        if debug:
            print("fw_ret:shape[{}]".format(fw_ret.shape))
            print("fw_ret:{}".format(fw_ret.ravel()))
            print("label:{}".format(self.train_label[idx].ravel()))
        loss = self.loss.loss(self.train_label[idx], fw_ret)
        bp = self.loss.jacbi(self.train_label[idx], fw_ret)
        print("loss:{}".format(loss))
        print("bp:{}".format(bp.ravel()))
        return [loss, bp]

    def fit(self, epoch, batch):
        """默认使用交叉熵损失函数"""
        size = len(self.train_data)
        print("===== init ======")
        self.print()
        print("+++++++++++++")
        for i in range(epoch):
            print("======epoch[{}]=======".format(i))
            times = size // batch
            for ts in range(times):
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
            self.print()
            print("+++++++++++")

    def predict(self,x_vector):
        """端到端，x_vector是行向量，且未加偏置项"""
        x = np.array(x_vector).reshape((1,4)).T
        fw_ret = self.forward(x).ravel()
        print("fw_ret:{}".format(fw_ret))
        idx = np.argmax(fw_ret)
        res = np.zeros(fw_ret.shape, dtype=np.int32)
        res[idx] = 1
        return res


def load_data():
    train_data = np.load("train_data.npy")
    train_out = np.load("train_label.npy")
    test_data = np.load("test_data.npy")
    test_label = np.load("test_label.npy")
    return [train_data, train_out, test_data, test_label]

if __name__ == "__main__":
    train_data, train_label, test_data, test_label = load_data()
    dnn = DNN()
    dnn.add_layer(20, 4)
    dnn.add_layer(20,20)
    dnn.add_layer(4,20)
    dnn.set_train(train_data, train_label)
    dnn.fit(10, 10)

    for i in range(len(test_data)):
        x = test_data[i]
        lb = test_label[i]
        l = dnn.predict(x)
        print("predict:{}\nlabel:{}\n".format(l, lb))


    print("finished")
