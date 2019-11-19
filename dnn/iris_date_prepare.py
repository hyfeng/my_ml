#!/bin/env python
#coding=utf8

#================================================================
#   Copyright (C) 2019 company. All rights reserved.
#   
#   file:iris_date_prepare.py
#   author:yourname
#   date:2019-11-17
#   description:将iris数据分成两部分，train和test两部分
#           train部分的输出数据部分，需要转成dnn的形式
#
#================================================================
from sklearn import datasets
import random
import numpy as np

import sys
def label_2_dnn_label(labels):
    train_label_tr = []
    for i in labels:
        tmp = [0,0,0,0]
        tmp[i] = 1
        train_label_tr.append(tmp)
    return np.array(train_label_tr)


if __name__ == "__main__":
    iris = datasets.load_iris()
    data = iris.data
    label = iris.target
    size = len(data)
    test_size = 20
    train_size = size - test_size
    indexs = []
    while len(indexs) < test_size:
        a = random.randint(0,size - 1)
        if a not in indexs:
            indexs.append(a)
    choose = [False if i not in indexs else True for i in range(size)]
    test_data = data[choose]
    test_label = label_2_dnn_label(label[choose])
    np.save("test_data", test_data)
    np.save("test_label", test_label)
    print("test:{}\ntest_data:\n{}".format(len(test_data), test_data))
    train_ch = [not i for i in choose]
    train_data = data[train_ch]
    train_label = label[train_ch]
    train_label_tr = label_2_dnn_label(train_label)
    np.save("train_data", train_data)
    np.save("train_label", train_label_tr)
    print("train:{}\ntrain_label:\n{}".format(len(train_data), train_label_tr))



