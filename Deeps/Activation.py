# -*- coding: utf-8 -*-
# 定义了一个激活函数的集合，后面所有的激活函数都可以在里面进行定义和使用
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def Relu(Input):
    Output = tf.nn.relu(Input)
    return Output

def Sigmoid(Input):
    Output = tf.nn.sigmoid(Input)
    return Output
