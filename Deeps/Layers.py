# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from Deeps import Utils as ut
from Deeps import Activation as act
from tensorflow.contrib import layers as tfclayers


def CommonConvLayer(Input, FilterShape, Stride=[1,1,1,1], Padding='SAME', RegularScale=None, Activation=act.Relu, Name=None):

    Filter = ut.GetVaribleWithRegularization(
        FilterShape, 
        tf.truncated_normal_initializer(stddev=5e-2),
        'Filter_'+Name,
        RegularScale=RegularScale
    )
    Conv = tf.nn.conv2d(Input, Filter, Stride, Padding)
    Output = Activation(Conv)
    return Output

def FullyConnectedLayer(Input, InChannels, OutChannels, RegularScale=0, Activation=act.Relu, Name=None):
    Weight = ut.GetVaribleWithRegularization(
        [InChannels, OutChannels],
        tf.truncated_normal_initializer(stddev=5e-2),
        'Weight_'+Name,
        RegularScale=RegularScale
    )
    Bias = ut.GetVarible(
        [OutChannels],
        tf.truncated_normal_initializer(stddev=5e-2),
        'Bias_'+Name,
    )
    FC = tf.matmul(Input, Weight) + Bias
    Output = Activation(FC)
    return Output

def PoolingLayer(Input, KernelShape, Stride, Padding='VALID', Pooling='max',Name=None):
    if Pooling=='max':
        Pool = tf.nn.max_pool(Input, KernelShape, Stride, Padding, name=Name)
    elif Pooling == 'avg':
        Pool = tf.nn.avg_pool(Input, KernelShape, Stride, Padding, name=Name)
    else:
        Pool = Pooling(Input, KernelShape, Stride, Padding, name=Name)
    return Pool


