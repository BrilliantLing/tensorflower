# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers as tfclayers

def GetSummary(x):
    return

def FilterSummary(Filter):
    return

def GetVarible(Shape, Init, Name, Device='cpu'):
    if Device is 'cpu':
        with tf.device('/cpu:0'):
            Dtype = tf.float32
            Variable = tf.get_variable(Name, Shape, Dtype, Init)
        return Variable
        
    elif Device is 'gpu':
        with tf.device('/gpu:0'):
            Dtype = tf.float32
            Variable = tf.get_variable(Name, Shape, Dtype, Init)
        return Variable

def GetVaribleWithRegularization(Shape, Init, Name, Regular='L2', RegularScale=None, Device='cpu'):
    if RegularScale is not None:
        if Regular=='L2':
            Regularizer = tfclayers.l2_regularizer(RegularScale)
        elif regular=='L1':
            Regularizer = tfclayers.l1_regularizer(RegularScaRegularizRegularizerle)
        if Device is 'cpu':
            with tf.device('/cpu:0'):
                Dtype = tf.float32
                Variable = tf.get_variable(Name, Shape, Dtype, Init, regularizer=Regularizer)
                return Variable
        elif Device is 'gpu':
            with tf.device('/gpu:0'):
                Dtype = tf.float32
                Variable = tf.get_variable(Name, Shape, Dtype, Init, regularizer=Regularizer)
                return Variable
    else:
        return GetVarible(Shape, Init, Name, Device)
