# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def GetTFMse(Real, Pred):
    Mse = tf.losses.mean_squared_error(Pred, Read)
    return Mse

def GetTFMre(Real, Pred):
    Mre = tf.reduce_mean(tf.div(tf.abs(tf.subtract(Real, Pred), Real)))
    return Mre

def GetTFMae(Real, Pred):
    Mae = tf.reduce_mean(tf.abs(tf.subtract(Real, Pred)))
    return Mae

def GetNPMse(Real, Pred):
    Diff = Real - Pred
    Mse = np.mean(np.power(Diff, 2))
    return Mse

def GetNPMre(Real, Pred):
    AbsDiff = np.abs(Real - Pred)
    Mre = np.mean(AbsDiff / Real)
    return Mre

def GetNPMae(Real, Pred):
    AbsDiff = np.abs(Real-Pred)
    Mae = np.mean(AbsDiff)
    return Mae