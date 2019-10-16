# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import scipy.io   as sio
import numpy      as np
from sklearn import preprocessing as pp

def ReadMatrixFromMatFile(Path, MatName):
    Mat = sio.loadmat(Path)
    Mat = Mat[MatName]
    return Mat

def WriterMatrixToNatFile(Mat, MatName, Path):
    sio.savemat(Path, {MatName:Mat})

def MaxMinNormalize(Mat):
    Max = np.max(Mat)
    Min = np.min(Mat)
    Normal = Mat.copy()
    Normal = (Normal - Min) / (Max - Min)
    return Normal

def ZScoreNormalize(Mat):
    Mean   = np.mean(Mat)
    Std    = np.std(Mat)
    Normal = Mat.copy()
    Normal = (Normal - Mean) / Std
    return Normal


    