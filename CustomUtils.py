# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def CongestionRate(Mat, Threshold):
    Count = Mat[Mat>Threshold].size
    Rate = Count / Mat.size
    return Rate

def CongestionRateVec(Mat, Threshold):
    Vec = np.array([])
    for Row in Mat:
        Count = Row[Row>Threshold].size
        Rate  = Count / Row.size
        np.append(Vec, Rate)
    return Vec

def CongestionMat(Mat, Threshold):
    CMat = Mat.copy()
    CMat[CMat<=Threshold] = 0
    CMat[CMat>Threshold]  = 1
    return CMat