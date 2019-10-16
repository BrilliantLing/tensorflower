# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import Configer as conf
from Deeps import Tester
from Deeps import Tester
from Deeps import DataReader
from CongestionCNN import CCNNModel

Epochs    = 300
BatchSize = 2
SampleNum = 1000
DataPath  = '/TFRecord/Test.tfrecords'
ArgsPath  = ''

DataIndexDict = {
    'Raw'   : tf.string,
    'Two'   : tf.string,
    'CRate' : tf.float32
}
LabelsIndexDict = {
    'Level' : tf.int64
}

DataIndexList   = ['Raw','Two','CRate']
LabelsIndexList = ['Level']

DataShapeDict  = {
    'Raw'   : [35, 108, 1],
    'Two'   : [35, 108, 1],
    'CRate' : [1, 1]
}
LabelShapeDict = [1]


Reader    = DataReader.TFRecordReader(
                DataPath, 
                DataIndexDict, 
                LabelsIndexDict, 
                DataIndexList, 
                LabelsIndexList, 
                DataShapeDict, 
                LabelShapeDict, 
                BatchSize
            )

Config      = conf.CommonConfiger(Epochs, BatchSize, SampleNum, DataPath, ArgsPath)
CCNN        = CCNNModel('CCNN', Reader)
CCNNTester  = Tester.ClassficationTester(CCNN, Config, 1)

if __name__ == '__main__':
    CCNNTester.Test()