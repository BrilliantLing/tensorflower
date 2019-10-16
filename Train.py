# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from Deeps import Configer as conf
from Deeps import Trainer
from Deeps import DataReader
from CongestionCNN import CCNNModel

Epochs    = 300
BatchSize = 2
SampleNum = 1000
DataPath  = '../TFRecord/Train.tfrecords'
ArgsPath  = '../Args/Congestion.ckpt'

DataIndexDict = {
    'Raw'   : 'string',
    'Two'   : 'string',
    'CRate' : 'string'
}
LabelsIndexDict = {
    'Level' : 'int64'
}

DataIndexList   = ['Raw','Two','CRate']
LabelsIndexList = ['Level']

DataShapeDict  = {
    'Raw'   : [35, 108, 1],
    'Two'   : [35, 108, 1],
    'CRate' : [35]
}
LabelShapeDict = {
    'Level' : [1]
} 


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
CCNNTrainer = Trainer.CommonTrainer(CCNN, 0.1, [True, 0.5, 50], Config)

if __name__ == '__main__':
    CCNNTrainer.Train()