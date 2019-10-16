# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import CustomWriter as cusw

TrainBasicDir   = '../Data/Train/'
TestBasicDir    = '../Data/Test/'
Classes         = ['0', '1', '2']
MatName         = 'speed'
Threshold       = 30
TrainRecordPath = '../TFRecord/Train.tfrecords'
TestRecordPath  = '../TFRecord/Test.tfrecords'

TrainWriter = cusw.CustomWriter(TrainBasicDir, TrainRecordPath, Classes, MatName, Threshold)
TestWriter  = cusw.CustomWriter(TestBasicDir, TestRecordPath, Classes, MatName, Threshold)

if __name__ == '__main__':
    TrainWriter.CreateRecord()
    TestWriter.CreateRecord()
