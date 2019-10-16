# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Deeps import DataWriter
from Deeps import MatOps

import os
import tensorflow as tf
import CustomUtils as cusop


class CustomWriter(DataWriter.DataWriter):
    def __init__(self, BasicDir, RecordPath, Classes, MatName, Threshold):
        self.BasicDir   = BasicDir
        self.RecordPath = RecordPath
        self.Classes    = Classes
        self.MatName    = MatName
        self.Threshold  = Threshold

    def CreateRecord(self):
        if(os.path.exists(self.RecordPath)):
            print('The tfrecord file exists, it will be deleted!')
            os.remove(self.RecordPath)

        Writer = tf.python_io.TFRecordWriter(self.RecordPath)
        for Index, Name in enumerate(self.Classes):
            ClassDir  = self.BasicDir  + Name + '/'
            FileNames = os.listdir(ClassDir)
            print(FileNames)
            for FileName in FileNames:
                FilePath = os.path.join(ClassDir, FileName)
                BasicMat = MatOps.ReadMatrixFromMatFile(FilePath, self.MatName)
                TwoMat   = cusop.CongestionMat(BasicMat, self.Threshold)
                CRateVec = cusop.CongestionRateVec(BasicMat, self.Threshold)

                BasicMat = MatOps.MaxMinNormalize(BasicMat)
            
                BasicMatBytes = BasicMat.tostring()
                TwoMatBytes   = TwoMat.tostring()
                CRateVecBytes = CRateVec.tostring()

                Example = tf.train.Example(
                    features = tf.train.Features(
                        feature = {
                            'Raw'   : tf.train.Feature(bytes_list = tf.train.BytesList(value=[BasicMatBytes])),
                            'Two'   : tf.train.Feature(bytes_list = tf.train.BytesList(value=[TwoMatBytes])),
                            'CRate' : tf.train.Feature(bytes_list = tf.train.BytesList(value=[CRateVecBytes])),
                            'Label' : tf.train.Feature(int64_list = tf.train.Int64List(value=[Index]))
                        }
                    )
                )
                Writer.write(Example.SerializeToString())
                print(FileName, "\t done!")
        Writer.close()

