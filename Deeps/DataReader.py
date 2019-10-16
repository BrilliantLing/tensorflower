# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class DataReader(object):
    def __init__(self, InputPath):
        self.InputPath = InputPath

    def Read(self):
        pass

class TFRecordReader(DataReader):
    def __init__(self, InputPath, DataIndexDict, LabelsIndexDict, DataIndexList, LabelsIndexList, DataShapeDict, LabelsShapeDict, BatchSize):
        DataReader.__init__(self, InputPath)
        self.DataIndexDict = DataIndexDict
        self.LabelsIndexDict = LabelsIndexDict
        self.DataIndexList = DataIndexList
        self.LabelsIndexList = LabelsIndexList
        self.DataShapeDict = DataShapeDict
        self.LabelsShapeDict = LabelsShapeDict
        self.BatchSize = BatchSize

    def __DecodeByType(self, Input, Shape, Type):
        if Type == 'string':
            Out = tf.decode_raw(Input, tf.float64)
            Out = tf.cast(Out, tf.float32)
            Out = tf.reshape(Out, Shape)
        elif Type == 'int64':
            Out = tf.cast(Input, tf.int32)
        elif Type == 'float32':
            Out = tf.cast(Input, tf.float32)
        return Out

    def __Dict2List(self, Dict, KeyList):
        ValList = []
        for Key in KeyList:
            ValList.append(Dict[Key])
        return ValList

    def __ReadItemToDict(self, IndexDict, SerializedExample, Shape):
        Dict = {}
        for Index, Type in IndexDict.items():
            print('\n\n\n',Type)
            Item = tf.io.parse_single_example(
                SerializedExample,
                features={
                    Index:tf.FixedLenFeature([], Type)
                }
            )
            Item = self.__DecodeByType(Item[Index], Shape[Index], Type)
            Dict[Index] = Item
        return Dict
            

    def Read(self, Shuffle=True):
        Queue = tf.train.string_input_producer([self.InputPath], Shuffle)
        TFReader = tf.TFRecordReader()
        _, SerializedExample = TFReader.read(Queue)
        DataItemDict = self.__ReadItemToDict(self.DataIndexDict, SerializedExample, self.DataShapeDict)
        LabelsItemDict = self.__ReadItemToDict(self.LabelsIndexDict, SerializedExample, self.LabelsShapeDict)
        DataItemList = self.__Dict2List(DataItemDict, self.DataIndexList)
        LabelsItemList = self.__Dict2List(LabelsItemDict, self.LabelsIndexList)
        if Shuffle is True:
            return tf.train.shuffle_batch(
                DataItemList+LabelsItemList,
                batch_size = self.BatchSize,
                num_threads=4,
                capacity = 10000,
                min_after_dequeue = 5000
            )
        else:
            return tf.train.batch(
                DataItemList+LabelsItemList,
                batch_size = self.BatchSize,
                num_threads=4,
                capacity = 10000,
                min_after_dequeue = 5000
            )

        