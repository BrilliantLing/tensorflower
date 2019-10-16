# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Deeps.Model import Model
from Deeps import Layers
import tensorflow as tf

class CCNNModel(Model):
    def __init__(self, Name, Reader):
        Model.__init__(self, Name, Reader)

    def __ReadAllBatch(self):
        return Reader.Read(self.Reader.DataIndexList, self.Reader.LogitsIndexList)
    
    def __ReadDataBatch(self):
        Raw, TwoValue, CRate, _ = Reader.Read(self.Reader.DataIndexList, self.Reader.LogitsIndexList)
        return [Raw, TwoValue, CRate]

    def __ReadLabelBatch(self):
        _, _, _, Logits = Reader.Read(self.Reader.DataIndexList, self.Reader.LogitsIndexList)
        return Logits

    def Inference(self, Input):
        Raw   = Input[0]
        Two   = Input[1]
        CRate = Input[2]

        # Raw branch
        RConv1 = Layers.CommonConvLayer(Raw, [3,21,1,16], Padding='VALID', Name='RConv1')
        RConv2 = Layers.CommonConvLayer(RConv1, [3,17,16,32], Padding='VALID', Name='RConv2')
        RPool3 = Layers.PoolingLayer(RConv2, [1,2,2,1], [1,2,2,1])
        RConv4 = Layers.CommonConvLayer(RPool3, [3,7,32,64], Padding='VALID', Name='RConv4')
        
        # TwoValues branch
        TConv1 = Layers.CommonConvLayer(Two, [3,21,1,16], Padding='VALID', Name='TConv1')
        TConv2 = Layers.CommonConvLayer(TConv1, [3,17,16,32], Padding='VALID', Name='TConv2')
        TPool3 = Layers.PoolingLayer(TConv2, [1,2,2,1], [1,2,2,1])
        TConv4 = Layers.CommonConvLayer(TPool3, [3,7,32,64], Padding='VALID', Name='TConv4')

        # Congestion rate branch
        CFC1 = Layers.FullyConnectedLayer(CRate, 35, 50, 0.001, Name='CFC1')
        CFC2 = Layers.FullyConnectedLayer(CFC1, 50, 50, 0.001, Name='CFC2')

        # Merge CNN
        CNNMerge         = tf.concat([RConv4, TConv4], 3)
        CNNMergeChannels = CNNMerge.get_shape()[3].value
        CNNMerge5        = Layers.CommonConvLayer(CNNMerge, [3,3,CNNMergeChannels,128], Padding='VALID', Name='MConv5')
        CNNMergePool6    = Layers.PoolingLayer(CNNMerge5, [1,2,6,1], [1,2,6,1])
        CNNMerge7        = Layers.CommonConvLayer(CNNMergePool6, [3,3,128,128], Padding='VALID', Name='MConv7')
        CNNMergeReshape  = tf.reshape(CNNMerge7, [self.Reader.BatchSize,-1])
        
        # Merger all
        Merge         = tf.concat([CNNMergeReshape, CFC2], 1)
        MergeChannels = Merge.get_shape()[1].value
        Merge8        = Layers.FullyConnectedLayer(Merge, MergeChannels, 300, 0.001, Name='MFC8')
        Merge9        = Layers.FullyConnectedLayer(Merge8, 300, 3, 0.001, Name='MFC9')
        return Merge9

    def LogitsAndLabels(self):
        Raw, Two, CRate, Labels = self.Reader.Read()
        Logits = self.Inference([Raw, Two, CRate])
        Labels = tf.cast(Labels, tf.int64)
        return Logits, Labels

    def InferenceLoss(self):
        Raw, Two, CRate, Labels = self.Reader.Read()
        Logits = self.Inference([Raw, Two, CRate])
        Labels = tf.cast(Labels, tf.int64)
        CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Logits, labels=Labels)
        CrossEntropyMean = tf.reduce_mean(CrossEntropy)
        return CrossEntropyMean



        