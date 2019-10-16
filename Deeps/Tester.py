# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class Tester(object):
    def __init__(self):
        pass
    
    def Test(self):
        pass

class ClassficationTester(Tester):
    def __init__(self, Model, Config, TopK=1):
        self.Model      = Model
        self.BatchNum   = Config.BatchNum
        self.SampleNum  = Config.SampleNum
        self.DataPath   = Config.DataPath
        self.ArgsPath   = Config.ArgsPath
        self.TopK       = TopK

    def Test(self):      
        Logits, Labels = self.Model.LogitsAndLabels()
        Sess = tf.Session()
        
        TopKOp = tf.nn.in_top_k(Logits, Labels, self.TopK)

        Saver = tf.train.Saver()
        Ckpt = tf.train.get_checkpoint_state(ArgsPath)
        if Ckpt and Ckpt.model_checkpoint_path:
            Saver.restore(Sess,Ckpt.model_checkpoint_path)
            GlobalSteps = Ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return 
        tf.train.start_queue_runners(sess=sess)
        CorrectNums = 0
        for step < self.BatchNum:
            Predictions =  Sess.run([TopKOp])
            CorrectNums += np.sum(Predictions)
            step        += 1

        Precision = CorrectNums / self.SampleNum

        print('Precision:/t %.3f' %(Precision))
            
