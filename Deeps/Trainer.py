## -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
import numpy as np
from six.moves import xrange

class Trainer(object):
    def __init__(self):
        pass
    
    def Train(self):
        pass

class CommonTrainer(Trainer):
    def __init__(self, Model, LR, LRMutable, Config):
        Trainer.__init__(self)
        self.Model     = Model
        self.Epochs    = Config.Epochs
        self.BasicLR   = LR
        self.LR        = LR
        self.LRMutable = LRMutable
        self.BatchNum  = Config.BatchNum
        self.DataPath  = Config.DataPath
        self.ArgsPath  = Config.ArgsPath

    def __GetTrainStep(self, Loss, GlobalStep, OptMethod=['GD']):
        if self.LRMutable[0] is True:
            self.LR = tf.train.exponential_decay(self.BasicLR, GlobalStep, self.LRMutable[1], self.LRMutable[2])
        if OptMethod[0] == 'GD':
            TrainStep = tf.train.GradientDescentOptimizer(self.LR).minimize(Loss)
        else:
            TrainStep = None
        return TrainStep

    def __InitModel(self):
        pass

    def Train(self):
        GlobalStep = tf.Variable(0, trainable=False)
        TotalLoss = self.Model.InferenceLoss() + tf.losses.get_regularization_loss()
        TrainStep = self.__GetTrainStep(TotalLoss, GlobalStep)
        Saver = tf.train.Saver(tf.global_variables())

        #init all variables
        Init  = tf.global_variables_initializer()
        Coord = tf.train.Coordinator()
        Sess  = tf.Session()
        Sess.run(init)

        tf.train.start_queue_runners(sess=Sess,coord=Coord)
        
        LossList = []
        StartTime = time.time*()

        for steps in xrange(self.Epochs * self.BatchNum):
            _, Loss = Sess.run([TrainStep, TotalLoss])
            LossList.append(Loss)
            if (steps+1) % self.BatchNum == 0:
                EndTime   = time.time()
                Duration  = EndTime - StartTime
                FormatStr = ("Epoch %d:\t\tLoss:%.4f\tTime:%dsec/epoch")
                print(FormatStr % ((steps+1) / self.BatchNum, np.mean(LossList), Duration))
                LossList.clear()
                StartTime = EndTime
            if (steps+1) % (self.BatchNum*20) == 0:
                Saver.save(Sess, self.ArgsPath, global_step=steps)
                StartTime = time.time()


                    