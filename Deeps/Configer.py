# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

#定义了一个人基类
class Configer(object):
    def __init__(self):
        pass

#继承于基类Configer，定义一个
class CommonConfiger(Configer):
    #构造函数
    def __init__(self, Epochs=None, BatchSize=None, SampleNum=None, DataPath=None, ArgsPath=None):
        Configer.__init__(self)
        self.__Epochs    = Epochs       #配置中的模型需要训练的次数
        self.__BatchSize = BatchSize    #配置中一批进入训练的模型的数据样本的数量
        self.__SampleNum = SampleNum    #样本集合的样本总数
        self.__DataPath  = DataPath     #样本（可用的数据源的）
        self.__ArgsPath  = ArgsPath

    #getter
    @property
    def Epochs(self):
        return self.__Epochs

    #setter
    @Epochs.setter
    def Epochs(self, Val):
        if isinstance(Val, int) and Val>0:
            self.__Epochs = Val

    #getter
    @property
    def BatchSize(self):
        return self.__BatchSize

    #setter
    @BatchSize.setter
    def BatchSize(self, Val):
        if isinstance(Val, int) and Val>0:
            self.__BatchSize = Val

    #getter
    @property
    def SampleNum(self):
        return self.__SampleNum

    #setter
    @SampleNum.setter
    def SampleNum(self, Val):
        if isinstance(Val, int) and Val>0:
            self.__SampleNum = Val

    #getter
    @property
    def DataPath(self):
        return self.__DataPath

    #setter
    @DataPath.setter
    def DataPath(self, Val):
        if isinstance(Val, str):
            self.__DataPath = Val

    #getter
    @property
    def ArgsPath(self):
        return self.__ArgsPath

    #setter
    @ArgsPath.setter
    def ArgsPath(self, Val):
        if isinstance(Val, str):
            self.__ArgsPath = Val

    #getter    
    @property
    def BatchNum(self):
        self.__BatchNum = int(math.ceil(self.SampleNum / self.BatchSize))
        return self.__BatchNum