# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Model(object):
    def __init__(self, Name, Reader):
        self.Name = Name
        self.Reader = Reader
    
    def Inference(self, Input):
        pass

    def LogitsAndLabels(self):
        pass

    def InferenceLoss(self):
        pass

    def GetName(self):
        return self.Name