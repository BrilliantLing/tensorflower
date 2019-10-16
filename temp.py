# -*- coding: utf-8 -*-
import csv
import os
import shutil

MatDir = '../../Mat/Done/'
C0Dir  = '../../class/0/'
C1Dir  = '../../class/1/'
C2Dir  = '../../class/2/'

FileNameList = os.listdir(MatDir)

CsvFile = open('../../Labels.csv', 'r',encoding='UTF-8')
Reader  = csv.reader(CsvFile)
for item in Reader:
    
    if Reader.line_num == 1:
        item[0] = item[0][1]
    Index = Reader.line_num-1
    FileName = FileNameList[Index]
    MatPath = os.path.join(MatDir, FileName)
    if int(item[0]) <= 3:
        C0FilePath = os.path.join(C0Dir, FileName)
        shutil.copyfile(MatPath, C0FilePath)
    elif int(item[0]) <=6:
        C1FilePath = os.path.join(C1Dir, FileName)
        shutil.copyfile(MatPath, C1FilePath)
    else:
        C2FilePath = os.path.join(C2Dir, FileName)
        shutil.copyfile(MatPath, C2FilePath)