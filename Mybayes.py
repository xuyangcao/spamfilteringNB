# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:17:16 2017

@author: Administrator
"""
import re
import numpy as np
    
def textParse(emailtext):#处理邮件内容分词并删除
    listOfTikens = re.split('\\W*',emailtext)
    return [tok.lower() for tok in listOfTikens if len(tok) > 2]


def createVocabList(docList):
    VocabSet = set([])
    for document in docList:
        VocabSet = VocabSet|set(document)
    return list(VocabSet)
def Nonum(VocabList):
    NonumList = []
    for i in range(len(VocabList)):
        if  VocabList[i].isdigit():
            VocabList[i] = VocabList[i]
        else:
            NonumList.append(VocabList[i])
    VocabSet = set([])
    aaa = []
    for i in range(len(NonumList)):
        aaa.append(filter(str.isalpha,NonumList[i]))
    for document in aaa:
        VocabSet = VocabSet|set(NonumList)      
    return list(VocabSet)
    
def setOfWords2Vec(NonumList,inputSet):
    returnVec = [0]*len(NonumList)
    for word in inputSet:
        if word in NonumList:
            returnVec[NonumList.index(word)] = 1
    return returnVec

def bagOfWords2Vec(NonumList,inputSet):
    returnVec = [0]*len(NonumList)
    for word in inputSet:
        if word in NonumList:
            returnVec[NonumList.index(word)] += 1
    return returnVec


def trainNB(trainMatrix,trainClass):
    numTrainDocs = len(trainMatrix)#训练集字典中的字数
    numWords = len(trainMatrix[0])#一个训练集的编码长度
    pAbusive = sum(trainClass)/float(numTrainDocs)#垃圾邮件的先验概率P（1）
    p0Num = np.ones(numWords);plNum = np.ones(numWords)#初始化
    p0Denom = 2.0;plDenom = 2.0
    for i in range(numTrainDocs):#遍历训练集
        if trainClass[i] == 1:#如果是垃圾邮件
            plNum  += trainMatrix[i]#矩阵相加
            plDenom  +=  sum(trainMatrix[i])#分母
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    plVect = np.log(plNum/plDenom)#垃圾邮件的似然率垃圾邮件类中以字典为特征的似然函数
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,plVect,pAbusive


def classifyNB(wordVector,P0v,plV,pSpam):
    pl = sum(wordVector * plV) + np.log(pSpam)
    P0 = sum(wordVector * P0v) + np.log(1.0 - pSpam)
    if pl > P0:
        return 1
    else:
        return 0

def createFinalList(p0v,p1v,NonumList):
    pp = [];finalList = []
    pp0v = p0v.tolist()
    pp1v = p1v.tolist()
    p = p1v - p0v
    pp = p.tolist()
    maxv = max(pp)
    minv = min(pp)
    th = (maxv - minv)*0.8/2
    for i in range(len(pp)):
        if pp[i]<th and pp[i]>-th:
            pp0v[i] = 0
            pp1v[i] = 0
        else:
            finalList.append(NonumList[i])
    fp0v =[];fp1v = []
    for i in range(len(pp)):
        if pp0v[i] == p0v[i]:
            fp0v.append(pp0v[i])
        if pp1v[i] == p1v[i]:
            fp1v.append(pp1v[i])
    return finalList,fp1v,fp0v


    
            
            
            
    
    
    
    
    