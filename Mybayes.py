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
    
# remove numbers from vocabulary of list
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


def classifyNB(wordVector,P0v,plV,pSpam, th):
    pl = sum(wordVector * plV) + np.log(pSpam)
    P0 = sum(wordVector * P0v) + np.log(1.0 - pSpam)
    print 'p1: '+str(pl)
    print 'p0: '+str(P0)
    print 'p1 - p0: '+ str(pl - P0)+'\n'
    if pl - P0 > th:
        return 1
    else:
        return 0
    
    
def classifrisk(wordVector,P0v,plV,pSpam,th):#基于最小风险分类
    p1 = sum(wordVector * plV)
    p0 = sum(wordVector * P0v)
    a01 = 1;a10 = 2
    th = np.log(a01*pSpam)-np.log(a10*(1-pSpam))
    if p1/float(p0)>th:
        return 0;
    else:
        return 1;
    
    
def count_tfidf(classList,NonumList,docList,p0v,p1v):
    spamnum = sum(classList)
    hamnum = len(classList)- spamnum
    p0idf =[];p1idf = []
    for i in range(len(NonumList)):
        num = 0
        for j in range(len(classList)):
            if classList[j] == 1:#如果垃圾邮件
                if NonumList[i] in docList[j]:
                    num = num +1
        idf1 = np.log(spamnum/float(1+num))
        p1idf.append(idf1)
    p1idf = np.array(p1idf)
    for i in range(len(NonumList)):
        num = 0
        for j in range(len(classList)):
            if classList[j] ==0:#如果不是垃圾邮件
                if NonumList[i] in docList[j]:
                    num =num+1
        idf0 = np.log(hamnum/float(1 + num))
        p0idf.append(idf0)
    p0idf = np.array(p0idf)
    p1tfidf = p1v*p1idf
    p0tfidf = p0v*p0idf
    return p1idf,p0idf,p1tfidf,p0tfidf

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

def createFinalList2(p0idf,p1idf,NonumList,p0v,p1v):
    pp = [];finalList2 = []
    pp0v = p0idf.tolist()
    pp1v = p1idf.tolist()
    pidf = p1idf- p0idf
    pp = pidf.tolist()
    #maxv = max(pp)
    #minv = min(pp)
    #th = (maxv-minv)*0.8#(maxv - minv)
    fp0v2 =[];fp1v2 = []
    for i in range(len(pp)):
        if pp[i]<2.5 and pp[i]>-2.5:
            pp0v[i] = 0
            pp1v[i] = 0
        else:
            finalList2.append(NonumList[i])
            fp0v2.append(p0v[i])
            fp1v2.append(p1v[i])
            
    return finalList2,fp1v2,fp0v2


def countROC(tsetresult,testLabel):
    tp = 0;fp =0;tn =0;fn =0
    for i in range(len(testLabel)):
        if testLabel[i] == 1:
            if tsetresult[i] != testLabel[i]:#À¬»øÓÊ¼þ´í·ÖÀà
                fn = fn+1
            else:
                tp = tp+1
        if testLabel[i]== 0:
            if tsetresult[i] !=testLabel[i]:
                fp =fp+1
            else:
                tn = tn+1
    TPF = tp/float(tp+fn)
    TNF = tn/float(fp+tn)
    return TNF,TPF

def getAUC(rocx, rocy):
    sum = 0
    for i in range(len(rocx) - 1):
        sum += (rocy[i] + rocy[i+1]) * (rocx[i+1] - rocx[i]) / 2
    return sum          
            
            
            
    
    
    
    
    