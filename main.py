# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 09:23:58 2017

@author: Administrator
"""
import os
import Mybayes as my
import pylab as pl
#import sys  
#reload(sys)  
#sys.setdefaultencoding('utf8')   
import numpy as np

errorrate = []

s = 'spm'
pth = 'C:\Users\Administrator\Desktop\grade\pattern\spam\mydata\lingspam_public_s\lingspam_public_s\lemm\part' 
#全部数据的读取
print pth

for i in range(10):
    filename = [];docList = [];fullText = [];classList = [];testLabel = []; testdocList = [];trainMat = [];tsetresult = []
    TP = []; TN = [];FP = [];FN = [];rocx = [];rocy = []
    print i
    for j in range(10):
        if j != i:
            jj = str(j+1)
            pthtrain = pth + jj
            trainfiles = os.listdir(pthtrain)
            for num in range(len(trainfiles)):
                label = s in trainfiles[num]
                if label == 1:
                    classList.append(1)
                    wordList = my.textParse(open(pthtrain+'\\'+trainfiles[num]).read())
                    docList.append(wordList)
                else:
                    classList.append(0)
                    wordList = my.textParse(open(pthtrain+'\\'+trainfiles[num]).read())
                    docList.append(wordList)
        if j == i:
            ii = str(i+1)
            pthtest = pth + ii
            testfiles = os.listdir(pthtest)#tenfiles包含part1--10
            for tnum in range(len(testfiles)):
                r = s in testfiles[tnum]
                testLabel.append(r)
                testdoc = my.textParse(open(pthtest+'\\'+testfiles[tnum]).read())
                testdocList.append(testdoc)
             
    VocabList = my.createVocabList(docList)#不重复词列表spam和ham
    NonumList = my.Nonum(VocabList)
    for trainnum in range(len(docList)):
        trainMat.append(my.setOfWords2Vec(NonumList,docList[trainnum]))#利用字典对训练集编码
    #开始训练
    p0v,p1v,pspam = my.trainNB(np.array(trainMat),np.array(classList))#计算先验概率
    print pspam
    errorcount = 0#训练完成设置错去率为0
    #在测试前对List进行化简根据likehood之差进行
    #FinalList , fp1v , fp0v = my.createFinalList(p0v,p1v,NonumList)
    #file1 = open('List_tf.txt','a+')
    #file1.write('第'+str(i)+'折提取到的词表:'+str(FinalList)+'\n');  
    #file1.close()

    p1idf,p0idf,p1tfidf,p0tfidf = my.count_tfidf(classList,NonumList,docList,p0v,p1v)
    FinalList2 , fp1v2 , fp0v2 = my.createFinalList2(p0idf,p1idf,NonumList,p0v,p1v)
    file2 = open('List_idf.txt','a+')
    file2.write('第'+str(i)+'折提取到的词表:'+str(FinalList2)+'\n');  
    file2.close()

    #开始测试
    #对测试邮件编码
    pp = []
    for i in range(10):#得到十个点
        th = 0#(-10) + 2*i 
        for testnum in range(len(testdocList)):  
            testMat = my.setOfWords2Vec(FinalList2,testdocList[testnum])
            p,re = my.classifyNB(np.array(testMat),fp0v2,fp1v2,pspam,th)
            pp.append(p)
            tsetresult.append(re)#一个th一个点一轮
            if re != testLabel[testnum] and th == 0:#只在th= 0时计算errorrate
                errorcount += 1
        TPF,TNF = my.countROC(tsetresult,testLabel)
        rocx.append(1-TNF)
        rocy.append(TPF)
    pl.title('ROC curve')
    pl.xlabel("1-TNF")
    pl.ylabel("TPF")
    pl.plot(rocx, rocy)# use pylab to plot x and y
    pl.show()# show the plot on the screen  
    print 'the error rate' + ii + 'is:',float(errorcount)/len(testdocList)#得到错误率
    errorrate.append(float(errorcount)/len(testdocList))
    file3 = open('errorrate.txt','a+')
    file3.write('第'+str(i)+'折错误率:'+ str(errorrate[i])+'\n');
    file3.close()
result = sum(errorrate)/10
file3.write('平均错误率：'+str(result)+'\n')
file3.close()
print 'the average error rate is:',result