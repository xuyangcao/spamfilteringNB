# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 09:23:58 2017

@author: Administrator
"""
import os
import Mybayes as my
#import sys  
#reload(sys)  
#sys.setdefaultencoding('utf8')   
import numpy as np
import time

errorrate = []

s = 'spm'
pth = 'C:/Users/new_xuyangcao/Desktop/spamfilteringNB/lingspam_public_s/lemm/part' 
#全部数据的读取
print pth

for i in range(1):
    filename = [];docList = [];fullText = [];classList = [];testLabel = []; testdocList = [];trainMat = []
    
    print 'the ' + str(i) + 'th fold...' 
    print '    set up train Mat...'
    start = time.time()
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
            testfiles = os.listdir(pthtest)
            for tnum in range(len(testfiles)):
                r = s in testfiles[tnum]
                testLabel.append(r)
                testdoc = my.textParse(open(pthtest+'\\'+testfiles[tnum]).read())
                testdocList.append(testdoc)
  
    VocabList = my.createVocabList(docList)#不重复词列表spam和ham
    NonumList = my.Nonum(VocabList)
    for trainnum in range(len(docList)):
        print '        the '+str(trainnum)+'th doc in '+str(len(docList)) +' docs'
        trainMat.append(my.setOfWords2Vec(NonumList,docList[trainnum]))#利用字典对训练集编码
    end = time.time()
    print '    set up trainMat: ' + str(end - start) + 's'
                       
    #开始训练
    print '    training...'
    start = time.time()
    p0v,p1v,pspam = my.trainNB(np.array(trainMat),np.array(classList))#计算先验概率
    print '    training time: ' + str(time.time()-start) + 's'
                              
    print '    prior: ' + str(pspam)
    print '    testing...'
    strat = time.time()
    errorcount = 0#训练完成设置错去率为0
    #在测试前对List进行化简根据likehood之差进行
    FinalList , fp1v , fp0v = my.createFinalList(p0v,p1v,NonumList)
    #开始测试
    #对测试邮件编码
    for testnum in range(len(testdocList)):
        testMat = my.setOfWords2Vec(NonumList,testdocList[testnum])

        re = my.classifyNB(np.array(testMat),p0v,p1v,pspam)
        if re != testLabel[testnum]:
            errorcount += 1
    print '    the error rate: ' + ii + 'is:',str(float(errorcount)/len(testdocList))#得到错误率

    errorrate.append(float(errorcount)/len(testdocList))
    print '    testing time: ' +str(time.time()-strat) + 's'
result = sum(errorrate)/10
print 'the average error rate is:',result
