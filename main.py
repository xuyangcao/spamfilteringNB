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
import pylab as pl

errorrate = []
auc = []

s = 'spm'
pth = 'C:/Users/new_xuyangcao/Desktop/spamfilteringNB/lingspam_public_s/lemm/part' 
#全部数据的读取
print pth

for i in range(10):
    filename = [];docList = [];fullText = [];classList = [];testLabel = []; testdocList = [];trainMat = []
    rocx = [0];rocy = [0]
    temp_auc = 0
    
    print 'the ' + str(i) + 'th fold...' 
    print '    read data from folder...'
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
                
    print '    creat vocabulary list...'    
    start = time.time()        
    VocabList = my.createVocabList(docList)#不重复词列表spam和ham
    NonumList = my.Nonum(VocabList)
    print '    creat vocabulary list ' + str(time.time() - start) +'s'
    
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
#    FinalList , fp1v , fp0v = my.createFinalList(p0v,p1v,NonumList)
#    file1 = open('List_tf.txt','a+')
#    file1.write('第'+str(i)+'折提取到的词表:'+str(FinalList)+'\n');  
#    file1.close()
    
    
#    print 'Final list over!'
#    test using idf
    p1idf, p0idf, p1tfidf, p0tfidf = my.count_tfidf(classList,NonumList,docList,p0v,p1v)
    idf_wordList, fp1v2,fp0v2 = my.createFinalList2(p0idf,p1idf,NonumList,p0v,p1v)
    file_idf = open('idf_feature_words.txt','a+')
    file_idf.write('第'+str(i)+'折提取到的词表:'+str(idf_wordList)+'\n');  
    file_idf.close()
    
#    开始测试
#    对测试邮件编码
    strat = time.time()
    for th in range(-100,110, 10):
        testresult = []      
        for testnum in range(len(testdocList)):
            testMat = my.setOfWords2Vec(idf_wordList,testdocList[testnum])
            re = my.classifyNB(np.array(testMat),fp0v2,fp1v2,pspam, th)
            testresult.append(re)
            if re != testLabel[testnum] and th == 0:
                errorcount += 1
        TPF,TNF = my.countROC(testresult,testLabel)
        rocx.append(1-TNF)
        rocy.append(TPF)     
    rocx.append(1)
    rocy.append(1)
    temp_auc = my.getAUC(rocx, rocy)
    
    pl.title('ROC curve of original feature vector')
    pl.xlabel("FPR")
    pl.ylabel("TPR")
    pl.plot(rocx, rocy, '*-')# use pylab to plot x and y
    pl.show()# show the plot on the screen    
    pl.savefig('%d.png' % i)
    print '    the error rate: ' + ii + ' is:',str(float(errorcount)/len(testdocList))#得到错误率
    print '    the AUC: ', temp_auc
    
    #save roc datas
    file_roc = open('roc_curve_data.txt', 'a+')
    file_roc.write(ii + 'fold roc curve data:'+str(rocx)+'\n\n'+str(rocy)+'\n\n\n')
    file_roc.close
    
    auc.append(temp_auc)
    errorrate.append(float(errorcount)/len(testdocList))
    print '    testing time: ' +str(time.time()-strat) + 's'
result = sum(errorrate)/10
auc_average = sum(auc) / len(auc)
print 'the average error rate is:',result
print 'the average auc is :', auc_average

#
#
#






















