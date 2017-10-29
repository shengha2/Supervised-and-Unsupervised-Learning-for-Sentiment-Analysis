import os
import re
import random
import math
import numpy as np
import operator
from pylab import *
from numpy import random
from scipy import linalg
#import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
#import time
#import matplotlib.image as mpimg
#from scipy.ndimage import filters
#import urllib
#import cPickle
def build_dict(data):
    i = 0
    d = {}
    for review in data:
        uniqueWords = []
        for word in review:
            if word in uniqueWords:
                continue
            if word in d:
                d[word] += 1
            else:
                d[word]  = 1
            uniqueWords.append(word)
    return d

def read_dir(myDir,numTraintext,numTesttext,numValtext):
    database = []       #database -> [[all words in review 1],[all words in review 2],.....]
    testData = []       #testData -> [[all words in review 1],[all words in review 2],.....]
    valData = []        #valData -> [[all words in review 1],[all words in review 2],.....]
    filenames = os.listdir(myDir)
    random.shuffle(filenames)
    if numTraintext+numTesttext+numValtext > len(filenames):  #check if the input data sizes are too large
        print('Input data size too large, Try smaller sizes.')
        return
    for filename in filenames[:numTraintext]:
        database.append(re.sub(r'[^\w\s\d]',' ',open(myDir+filename).read().lower()).split())
    for filename in filenames[numTraintext:numTraintext+numTesttext]:
        testData.append(re.sub(r'[^\w\s\d]',' ',open(myDir+filename).read().lower()).split())
    for filename in filenames[numTraintext+numTesttext:numTraintext+numTesttext+numValtext]:
        valData.append(re.sub(r'[^\w\s\d]',' ',open(myDir+filename).read().lower()).split())
    return (database, testData, valData)




negDir = 'txt_sentoken/neg/'
posDir = 'txt_sentoken/pos/'

(negData,negTestSet,negValSet) = read_dir(negDir,600,200,200)
(posData,posTestSet,posValSet) = read_dir(posDir,600,200,200)

negDict = build_dict(negData)
posDict = build_dict(posData)
#print(posDict)
def build_probability(negDict,posDict):
    lenNeg=0
    lenPos=0
    for word in negDict:
        lenNeg+=negDict[word]
    for word in posDict:
        lenPos+=posDict[word]
    posProb={}
    negProb={}
    for word in negDict:
        negProb[word]=(float(negDict[word]))/(600)
    for word in posDict:
        posProb[word]=(float(posDict[word]))/(600)
    return (posProb, negProb)

def judge_sentence(sentence,posDict,negDict,m,k):
    lenNeg=0
    lenPos=0
    for word in negDict:
        lenNeg+=negDict[word]
    for word in posDict:
        lenPos+=posDict[word]
    probPos=0
    probNeg=0
    for word in sentence:
        #try:
            #if posDict[word]+negDict[word]<1:
                #continue
        #except:
            #continue
        if word in posDict:
            probPos+=math.log((float(posDict[word])+m*k)/(600.0+k)*posDict[word])
        else:
            probPos+=math.log(float(m*k)/(float(lenPos)+k))
        if word in negDict:
            probNeg+=math.log((float(negDict[word])+m*k)/(600.0+k)*negDict[word])
        else:
            probNeg+=math.log(float(m*k)/(float(lenNeg)+k))
        #print (negDict[word])
        #print (posDict[word])
    #print (probPos)
    #print (probNeg)
    if probPos>=probNeg:
        return (1)
    return(0)

def get_accuracy(posData,negData,m,k):
    count_right=0
    for sentence in posData:
        if judge_sentence(sentence,posDict,negDict,m,k):
            count_right+=1
    #print (float(count_right)/float(len(posData)))
    for sentence in negData:
        if judge_sentence(sentence,posDict,negDict,m,k)==0:
            count_right+=1
    accuracy = float(count_right)/float((len(negData)+len(posData)))
    return (accuracy)
#for k in [0.005,0.01,0.015,0.02,0.03,0.05,0.3]:
    #for m in [1,2,3,5,10,15,30]:
        #print('k',k,' m',m,' Accuracy',get_accuracy(posValSet,negValSet,m,k))
print ('Part 2')
m= 1
k=0.015
print("Accuracy on the test set:", get_accuracy(posTestSet,negTestSet,m,k))
print("Accuracy on the training set:", get_accuracy(posData,negData,m,k))

###########################part3
def part3(the_Dict,cpDict,cut):
    freq = []
    largest = []
    notFull = True
    for word in the_Dict:
        if the_Dict[word] > cut:
            if word in cpDict:
                f = log(float(the_Dict[word])/float(600))-log(float(cpDict[word])/float(600))
            else:
                f = 1
            if notFull:
                freq.append(word)
                largest.append(f)
                if len(freq) == 10:
                    posMin = largest.index(min(largest))
                    notFull = False
            else:
                if f > largest[posMin]:
                    largest[posMin] = f
                    freq[posMin] = word
                    posMin = largest.index(min(largest))
    return freq

negTen = part3(negDict,posDict,-100000000)
posTen = part3(posDict,negDict,-100000000)
print ("Part 3")
print('Top ten words that suggest negative:')
print(re.sub(r'[\[\]]','',str(negTen)))
print('Top ten words that suggest positive:')
print(re.sub(r'[\[\]]','',str(posTen)))
