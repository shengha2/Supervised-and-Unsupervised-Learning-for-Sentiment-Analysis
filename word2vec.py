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
# print ('Part 2')
# m= 1
# k=0.015
# print("Accuracy on the test set:", get_accuracy(posTestSet,negTestSet,m,k))
# print("Accuracy on the training set:", get_accuracy(posData,negData,m,k))

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
                if len(freq) == 50:
                    posMin = largest.index(min(largest))
                    notFull = False
            else:
                if f > largest[posMin]:
                    largest[posMin] = f
                    freq[posMin] = word
                    posMin = largest.index(min(largest))
    return freq

# negTen = part3(negDict,posDict,-100000000)
# posTen = part3(posDict,negDict,-100000000)
# print ("Part 3")
# print('Top ten words that suggest negative:')
# print(re.sub(r'[\[\]]','',str(negTen)))
# print('Top ten words that suggest positive:')
# print(re.sub(r'[\[\]]','',str(posTen)))

#posProb,negProb=build_probability(negDict, posDict)
#count=0
#sorted_pos = sorted(posProb.items(), key=operator.itemgetter(1))
#print(sorted_pos)

#for i in range(0,100):
    #word=sorted_pos[sorted_pos.keys()[i]]
    #if posDict[word]>20:
        #print (word)
        #count+=1
    #if count>10:
        #break
#print (posProb)
#print (judge_sentence(['bad'],posDict,negDict,0.0,0))

############################part7
def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def tanh_layer(y, W, b):
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output

def NLL(y, y_):
    return -sum(y_*log(y))

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T )


def singlelayer_forward(x,W):
    O=np.dot(W.T,x)
    output=softmax(O)
    return output

#def sum_neg_log_costfunc():

def deriv_singlelayer(W, x, y, y_):

    dCdO =  y - y_
    #import pdb; pdb.set_trace()
    dCdW =  dot(x, dCdO.T )
    #dCdb =  dCdO
    return dCdW


def train_the_network(W, x, y_):
    max_iter=10000
    esp=1e-8
    iteration=0
    pre=W-1
    learning_rate=1e-5
    Record=[]
    while (iteration<max_iter) and (norm(pre-W)>esp):
        y = singlelayer_forward(x,W)
        delta_W=deriv_singlelayer(W, x, y, y_)
        W=W-learning_rate*delta_W
        iteration+=1

        if iteration%200==0:
            #print(type(Record))
            Record.append(W)
            #print(Record)
    return W,Record
###################from assingment2

def get_embd(word,dic):
    try:
        return dic[word]
    except KeyError:
        return False
def getword_pair(word1,word2,embd):
    try:
        embd1= embd[word1]
        embd2= embd[word2]
        #print(embd1.shape,embd2.shape)
        return np.hstack((embd1,embd2))
    except:
        return(False)

def gen_data(Data,embd,size):
    X=np.zeros((1,256))
    count_right=0
    count_wrong=0
    for sentence in Data:
        for i in range(10,20):
            temp1=getword_pair(sentence[i],sentence[i+1],embd)
            temp2=getword_pair(sentence[i+1],sentence[i],embd)
            if type(temp1) != bool:
                X = np.vstack((X,temp1))
                count_right+=1
            if type(temp2) != bool:
                X = np.vstack((X,temp2))
                count_right+=1
    X = X[1:,:]
    temp=0
    while temp<count_right:
        word1=random.choice(embd.keys())
        word2=random.choice(embd.keys())
        try:
            if (negDict[word1]>10)or (negDict[word2]>10):
                continue
        except:
            pass
        temp1=getword_pair(word1,word2,embd)
        if type(temp1) != bool:

            X = np.vstack((X,temp1))
            temp+=1
    return(X,count_right)

print ("Part 7")
loaded = np.load("embeddings.npz")["emb"]
indice = np.load("embeddings.npz")["word2ind"].flatten()[0]
size=len(loaded)
embd = {}
for i,j in indice.items():
    embd[j] = loaded[i]

#print (getword_pair("hello","hi",embd))
#print (np.array(negData[0:3]).shape)
#print ('########################')
#print (negData[0:3][0:3].shape)
#Data=negData[0:3][0:3]+posData[0:3][0:3]
#print (Data)

X_train,count_train=gen_data(posData[0:500],embd,size)
Y_train=np.zeros((2*count_train,2))
for i in range(0,count_train):
    Y_train[i,0],Y_train[i+count_train,1]=1,1

#print (X_train.shape)
W=np.zeros((256,2))+np.random.rand(256,2)/100
resultW,record=train_the_network(W, X_train.T, Y_train.T)
X_test,count=gen_data(posData[500:520],embd,size)
print('Trainning done! Accuracy on the test set:')
acc=0
for i in range(0,2*count):
    if i<count:
        temp=np.dot(X_test[i,:],resultW)
        if temp[0]>temp[1]:
            acc+=1
    if i>=count:
        temp=np.dot(X_test[i,:],resultW)
        if temp[0]<temp[1]:
            acc+=1
print (float(acc)/float(count*2))

y_test=[]
y_train=[]
for j in range(len(record)):
    acc=0
    for i in range(0,2*count):
        if i<count:
            temp=np.dot(X_test[i,:],record[j])
            if temp[0]>temp[1]:
                acc+=1
        if i>=count:
            temp=np.dot(X_test[i,:],record[j])
            if temp[0]<temp[1]:
                acc+=1
    y_temp=float(acc)/float(count*2)
    y_test.append(y_temp)

for j in range(len(record)):
    acc=0
    for i in range(0,2*count_train):
        if i<count_train:
            temp=np.dot(X_train[i,:],record[j])
            if temp[0]>temp[1]:
                acc+=1
        if i>=count_train:
            temp=np.dot(X_train[i,:],record[j])
            if temp[0]<temp[1]:
                acc+=1
    y_temp=float(acc)/float(count_train*2)
    y_train.append(y_temp)

x=range(100,10000,200)
plt.plot(x, y_train, '-',label="Training set")
plt.plot(x, y_test, '-',label="Test set")
plt.axis([0, 10000, 0, 1.1])
plt.legend()
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.figure(1)
plt.show()
#######################part8
print ("Part 8")
def cosine_theta(v1, v2):
    dot_product = np.dot(v1,v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v1)
    cosine = dot_product/(mag_v1*mag_v2)
    return cosine

distance={}
target='good'
for word in embd:
    distance[word]=1-cosine_theta(embd[target],embd[word])
#print (distance)
distance = sorted(distance.items(), key=operator.itemgetter(1))
#print (distance)
print("The top ten closest word to good are:")
for i in range(1,11):
    print(distance[i][0])
#print ()

distance={}
target='story'
for word in embd:
    distance[word]=1-cosine_theta(embd[target],embd[word])
#print (distance)
distance = sorted(distance.items(), key=operator.itemgetter(1))
print("The top ten closest word to story are:")
for i in range(1,11):
    print(distance[i][0])
target='i'
for word in embd:
    distance[word]=1-cosine_theta(embd[target],embd[word])
#print (distance)
distance = sorted(distance.items(), key=operator.itemgetter(1))
print("The top ten closest word to i are:")
for i in range(1,11):
    print(distance[i][0])
target='man'
for word in embd:
    distance[word]=1-cosine_theta(embd[target],embd[word])
#print (distance)
distance = sorted(distance.items(), key=operator.itemgetter(1))
print("The top ten closest word to man are:")
for i in range(1,11):
    print(distance[i][0])
