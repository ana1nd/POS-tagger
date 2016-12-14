# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:51:48 2016

@author: anand
"""

from __future__ import division
from tempfile import TemporaryFile
import numpy as np
import pandas as pd

class Neural : 
    
    input_dim = 42
    output_dim = 14
    hidden_dim = 60  #number of hidden layers is 2
    eta = 0.8
    examples = 543149
    no_passes = 5000
    filemode = 25
    mode = 50

def findRareToken() :
    
    wordCount_dict = dict()
    
    i=0
    mode = 1000
    
    with open("Brown_train.txt","r") as f : 
        for line in f : 
            if i%mode==0:
                print i
            i+=1
            ls_line = line.split(" ")
            for word in ls_line :
                if word in wordCount_dict.keys() :
                    wordCount_dict[word]+=1
                else :
                    wordCount_dict[word] = 1
                    
    return wordCount_dict
    
def calculateVariables(wordCount_dict) :
    
    unigram = dict()
    bigram = dict()
    trigram = dict()
    wordTag_dict = dict()
  
    i = 0
    mode = 1000
    with open("Brown_tagged_train.txt","r") as f:
        for line in f:
            
            ls_line = line.split(" ")
            
            if ls_line[len(ls_line)-1] == '\r\n' :
                ls_line = ls_line[:-1]
            
            first_wordTag = ['Start/<s>']  
            ls_line = first_wordTag + ls_line
            ls_line = first_wordTag + ls_line
            
            for j in range(0,len(ls_line),1) :
                
                if j == 0:
                    count = ls_line[j].count("/")
                    groups = ls_line[j].split("/")
                    mini_ls = ['/'.join(groups[:count]), '/'.join(groups[count:])]
                    tag = mini_ls[1]
                    
                    count = ls_line[j+1].count("/")
                    groups = ls_line[j+1].split("/")
                    mini_ls = ['/'.join(groups[:count]), '/'.join(groups[count:])]
                    tag_next = mini_ls[1]
                    
                    common = tag + "/" + tag_next 
                    if common in bigram.keys():
                        bigram[common] += 1
                    else :
                        bigram[common] = 1
                        
                    count = ls_line[j+2].count("/")
                    groups = ls_line[j+2].split("/")
                    mini_ls = ['/'.join(groups[:count]), '/'.join(groups[count:])]
                    tag_next_next = mini_ls[1]
                    
                    common = tag + "/" + tag_next + "/" + tag_next_next
                    if common in trigram.keys():
                        trigram[common] += 1
                    else :
                        trigram[common] = 1
                    continue
                
                count = ls_line[j].count("/")
                groups = ls_line[j].split("/")
                mini_ls = ['/'.join(groups[:count]), '/'.join(groups[count:])]
                temp_w = mini_ls[0]
                temp_t = mini_ls[1] 
                tag = mini_ls[1]
                #print ls_line[j],temp_w,temp_t,type(temp_w),mini_ls
                if wordCount_dict[temp_w]<=1 :
                    temp_w = "_RARE_"
                    word_tag = temp_w + "/" + temp_t
                else :
                    word_tag = ls_line[j]
                    
                if word_tag in wordTag_dict.keys() :
                    wordTag_dict[word_tag] +=1
                else :
                    wordTag_dict[word_tag] = 1
                    
                
                #print tag
                if tag in unigram.keys() : 
                    unigram[tag] +=1
                else :
                    unigram[tag] = 1
                    
                if j < len(ls_line) - 2:
                    current_tag = tag
                    
                    count = ls_line[j+1].count("/")
                    groups = ls_line[j+1].split("/")
                    mini = ['/'.join(groups[:count]), '/'.join(groups[count:])]
                    next_tag = mini[1]
                    common = current_tag + "/" + next_tag
                    if common in bigram.keys():
                        bigram[common] += 1
                    else :
                        bigram[common] = 1
                                    
                    count2 = ls_line[j+2].count("/")
                    groups2 = ls_line[j+2].split("/")
                    mini2 = ['/'.join(groups2[:count2]), '/'.join(groups2[count2:])]
                    next_next_tag = mini2[1]
                    
                    common2 = current_tag + "/" + next_tag + "/" + next_next_tag
                    if common2 in trigram.keys():
                        trigram[common2] += 1
                    else :
                        trigram[common2] = 1
                  
                elif j < len(ls_line)-1 :  #not the last word/tag 
                    
                    current_tag = tag
                    count2 = ls_line[j+1].count("/")
                    groups2 = ls_line[j+1].split("/")
                    mini_ls2 = ['/'.join(groups2[:count2]), '/'.join(groups2[count2:])]
                    next_tag =  mini_ls2[1]
                    common = current_tag + "/" + next_tag
                    
                    if common in bigram.keys() :
                        bigram[common] += 1
                    else :
                        bigram[common] = 1
                                
            #break
            if i% mode == 0 :
                print i
            i+=1
    return wordTag_dict,unigram,bigram,trigram
    
def computeEmission(wordTag_dict,unigram_freq) :
    
    emission = dict()
    for k in wordTag_dict :         
        
        count = k.count("/")
        groups = k.split("/")
        mini_ls = ['/'.join(groups[:count]), '/'.join(groups[count:])]
        
        word = mini_ls[0]
        tag = mini_ls[1]        
        
        #print word,tag,type(word),type(tag)
        if tag in emission.keys() :
            if word not in emission[tag].keys() :
                emission[tag][word] = wordTag_dict[k]/unigram_freq[tag]
        else :
            emission[tag] = dict()
            emission[tag][word] = wordTag_dict[k]/unigram_freq[tag]

    return emission
    
def preprocess(emission) :
    
    emission['<e>'] = dict()
    emission['<e>']['End'] = 1.0
    tag_ls = emission.keys()
    z = 0
    with open("Brown_tagged_train.txt","r") as f :
        var_ls = []
        y_ls = []
        for line in f :
            ls_line = line.split(" ")
            
            if ls_line[len(ls_line)-1] == '\r\n' :
                ls_line = ls_line[:-1]
                
            ls_line = ['Start/<s>'] + ls_line + ['End/<e>']
            i = 0
            for i in range(1,len(ls_line)-1,1) :
                
                pcount = ls_line[i-1].count("/")
                pgroups = ls_line[i-1].split("/")
                pmini_ls = ['/'.join(pgroups[:pcount]), '/'.join(pgroups[pcount:])]
                pword = pmini_ls[0]
                ptag = pmini_ls[1]
                
                ccount = ls_line[i].count("/")
                cgroups = ls_line[i].split("/")
                cmini_ls = ['/'.join(cgroups[:ccount]), '/'.join(cgroups[ccount:])]
                cword = cmini_ls[0]
                ctag = cmini_ls[1]
                
                ncount = ls_line[i+1].count("/")
                ngroups = ls_line[i+1].split("/")
                nmini_ls = ['/'.join(ngroups[:ncount]), '/'.join(ngroups[ncount:])]
                nword = nmini_ls[0]
                ntag = nmini_ls[1]
                
                pProb = []
                cProb = []
                cTag = []
                nProb = []
                common_ls = []
                
                
                for t in tag_ls :
                    if pword in emission[t].keys():
                        pProb.append(emission[t][pword])
                    else:
                        pProb.append(0)
                        
                    if cword in emission[t].keys():
                        cProb.append(emission[t][cword])
                    else:
                        cProb.append(0)
                        
                    if t==ctag :
                        cTag.append(1.0)
                    else :
                        cTag.append(0.0)
                    
                    if nword in emission[t].keys():
                        nProb.append(emission[t][nword])
                    else:
                        nProb.append(0)
                    
                    
                common_ls = pProb + cProb + nProb
                var_ls.append(common_ls)
                y_ls.append(cTag)
            if z%1000 == 0:
                print z
            z+=1
    
    x = np.array(var_ls)
    y = np.array(y_ls)
    return x,y
    
def testPreprocess(emission):
    emission['<e>'] = dict()
    emission['<e>']['End'] = 1.0
    tag_ls = emission.keys()
    z = 0
    with open("Brown_tagged_dev.txt","r") as f:
        var_ls = []
        y_ls = []
        for line in f:
            ls_line = line.split(" ")
            
            if ls_line[len(ls_line)-1] == '\r\n' :
                ls_line = ls_line[:-1]
                
            ls_line = ['Start'] + ls_line + ['End']
            i = 0
            for i in range(1,len(ls_line)-1,1) :
                
                pcount = ls_line[i-1].count("/")
                pgroups = ls_line[i-1].split("/")
                pmini_ls = ['/'.join(pgroups[:pcount]), '/'.join(pgroups[pcount:])]
                pword = pmini_ls[0]
                ptag = pmini_ls[1]
                
                ccount = ls_line[i].count("/")
                cgroups = ls_line[i].split("/")
                cmini_ls = ['/'.join(cgroups[:ccount]), '/'.join(cgroups[ccount:])]
                cword = cmini_ls[0]
                ctag = cmini_ls[1]
                
                ncount = ls_line[i+1].count("/")
                ngroups = ls_line[i+1].split("/")
                nmini_ls = ['/'.join(ngroups[:ncount]), '/'.join(ngroups[ncount:])]
                nword = nmini_ls[0]
                ntag = nmini_ls[1]
                
                saviour = '_RARE_'
                
                pProb = []
                cProb = []
                nProb = []
                cTag = []
                common_ls = []
                
                pflag = 0
                cflag = 0
                nflag = 0
                
                for t in tag_ls :
                    if pword in emission[t].keys():
                        pflag = 1
                    if nword in emission[t].keys():
                        nflag = 1
                    if cword in emission[t].keys():
                        cflag = 1
                
                for t in tag_ls :
                    if pword in emission[t].keys():
                        pProb.append(emission[t][pword])
                    else:
                        if pflag == 1 or saviour not in emission[t].keys():
                            pProb.append(0.0)
                        else :
                            pProb.append(emission[t][saviour])
                        
                    if cword in emission[t].keys():
                        cProb.append(emission[t][cword])
                    else:
                        if cflag == 1 or saviour not in emission[t].keys():
                            cProb.append(0.0)
                        else :
                            cProb.append(emission[t][saviour])
                    
                    if nword in emission[t].keys():
                        nProb.append(emission[t][nword])
                    else:
                        if nflag == 1 or saviour not in emission[t].keys():
                            nProb.append(0.0)
                        else :
                            nProb.append(emission[t][saviour])
                            
                    if t==ctag :
                        cTag.append(1.0)
                    else :
                        cTag.append(0.0)
                        
                            
                common_ls = pProb + cProb + nProb
                var_ls.append(common_ls)
                y_ls.append(cTag)
            if z%1000 == 0:
                print z
            z+=1
            
    
    x_test = np.array(var_ls)
    y_test_answer = np.array(y_ls)
    return x_test,y_test_answer
            
    
def give_weights() :
    
    np.random.seed(0)
    W1 = 2*np.random.random((Neural.input_dim,Neural.hidden_dim)) -1
    W2 = 2*np.random.random((Neural.hidden_dim,Neural.hidden_dim)) -1
    W3 = 2*np.random.random((Neural.hidden_dim,Neural.output_dim)) -1   
    return W1,W2,W3
    
def build_model(X,y_bits,W1,W2,W3,no_passes,x_test,y_actual) : 
    
    error_ls = []
    x_pass = []
    
    for i in range(1,no_passes+1,1) :
        
        loss = 0
        for j in range(0,len(X),1) :
            
            j_X = X[j]
            temp = j_X.reshape((1,42))  #1x42
            
            j_y = y_bits[j]
            temp_y = j_y.reshape((1,14))  #1x14
            
            sum1 = temp.dot(W1)
            sigma1 = 1/(1+np.exp(-sum1))  #1Xhidden_dim 1X40
            
            sum2 = sigma1.dot(W2)
            sigma2 = 1/(1+np.exp(-sum2))  #1xhidden_dim 1X40
            
            sum3 = sigma2.dot(W3)
            sigma3 = 1/(1+np.exp(-sum3))  #1x14
            
            delta3 = (sigma3 - temp_y) * (sigma3*(1-sigma3))  #1x14
            delta2 = delta3.dot(W3.T) *  (sigma2*(1-sigma2))  #1xhidden_dim 1X40
            delta1 = delta2.dot(W2.T) *  (sigma1*(1-sigma1))  #1xhidden_dim 1X40
            
            W3 -= (Neural.eta * sigma2.T.dot(delta3))  #hidden_dimx14 
            W2 -= (Neural.eta * sigma1.T.dot(delta2))  #hiddenxhidden
            W1 -= (Neural.eta * temp.T.dot(delta1))    #42xhidden
            
            loss+= np.sum(np.square(sigma3 - temp_y))
        
        
        loss = loss/len(X)        
        error_ls.append(loss)
        x_pass.append(i)
        print i,"   ",loss    
        if i%Neural.mode ==0 :        
            trainingAccuracy(i,loss,X,y,W1,W2,W3)
        
        if i% Neural.filemode == 0:
            makeCSVfile(x_test,y_actual,W1,W2,W3)
         
    return W1,W2,W3
    
def trainingAccuracy(i,loss,X,y,W1,W2,W3) :
    
    sum1 = X.dot(W1)
    sigma1 = 1/(1+np.exp(-sum1))
    
    sum2 = sigma1.dot(W2)
    sigma2 = 1/(1+np.exp(-sum2))
    
    sum3 = sigma2.dot(W3)
    sigma3 = 1/(1+np.exp(-sum3))
    
    result = np.argmax(sigma3,axis=1) 
    y = np.argmax(y,axis=1)
    correct = np.sum(result==y)
    total = len(X)
    print "acc=",i,"   ",loss,"   ",np.sum(result==y),"   ",correct/total
    if i>0 :
        return 
    result = pd.DataFrame(result)
    print "Training Acc :"
    countFrequency(result)
    
def makeCSVfile(x_test,y_actual,W1,W2,W3) :
    
    output_class = predict(x_test,W1,W2,W3)    
    result = np.argmax(output_class,axis=1) #predicted
    y_actual = np.argmax(y_actual,axis=1)
    total = len(x_test)
    correct = np.sum(result==y_actual)
    acc = correct/total
    print "test Acc = ",correct,"   ",acc
    '''output_class = pd.DataFrame(output_class)    
    result = pd.DataFrame(result)
    result.to_csv("output.csv",header=True)
    return result'''
    
def predict(X_test,W1,W2,W3) :
       
     sum1 = X_test.dot(W1)
     sigma1 = 1/(1+np.exp(-sum1))
     
     sum2 = sigma1.dot(W2)
     sigma2 = 1/(1+np.exp(-sum2))
     
     sum3 = sigma2.dot(W3)
     sigma3 = 1/(1+np.exp(-sum3))
     
     return sigma3


def main():
    wordCount_dict = findRareToken()
    wordTag_dict,unigram_freq,bigram_freq,trigram_freq = calculateVariables(wordCount_dict)  
    emission = computeEmission(wordTag_dict,unigram_freq)
    x,y = preprocess(emission)
    W1,W2,W3 = give_weights()
    x_test,y_actual = testPreprocess(emission)
    #W1,W2,W3 = build_model(x,y,W1,W2,W3,Neural.no_passes,X_bits,X_test_bits)
    W1,W2,W3 = build_model(x,y,W1,W2,W3,Neural.no_passes,x_test,y_actual)
 
if __name__ == "__main__" :
X,y,W1,W2,W3,result = main()