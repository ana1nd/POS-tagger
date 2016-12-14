# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:57:17 2016

@author: anand
"""



from __future__ import division
from math import *
import numpy as np
import pandas as pd

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

def computeTransitional(bigramTag_dict,tag_dict) :
    
    transitional = dict()
    for k in bigramTag_dict :
        
        #print k,bigramTag_dict[k]
        ls = k.split("/")
        tag1 = ls[0]
        tag2 = ls[1]
        #print tag1,tag2
         
        if tag1 in transitional.keys():
            if tag2 in transitional[tag1].keys() :
                a=1
            else :
                transitional[tag1][tag2] = bigramTag_dict[k]/tag_dict[tag1]
        else :
           transitional[tag1] = dict()
           transitional[tag1][tag2] = bigramTag_dict[k]/tag_dict[tag1]

    return transitional

def computeEmission(wordTag_dict,tag_dict) :
    
    emission = dict()
    for k in wordTag_dict :         
        
        count3 = k.count("/")
        groups3 = k.split("/")
        mini_ls3 = ['/'.join(groups3[:count3]), '/'.join(groups3[count3:])]
        
        word = mini_ls3[0]
        tag = mini_ls3[1]        
        
        #print word,tag,type(word),type(tag)
        if tag in emission.keys() :
            if word in emission[tag].keys() :
                a=1
            else :
                emission[tag][word] = wordTag_dict[k]/tag_dict[tag]
        
        else :
            emission[tag] = dict()
            emission[tag][word] = wordTag_dict[k]/tag_dict[tag]

    return emission
    
    
def computeAccuracy() :
    
    var_tag = []
    with open("Brown_tagged_dev.txt","r") as f:
        
        for line in f :
            ls = []
            ls_line = line.split(" ")
            if ls_line[len(ls_line)-1]=='\r\n' :
                ls_line = ls_line[:-1]
                
            for word in ls_line :
                
                count = word.count("/")
                groups = word.split("/")
                mini_ls = ['/'.join(groups[:count]), '/'.join(groups[count:])]
                w = mini_ls[0]
                t = mini_ls[1]                 
                ls.append(t)
            var_tag.append(ls)
    return var_tag
                
    
actual_tags = []  
predicted_tags = []  
def viterbiAlgorithm(ls_line, tag_dict,transitional,emission,var_tag) :
    
    global actual_tags 
    global predicted_tags
    T = len(ls_line) 
    
    viterbi = dict()
    backPointers = dict()
    tags = []
    
    for k in tag_dict.keys() :
        if k!='<s>' : 
            tags.append(k)
            viterbi[k] = []
            backPointers[k] = []
            
    
    #initialization step   
    start = '<s>'
    first_word = ls_line[0]
    saviour = '_RARE_'
    
    for i in range(0,len(tags),1) :
        
        s = tags[i]
        a = log(transitional[start][s],2)
        if first_word in emission[s].keys() :
            b = log(emission[s][first_word],2)
        else :
            b = 0 
        
        value = a*b
        #print tags[i],"  ",a,"  ",b,"   ",value
        viterbi[s].append(value)
        backPointers[s].append(0)
    
    outer = len(ls_line)
    for t in range(1,outer,1) :
        word = ls_line[t]
    
        flag = 0
        for s in tags :
            if word in emission[s].keys() :
                flag=1
        
        for i in range(0,len(tags),1) : 
           
            s = tags[i]
            mx1 = -9999999999
            mx2 = -9999999999
            index = 0
            
            for j in range(0,len(tags),1) :
                sprime = tags[j]
                
                previousValue = viterbi[sprime][t-1]
                '''if viterbi[sprime][t-1]!=0 :
                    previousValue = log(viterbi[sprime][t-1],2)
                else :
                    previousValue = 0'''
                #print sprime,s
    
                if s in transitional[sprime].keys() :
                    #a = log(transitional[sprime][s],2)
                    a = transitional[sprime][s]
                else :
                    a = 0
                
                
                if word in emission[s].keys():
                    #b = log(emission[s][word],2)
                    b = emission[s][word]
                    #print word," ",s," ",sprime," ",previousValue*a*b
                else :
                    #print word,"   ",s,"  ",sprime,"  Absent"
                    if flag==0 :                    
                        #b = log(emission[s]['_RARE_'],10)
                        b = emission[s]['_RARE_']
                    else : 
                        b = 0
                
                #currentValue = previousValue * a * b
                currentValue = previousValue * a * b
                if currentValue >  mx1 :
                    mx1 = currentValue
                    #print tags[i],"  ",tags[j],"  ",mx1,"  ",a,"   ",b
                
                mx1 = max(mx1,currentValue)                
                
                if previousValue * a > mx2 :
                    mx2 = previousValue * a
                    index = j
                
            viterbi[s].append(mx1)
            backPointers[s].append(index)
                 
            
    arr = np.zeros((len(tags),len(ls_line)),np.double)
    for i in range(0,len(tags),1):
         tag = tags[i]
         v = viterbi[tag]
         for j in range(0,len(v),1):
             arr[i][j] = v[j]
             
    for j in range(0,len(ls_line),1) :
        mx= -1 
        index = 0
        for i in range(0,len(tags),1):
            if arr[i][j]>mx :
                mx = arr[i][j]
                index = i
        #print ls_line[j],tags[index],mx
    
    
    temp = np.argmax(arr,0)            
    
    
    count = 0
    total = len(ls_line)
    for i in range(0,len(ls_line),1) :
        actual_tags.append(var_tag[i])
        predicted_tags.append(tags[temp[i]])
        if tags[temp[i]] == var_tag[i] :
            count+=1
        #print i,tags[temp[i]],var_tag[i]
    return count/total,actual_tags,predicted_tags
    
def main() :
    
    wordCount_dict = findRareToken()
    
    i=0
    mode = 1000
    wordTag_dict = dict()
    tag_dict = dict()    
    bigramTag_dict = dict()
    
    with open("Brown_tagged_train.txt","r") as f :
        
        for line in f :
            
            '''if i>=1000:
                break'''
            i+=1
            
            ls_line = line.split(" ")
            
            if ls_line[len(ls_line)-1] == '\r\n' :
                ls_line = ls_line[:-1]
            
            first_wordTag = ['Start/<s>']            
            ls_line = first_wordTag + ls_line
            
            
            j = 0            
            for j in range(0,len(ls_line),1) :
                
                count = ls_line[j].count("/")
                groups = ls_line[j].split("/")
                mini_ls = ['/'.join(groups[:count]), '/'.join(groups[count:])]
                temp_w = mini_ls[0]
                temp_t = mini_ls[1] 
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
                    
                
                count = word_tag.count("/")
                groups = word_tag.split("/")
                mini_ls = ['/'.join(groups[:count]), '/'.join(groups[count:])]
                tag = mini_ls[1]
                
                if tag in tag_dict.keys() : 
                    tag_dict[tag] +=1
                else :
                    tag_dict[tag] = 1
                
                if j < len(ls_line)-1 :  #not the last word/tag 
                    
                    current_tag = tag
                    count2 = ls_line[j+1].count("/")
                    groups2 = ls_line[j+1].split("/")
                    mini_ls2 = ['/'.join(groups2[:count2]), '/'.join(groups2[count2:])]
                    next_tag =  mini_ls2[1]
                    common = current_tag + "/" + next_tag
                    
                    if common in bigramTag_dict.keys() :
                        bigramTag_dict[common] += 1
                    else :
                        bigramTag_dict[common] = 1
                    
                j +=1
                
            if i%mode == 0:
                print i
     
            #print ls_line
     
     
    transitional = computeTransitional(bigramTag_dict,tag_dict) 
    emission = computeEmission(wordTag_dict,tag_dict)
    
    var_tag = computeAccuracy() 
    actual_tags = []
    predicted_tags = []
    mx = -1
    z = 0
    sum = 0
    with open("Brown_dev.txt","r") as f :
        for line in f :
            ls_line = line.split(" ")
            if ls_line[len(ls_line)-1] == '\r\n' :
                ls_line = ls_line[:-1]
            if len(ls_line)>mx :
                mx = len(ls_line)
                str=ls_line
            acc,actual,predicted = viterbiAlgorithm(ls_line,tag_dict,transitional,emission,var_tag[z])
            sum += acc            
            '''print z,"  ",acc,"   ",len(var_tag[z])
            print
            print'''
            '''if z>=500:
                break'''
            z+=1
            
            if z%100==0 :
                print z,acc,sum/z
            
    actual = pd.Series(actual,name='Actual')
    predicted = pd.Series(predicted,name='Predicted')
    confusion_matrix = pd.crosstab(actual,predicted,margins=True)
    print confusion_matrix
    print type(confusion_matrix)
    pd.DataFrame.to_csv(confusion_matrix,"matrix.csv")
    return wordTag_dict,bigramTag_dict,tag_dict,wordCount_dict,transitional,emission,actual,predicted
            
    


if __name__ == "__main__" :
    wordTag_dict,bigramTag_dict,tag_dict,wordCount_dict,transitional,emission,actual,predicted = main()