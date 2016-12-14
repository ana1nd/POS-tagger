# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:11:57 2016

@author: anand
"""

#here emission probability is calculated by sir

from __future__ import division
from itertools import chain
from math import *
import numpy as np
import pandas as pd


def computeAccuracy(myfile) :
    print "ComputeAccuracy Start"
    var_tag = []
    with open(myfile,"r") as f:
        
        for line in f :
            ls = []
            ls_line = line.split(" ")
            if ls_line[len(ls_line)-1]=='\r\n' or ls_line[len(ls_line)-1]=='\n' :
                ls_line = ls_line[:-1]
                
            for word in ls_line :
                
                count = word.count("/")
                groups = word.split("/")
                mini_ls = ['/'.join(groups[:count]), '/'.join(groups[count:])]
                w = mini_ls[0]
                t = mini_ls[1]                 
                ls.append(t)
            var_tag.append(ls)
    print "ComputeAccuracy End"
    return var_tag
    
def computeTransitional() :
    
    print "Transition Start"
    i = 0    
    transition = dict()
    with open("B2_GS.txt","r") as f :
        for line in f :
            '''if i>=20:
                break'''
            ls_line = line.split(" ")
            first_tag = ls_line[1]
            second_tag = ls_line[2]
            third_tag = ls_line[3]
            if first_tag == '*' :
                first_tag = '<s>'
            if second_tag == '*' :
                second_tag = '<s>'
            value = ls_line[4]
            #value = value[:len(value)-1]
            value = float(value)
            value = pow(2,value)
            #print first_tag,second_tag,third_tag,value
            if first_tag not in transition.keys() :
                transition[first_tag] = dict()
                transition[first_tag][second_tag] = dict()
                transition[first_tag][second_tag][third_tag] = value
            else :
                if second_tag not in transition[first_tag].keys():
                    transition[first_tag][second_tag] = dict()
                    transition[first_tag][second_tag][third_tag] = value
                else :
                    transition[first_tag][second_tag][third_tag] = value
                
            i+=1
    print "Transition End"
    return transition
            
def computeEmission() :
    print "Emission Start"
    i = 0
    emission = dict()
    with open("B4_GS.txt","r") as f:
        for line in f:
            ls_line = line.split(" ")
            word = ls_line[0]
            tag = ls_line[1]
            value = float(ls_line[2])
            if tag not in emission.keys():
                emission[tag] = dict()
                emission[tag][word] = pow(2,value)
            else :
                emission[tag][word] = pow(2,value)
            i+=1 
    emission['<s>'] = emission.pop('*')
    emission['<s>']['<s>'] = emission['<s>'].pop('*')
    print "Emission End"
    return emission
    
def checkAccuracy(myanswer,actualAnswer) :
    count = 0
    
    for i in range(0,len(myanswer),1):
        #print i,myanswer[i],actualAnswer[i]
        if myanswer[i] == actualAnswer[i]:
            count+=1
    accuracy = count/len(myanswer)
    return accuracy,myanswer


def viterbiAlgorithm(ls_line, tag_ls,transitional,emission) :
    
    global actual_tags 
    global predicted_tags
    T = len(ls_line) 
    
    viterbi = dict()
    tags = []
    main_tags = []
    
    for k in tag_ls :
        if k!='<s>':
            main_tags.append(k)
            viterbi[k] = []
        tags.append(k)
        
        
    start = '<s>'
    first_word = ls_line[0]
    saviour = '_RARE_'

    flag = 0    
    for s in main_tags :
        if first_word in emission[s].keys():
            flag = 1
            break
        
    for i in range(0,len(main_tags),1) :
        
        s = main_tags[i]
        a = transitional[start][start][s]
        if first_word in emission[s].keys() :
            b = emission[s][first_word]
        else :
            if flag == 1 :
                b = 0
            else :
                b = emission[s][saviour]
        #print s,a,b
        
        value = a*b
        #print tags[i],"  ",a,"  ",b,"   ",value
        viterbi[s].append(value)
          
    second_word = ls_line[1]
    flag = 0    
    for s in main_tags :
        if second_word in emission[s].keys():
            flag = 1
            break   
        
    for i in range(0,len(main_tags),1):
        s = main_tags[i]
        mx = -9999999999
        for j in range(0,len(main_tags),1):
            sprime = main_tags[j]
            previousValue = viterbi[sprime][0]
            if sprime in transitional[start].keys():
                if s in transitional[start][sprime].keys():
                    a = transitional[start][sprime][s]
                else:
                    a = 0
            else :
                a = 0
                
            if second_word in emission[s].keys():
                b = emission[s][second_word]
            else :
                if flag == 1:
                    b = 0 
                else :
                    b = emission[s][saviour]
                
            
            currentValue = previousValue * a * b
            mx = max(mx,currentValue)
            
        viterbi[s].append(mx)
    
    outer = len(ls_line)
    for t in range(2,outer,1):
        word = ls_line[t]
        
        flag = 0
        for i in main_tags :
            if word in emission[i].keys():
                flag = 1
                break
        
        for i in range(0,len(main_tags),1):
            s = main_tags[i]
            mx = -9999999999
            for j in range(0,len(main_tags),1):
                sprime = main_tags[j]
                for k in range(0,len(main_tags),1):
                    
                    sdoubleprime = main_tags[k]
                    previousValue = viterbi[sprime][t-1]
                    
                    if sprime in transitional[sdoubleprime].keys():
                        if s in transitional[sdoubleprime][sprime].keys():
                            a = transitional[sdoubleprime][sprime][s]
                        else :
                            a = 0
                    else :
                        a = 0
                    
                    if word in emission[s].keys():
                        b = emission[s][word]
                    else :
                        if flag == 0:
                            b = emission[s]['_RARE_']
                        else :
                            b = 0
                    
                    currentValue = previousValue * a * b
                    mx = max(currentValue,mx)
                    #if currentValue !=0 and sprime=='CONJ' and (s=='X' or s=='VERB'):
                     #   print word,s,sprime,sdoubleprime,previousValue,a,b,currentValue/previousValue
                    
            viterbi[s].append(mx)
            
    var_ls = []
    for s in main_tags :
        var_ls.append(viterbi[s])
    arr = np.array(var_ls)
    argmax = np.argmax(arr,axis=0)
    answer = []
    for i in range(0,len(argmax),1):
        answer.append(main_tags[argmax[i]])
    return answer
    
def main() :
    
    transitional = computeTransitional()
    emission = computeEmission()
    var_tag = computeAccuracy("Brown_tagged_dev.txt")
    tag_ls = transitional.keys()
    
    
    actual = []
    predicted = []

    mx = -1
    mn = 999999
    z = 0
    sum = 0
    with open("Brown_dev.txt","r") as f :
        for line in f :
            ls_line = line.split(" ")
            if ls_line[len(ls_line)-1] == '\r\n' or ls_line[len(ls_line)-1]=='\n' :
                ls_line = ls_line[:-1]
            if len(ls_line)>mx :
                mx = len(ls_line)
                str=ls_line
            myanswer = viterbiAlgorithm(ls_line,tag_ls,transitional,emission)
            acc , temp_predicted = checkAccuracy(myanswer,var_tag[z])
            actual.append(var_tag[z])
            predicted.append(temp_predicted)
            sum += acc            
            #print z,"  ",acc,"   ",len(var_tag[z])
            '''if z>=0:
                break'''
            z+=1
            if acc < mn :
                mn = acc
                line_num = z
                line_length = len(ls_line)
            if z%100==0 :
                print z,acc,sum/z
                
    
    print mn,line_num,line_length    
    actual = list(chain.from_iterable(actual))
    predicted = list(chain.from_iterable(predicted))
    actual = pd.Series(actual,name='Actual')
    predicted = pd.Series(predicted,name='Predicted')
    confusion_matrix = pd.crosstab(actual,predicted,margins=True)
    print confusion_matrix
    print type(confusion_matrix)
    pd.DataFrame.to_csv(confusion_matrix,"matrix_trigram.csv")
    return transitional,emission,actual,predicted
  
if __name__ == "__main__" :
    transitional,emission,actual,predicted = main()