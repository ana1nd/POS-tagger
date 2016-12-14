
from __future__ import division
from itertools import chain
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
                if wordCount_dict[temp_w]<=4 :
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
            if i%mode == 0 :
                print i
            i+=1
    return wordTag_dict,unigram,bigram,trigram

def computeAccuracy(myfile) :
    
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
    return var_tag
    
def computeTransitional(unigram_freq,bigram_freq,trigram_freq) :
    
    transitional_bigram = dict()
    for k in bigram_freq :
        
        ls = k.split("/")
        tag1 = ls[0]
        tag2 = ls[1]
         
        if tag1 in transitional_bigram.keys():
            if tag2 not in transitional_bigram[tag1].keys() :
                transitional_bigram[tag1][tag2] = bigram_freq[k]/unigram_freq[tag1]
        else :
           transitional_bigram[tag1] = dict()
           transitional_bigram[tag1][tag2] = bigram_freq[k]/unigram_freq[tag1]

    transitional_trigram = dict()
    for k in trigram_freq :
        ls = k.split("/")
        tag1 = ls[0]
        tag2 = ls[1]
        tag3 = ls[2]
        common = tag1 + "/" + tag2
        
        if tag1 not in transitional_trigram.keys():
            transitional_trigram[tag1] = dict()
            transitional_trigram[tag1][tag2] = dict()
            transitional_trigram[tag1][tag2][tag3] = trigram_freq[k]/bigram_freq[common]
        else:
            if tag2 not in transitional_trigram[tag1].keys():
                transitional_trigram[tag1][tag2] = dict()
                transitional_trigram[tag1][tag2][tag3] = trigram_freq[k]/bigram_freq[common]
            else :
                transitional_trigram[tag1][tag2][tag3] = trigram_freq[k]/bigram_freq[common]
        
    N = 0
    for item in unigram_freq.keys():
        N += unigram_freq[item]
    
    transitional_unigram = dict()
    for k in unigram_freq :
        transitional_unigram[k] = unigram_freq[k]/N
        
    return transitional_unigram,transitional_bigram,transitional_trigram
            
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
    
def checkAccuracy(myanswer,actualAnswer) :
    count = 0
    
    for i in range(0,len(myanswer),1):
        #print i,myanswer[i],actualAnswer[i]
        if myanswer[i] == actualAnswer[i]:
            count+=1
    accuracy = count/len(myanswer)
    return accuracy,myanswer
    
    
def learnWeights(unigram,bigram,trigram) :
    
    N = 0
    for item in unigram :
        N += unigram[item]
        
    lambda1 = 0
    lambda2 = 0
    lambda3 = 0
    
    for item in trigram:
        
        groups = item.split("/")
        first_tag = groups[0]
        second_tag = groups[1]
        third_tag = groups[2]
        
        a_Nr = first_tag + "/" + second_tag + "/" + third_tag
        a_Dr = first_tag + "/" + second_tag
        
        b_Nr = second_tag + "/" + third_tag
        b_Dr = second_tag
        
        c_Nr = third_tag
        c_Dr = N
        
        if a_Dr in bigram.keys() and bigram[a_Dr] != 1 :
            a = (trigram[a_Nr] - 1) / (bigram[a_Dr] - 1)
        else :
            a = 0
        
        if unigram[b_Dr] != 1:       
            b = (bigram[b_Nr] - 1) / (unigram[b_Dr] - 1)
        else :
            b = 0
        
        c = (unigram[c_Nr] - 1) / (N - 1)
        
        if max(a,max(b,c)) == a :
            lambda1 += trigram[a_Nr]
        elif max(a,max(b,c)) == b :
            lambda2 += trigram[a_Nr]
        else :
            lambda3 += trigram[a_Nr]
        
    
    sum = lambda1 + lambda2 + lambda3
    lambda1 = lambda1/sum
    lambda2 = lambda2/sum
    lambda3 = lambda3/sum
    #print lambda1,lambda2,lambda3    
    return lambda1,lambda2,lambda3

def viterbiAlgorithm(ls_line, tag_ls,transitional_unigram,transitional_bigram,transitional_trigram,emission,l1,l2,l3) :
    
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
        
        a_trigram = transitional_trigram[start][start][s]
        a_bigram = transitional_bigram[start][s]
        a_unigram = transitional_unigram[s]
        a = l1*a_trigram + l2*a_bigram + l3*a_unigram
        
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
            
            a_trigram = 0
            a_bigram = 0
            a_unigram = 0
            
            if sprime in transitional_trigram[start].keys():
                if s in transitional_trigram[start][sprime].keys():
                    a_trigram = transitional_trigram[start][sprime][s]
                    
            if s in transitional_bigram[sprime].keys():
                a_bigram = transitional_bigram[sprime][s]
                
            if s in transitional_unigram.keys():
                a_unigram = transitional_unigram[s]
                
            a = l1*a_trigram + l2*a_bigram + l3*a_unigram
            
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
                    
                    a_trigram = 0
                    a_bigram = 0
                    a_unigram = 0
                    
                    if sprime in transitional_trigram[sdoubleprime].keys():
                        if s in transitional_trigram[sdoubleprime][sprime].keys():
                            a_trigram = transitional_trigram[sdoubleprime][sprime][s]
                    
                    if s in transitional_bigram[sprime].keys():
                        a_bigram = transitional_bigram[sprime][s]
                    
                    if s in transitional_unigram.keys():
                        a_unigram = transitional_unigram[s]
                        
                    a = l1*a_trigram + l2*a_bigram + l3*a_unigram
                    
                    if word in emission[s].keys():
                        b = emission[s][word]
                    else :
                        if flag == 0:
                            b = emission[s]['_RARE_']
                        else :
                            b = 0
                    
                    currentValue = previousValue * a * b
                    mx = max(currentValue,mx)
                    #print word,s,sprime,sdoubleprime,previousValue,a,b,currentValue
                    
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
    
    wordCount_dict = findRareToken()
    wordTag_dict,unigram_freq,bigram_freq,trigram_freq = calculateVariables(wordCount_dict)  
    transitional_unigram,transitional_bigram,transitional_trigram = computeTransitional(unigram_freq,bigram_freq,trigram_freq)
    l1,l2,l3 = learnWeights(unigram_freq,bigram_freq,trigram_freq)    
    emission = computeEmission(wordTag_dict,unigram_freq)
    var_tag = computeAccuracy("Brown_tagged_dev.txt")
    tag_ls = unigram_freq.keys()
    actual = []
    predicted = []

    mx = -1
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
            myanswer = viterbiAlgorithm(ls_line,tag_ls,transitional_unigram,transitional_bigram,transitional_trigram,emission,l1,l2,l3)
            acc , temp_predicted = checkAccuracy(myanswer,var_tag[z])
            actual.append(var_tag[z])
            predicted.append(temp_predicted)
            sum += acc            
            #print z,"  ",acc,"   ",len(var_tag[z])
            '''if z>=0:
                break'''
            z+=1
            
            if z%100==0 :
                print z,acc,sum/z
                
    actual = list(chain.from_iterable(actual))
    predicted = list(chain.from_iterable(predicted))
    actual = pd.Series(actual,name='Actual')
    predicted = pd.Series(predicted,name='Predicted')
    confusion_matrix = pd.crosstab(actual,predicted,margins=True)
    print confusion_matrix
    print type(confusion_matrix)
    pd.DataFrame.to_csv(confusion_matrix,"anand.csv")
    return transitional_unigram,transitional_bigram,transitional_trigram,emission,actual,predicted

    
    


if __name__ == "__main__" :
    transitional_unigram,transitional_bigram,transitional_trigram,emission,actual,predicted = main()