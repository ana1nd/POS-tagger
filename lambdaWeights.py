# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 18:35:28 2016

@author: anand
"""

from __future__ import division

def learnWeights():
    
    unigram = dict()
    bigram = dict()
    trigram = dict()
  
    i = 0
    mode = 1000
    with open("Brown_tagged_train.txt","r") as f:
        for line in f:
            
            ls_line = line.split(" ")
            
            if ls_line[len(ls_line)-1] == '\r\n' :
                ls_line = ls_line[:-1]
            
            first_wordTag = ['Start/*']  
            ls_line = first_wordTag + ls_line
            ls_line = first_wordTag + ls_line
            
            j = 0            
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
                tag = mini_ls[1] 
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
    return unigram,bigram,trigram
            
def logic(unigram,bigram,trigram) :
    
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
            
unigram,bigram,trigram = learnWeights()
l1,l2,l3 = logic(unigram,bigram,trigram)
            
    