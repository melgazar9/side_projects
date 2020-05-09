#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:12:28 2017

@author: melgazar9
"""

# Find the longest word in the sentence

import re
import collections

S = 'Hey You! We test coders. Give us a try? Do you want to be my girlfriend?!'

def solution(S):
    
    l=[]
    A = re.split('[?.,!]', S)
    
    for sentence in A: 
        #print sentence
        #print len(sentence.split())
        num_words = len(sentence.split())
        l.append(num_words)
        
    print l
    print max(l)
    return max(l)
    
solution(S)