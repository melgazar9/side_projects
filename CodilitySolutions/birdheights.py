#!/usr/bin/python

from itertools import combinations
import itertools

H=[13,2,5]
def solution(H):
    l=[]
    i=0
    while i < len(H):
        combs = [x for x in combinations(H,i)]
        l.append(combs)
        i+=1
        
    lst = list(itertools.chain(*l))
    #print lst
    print 1000000007 % len(lst)
    return 1000000007 % len(set(lst))
    #return 1000000007 % (len(lst))
    
solution(H)
