#!/usr/bin/python

A=[3,1,2,4,3]
l1=[]
l2=[]
sum_lstA=[]
sum_lstB=[]
s_lst = []
def solution(A):
	i=0
	while i < len(A):
		sideA = A[:i+1]
		#print sideA
		sideB = A[i+1:]
		#print sideB
		l1.append(sideA)
		l2.append(sideB)
		#sumA=sum(l1)
		#sumB=sum(l2)
		#abs_diff=abs(sumA-sumB)
		#answer = min(abs_diff)
		i+=1
	#print l1
	#print l2
	for x in l1:
		sumA = sum(x)
		sum_lstA.append(sumA)
		#print sumA

	for y in l2:
		sumB = sum(y)
		sum_lstB.append(sumB)
		#print sumB
	#print sum_lstA
	#print sum_lstB

	w = 0
	while w < len(sum_lstA):
		sum_w = abs(sum_lstA[w] - sum_lstB[w])
		s_lst.append(sum_w)
		w+=1
	s_lst.pop(-1)
	#print s_lst
	#print min(s_lst)
	answer = min(s_lst)
	#print answer
	return answer

solution(A)
