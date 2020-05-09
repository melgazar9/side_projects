#!/usr/bin/python

T=(5, (3, (20, None, None), (21, None, None)), (10, (1, None, None), None))
print T

class Tree(object):
	x = 0
	l = None
	r = None

	def solution(T):
		if T.l == None and T.r == None:
			# Has no subtree
			return 0
		elif T.l == None:
			# Only has right subtree
			return 1 + solution(T.r)
		elif T.r == None:
			# Only has left subtree
			return 1 + solution(T.l)
		else:
			# Have two subtrees
			return 1 + max(solution(T.l), solution(T.r))

	solution(T)
