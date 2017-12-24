#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import permutations as pmu

all= list(range(1,21))
print(all)

a = [2, 5, 16, 9, 7]
A = sum(a)

print(a,"sum=",A)

#c‚è•ª
rest = set(all)-set(a)
print(rest)

b =[]
c =[]

for x,y in pmu(rest, 2):
	if( (x+y*2)==A):
		#print(x,y)
		b=[x,y]
		rest2=set(rest)-set(b)
		for xx, yy, zz in pmu(rest2, 3):
			if((xx*3+yy*2)== zz*3):
				c = [xx, yy, zz]
				rest3=set(rest2)-set(c)
				for q in (rest3):
					ql=[q]
					rest4=set(rest3)-set(ql)
					sum_rest4=sum(rest4)
					if((sum_rest4*3+q)==(2*A+2*sum(b)+4*sum(c))):
						for s, t in pmu(rest4,2):
							S=[s,t]
							rest5=set(rest4)-set(S)
							if((s*2+t)==sum(rest5)):
								print(x, y, xx, yy, zz, q)

