#!/usr/bin/env python
# coding: utf-8

# In[3]:


import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import copy, deepcopy
from pprint import pprint
from functools import partial
import math
from matplotlib import pyplot as plt


# In[7]:


# Q2)i To check whether given matrix is Diagonally Dominant Matrix.
# Reading the size of linear system
def canbeDDM(m, n) :
 
    # For each row
    for i in range(0, n) :        
     
        # For each column, finding sum of each row.
        max = None
        maxColIndex = 0
        maxColIndexList = []
        sum = 0
        for j in range(0, n) :
            colValue = m[i][j]
            sum = sum + abs(colValue)

            #Check for Max
            if(j == 0):
                max = abs(colValue)
            elif(max < abs(colValue)):
                max = abs(colValue)
                maxColIndex = j

 
        # removing the max element.
        sum = sum - max        
 
        # Checking if max element is less than sum of all remaining element.
        if (max < sum) :
            return False
        
        # Check if max column already added. If yes than matrix can't be made diagonally dominant
        if maxColIndex in maxColIndexList :
            return False

        maxColIndexList.append(maxColIndex)
 
    return True


# In[4]:


a = np.float_(np.random.randint(1,10,size=(4,4)))
print("input matrix\n",a) 
b =(np.diag(a))
for i in range(0, 4):
  n = b[i]
  for j in range(0,4):
    factor = a[i, j] /  n
    a[i, j] =factor

U = np.triu(a,1)
L = np.tril(a,-1)
I = np.tril(np.triu(a))
D= I+L

def norm(n,m,d):
  f = 0
  for i in np.arange(n):
    for j in np.arange(m):
      f = f + np.sum(np.power(np.abs(d[i, j]), 2))
  print("\nFrobenius Norm", np.sqrt(f))

  colsums = []
  for i in np.arange(m):
      v = np.sum(np.abs(d[:, i]))
      colsums.append(v)

  print("1 -Norm", np.max(colsums))

  rowsums = []
  for i in np.arange(n):
      v = np.sum(np.absolute(d[i, :]))
      rowsums.append(v)

  print("∞ -Norm", np.max(rowsums))

def jacobi(U,L):
  d = -(L+U)
  print("Gause jacobi iteration matrix\n", d)
  n, m = d.shape
  norm(n,m,d)

def seidel(D,U):
  c = np.linalg.inv(D)
  d = np.dot(c,U)
  print("\nGause seidel iteration matrix\n", d)
  n, m = d.shape
  norm(n,m,d)

jacobi(U,L)
seidel(D,U)


# In[6]:


import numpy as np
import numpy.linalg as la
import time

def GaussSeidel(A,b):
       # dimension of the non-singular matrix
       n = len(A)

       # def. max iteration and criterions
       Kmax = 100;
       tol  = 1.0e-4;
       btol = la.norm(b)*tol


       x0   = np.zeros(n)
       k    = 0 ;
       stop = False
       x1   = np.empty(n)

       while not(stop) and k < Kmax:
           print ("begin while with k =", k)
           x1 = np.zeros(n)
           for i in range(n):          # rows of A
               x1[i] = ( b[i] - np.dot(A[i,0:i], x1[0:i]) - np.dot(A[i,i+1:n], x0[i+1:n]) ) / A[i,i]
               print("x1 =", x1)

           r    = b - np.dot(A,x1)
           stop = (la.norm(r) < btol) and (la.norm(x1-x0) < tol)
           print("end of for i \n")
           print("x0 =", x0)
           print("btol = %e \t; la.norm(r) = %e \t; tol = %e \t; la.norm(x1-x0) = %e; stop = %s " % (btol, la.norm(r), tol, la.norm(x1-x0), stop))
           x0   = x1
           print("x0 =", x0, end='\n')
           print("end of current while \n\n")
           k    = k + 1

       if not(stop):   # or if k >= Kmax
           print('Not converges in %d iterations' % Kmax)

       return x1, k
A = np.array( [
          [  3, -0.1, -0.2],
          [0.1,    7, -0.3],
          [0.3, -0.2,   10]
        ], dtype='f')

b = np.array( [7.85, -19.3, 71.4] )

xsol = la.solve(A,b)

start    = time.time()
x, k     = GaussSeidel(A,b)
ending   = time.time()
duration = ending-start
err      = la.norm(xsol-x)
print('Iter.=%d  duration=%f  err=%e' % (k,duration,err))


# In[ ]:




