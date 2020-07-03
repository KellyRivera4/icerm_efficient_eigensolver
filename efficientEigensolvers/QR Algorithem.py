#!/usr/bin/env python
# coding: utf-8

# # QR Algorithem

# This code is a draft implementation of a PageRank Algorithm variant using QR Decomposition
# Implementing Householder and Gram-Schmidt Process

# ## Import Libraries

# In[44]:


#Import Python Libraries
import numpy as npy
import scipy as sp
import scipy.linalg as spl  # SciPy Linear Algebra Library


# ## Householder transformation method

# In[45]:


# QR Factorization 
# using numpy command
def qr_householderN(x):
    Q, R = npy.linalg.qr(x)
    return Q, R 


# In[46]:


# QR Factorization 
# using scipy command
def qr_householderS(x):
    Q, R = spl.qr(x)
    return Q, R


# In[47]:


# RQ Decomposition of a matrix
# Calculate decomposition of a = qr where q is unitary/orthogonal and r upper triangular.
def rq_householder(x):
    R, Q = spl.rq(x)
    return R, Q


# ## Gram-Schmidt Process

# In[48]:


# QR Factorization using the Gram-Schmidt precess
def qr_GS(x):
    m, n = x.shape
    Q = npy.zeros((m, n))
    R = npy.zeros((n, n))

    for j in range(n):
        v = x[:, j]

        for i in range(j - 1):
            q = Q[:, i]
            R[i, j] = q.dot(v)
            v = v - R[i, j] * q

        vNorm = npy.linalg.norm(v)
        Q[:, j] = v / vNorm
        R[j, j] = vNorm
    return Q, R


# ## Testing the functions

# In[49]:


# testing qr factorization Gram-schmidt function
A = npy.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
x, y  = qr_GS(A)
print(x,"\n", y )


# In[50]:


# testing qr factorization Householder function (using numpy.linalg)
A = npy.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
x, y  = qr_householderN2(A)
print(x,"\n", y )


# In[51]:


# testing qr factorization Householder function (using scipy.linalg command)
A = npy.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
x, y  = qr_householderS(A)
print(x,"\n", y)


# In[52]:


# testing rq decomposition Householder function 
A = npy.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
x, y = rq_householder(A)
print(x,"\n", y)

