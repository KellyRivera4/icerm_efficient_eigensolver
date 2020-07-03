# # QR Algorithem

# This code is a draft implementation of a PageRank Algorithm variant using QR Decomposition
# Implementing Householder and Gram-Schmidt Process

# ## Import Libraries

#Import Python Libraries
import numpy as npy
import scipy as sp
import scipy.linalg as spl  # SciPy Linear Algebra Library


# ## Householder transformation method

# QR Factorization 
# using numpy command
def qr_householderN(x):
    Q, R = npy.linalg.qr(x)
    return Q, R 


# QR Factorization 
# using scipy command
def qr_householderS(x):
    Q, R = spl.qr(x)
    return Q, R

# RQ Decomposition of a matrix
# Calculate decomposition of a = qr where q is unitary/orthogonal and r upper triangular.
def rq_householder(x):
    R, Q = spl.rq(x)
    return R, Q


# ## Gram-Schmidt Process

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
