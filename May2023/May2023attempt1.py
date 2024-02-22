# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:39:32 2023

@author: danie
"""
import numpy as np
import time

k = 5   
n = 2**30
   
def Q(i):
    return [int(2**k*(np.sin((i+1)*(t+1)) - np.floor(np.sin((i+1)*(t+1))))) for t in range(k)]

def x(i):
    return -1 + 2 * i / (n-1)

def a(t): 
    return t / (2**k - 1)

def A(i,j):
    # Generate length k Q vectors
    qi = Q(i)
    qj = Q(j)
    
    # compute the index of a
    index = 0
    for t in range(k):
        if qi[t] == qj[t]:
            index += 2**t
            
    return a(index)


# Recursively compute x.T A x 
def rcompute_qform(N):
    if N == 0:
        return 0.0
    
    if N == 1:
        return 1.0
    
    else:
        # Compute the dot product
        dot_prod = 0.0
        
        for i in range(N-1):
            dot_prod += A(i, N-1) * x(i)
        
        return x(N-1)**2 + 2*dot_prod*x(N-1) + rcompute_qform(N-1)


# Iteratively compute x.T A x
def icompute_qform(N):
    tot = 1
    T = 2

    while T <= N:
        # Compute the dot product
        dot_prod = 0.0
        
        for i in range(T-1):
            dot_prod += A(i, T-1) * x(i)
            
        tot += x(T-1)**2 + 2*dot_prod*x(T-1)    
        T += 1

    return tot 


# Test cases
# Amat = np.array([[A(i,j) for j in range(n)] for i in range(n)])
# xvec = np.linspace(-1.0, 1.0, num=n)

# start = time.time()
# print(f'n = {n}: recursive q-form = {rcompute_qform(n)}')
# end = time.time()
# print(f'Time: {end - start}')

# start = time.time()
# print(f'n = {n}: iterative q-form = {icompute_qform(n)}')
# end = time.time()
# print(f'Time: {end - start}')

def Pq(K):
    x = np.arcsin(-1 + (K + 1)/32) - np.arcsin(-1 + K/32)
    y = np.arcsin((K+1)/32) - np.arcsin(K/32)
    return (x+y)/np.pi

p = sum([Pq(i)**2 for i in range(32)]) # prob that two independent samples drawn from Q agree

def Pa(K):
    hw = bin(K).count("1") # hamming weight of K (in binary)
    return p**(hw)*(1-p)**(5 - hw)

Ea = sum([a(i) * Pa(i) for i in range(32)])

# iteratively compute x.T A x, using Ea instead of A[i, j] for all i != j
def ecompute_qform(N):
    tot = 1
    T = 2
    
    while T <= N:
        # estimate the dot product
        dot_prod = Ea*(1 - T + (T-1)*(T-2)/(n-1))
        
        tot += x(T-1)**2 + 2*dot_prod*x(T-1)    
        T += 1

    return tot 
    
    
# num = 0
# tot = 0

# for i in range(n):
#     for j in range(n):
#         if i!=j:
#             num += 1
#             tot += Amat[i, j]
    
# -------------------------------------------
#
# ATTEMPT 2: Diagonalizing A
#
# -------------------------------------------












def make_QQt_mat(n):
# Constructs the matrix Q @ Q.T: R^160 -> R^160
    shape = (5, 32, 5, 32)
    def entry(a,i,b,j):
        result_sum = 0.0
        for s in range(n):
            if j == f((s+1)*(b+1)) and i == f((s+1)*(a+1)):
                result_sum += np.sqrt(2**(2*k-i-j-2))
            else:
                continue
        return result_sum
    

    vectorized_entry = np.vectorize(entry, otypes=[float])
    QQt = vectorized_entry(*np.indices(shape))
    QQt = QQt.reshape((160,160))
    
    return QQt
    
def entry(a,i,b,j):
        result_sum = 0
        for s in range(n):
            if j == f((s+1)*(b+1)) and i == f((s+1)*(a+1)):
                result_sum += 1
            else:
                continue
        return result_sum   

#--------------------------
#
# More direct...
#
#-------------------------

def f(x):
    return int(32*(np.sin(x) - np.floor(np.sin(x))))

def compute_indices(n):
    indices = [[[] for l in range(32)] for a in range(5)]
    for a in range(5):
        for i in range(n):
            indices[a][f((i+1)*(a+1))].append(i)
    return indices
    
def compute_qform_clever(n):
    tot = 0.0
    indices = compute_indices(n)
    for a in range(5):
        for l in range(32):
            inner_sum = 0
            for i in indices[a][l]:
                inner_sum += x(i)
            tot += (inner_sum**2)*2**a
    return tot/31

def compute_inner_sums(n):
    inner_sums = [[0 for l in range(32)] for a in range(5)]
    for a in range(5):
        for i in range(n):
            inner_sums[a][f((i+1)*(a+1))] += x(i)
    return inner_sums

def compute_qform_clever2(n):
    tot = 0.0
    inner_sums = compute_inner_sums(n)
    for a in range(5):
        for l in range(32):
            tot += (inner_sums[a][l]**2)*2**a
    return tot/31
    
# def cmp():
#     diff = icompute_qform(n) - compute_qform_clever(n)
#     return diff

# start = time.time()
# print(f'n = {n}: iterative q-form = {icompute_qform(n)}')
# end = time.time()
# print(f'Time: {end - start}')

# start = time.time()
# print(f'n = {n}: clever q-form = {compute_qform_clever(n)}')
# end = time.time()
# print(f'Time: {end - start}')

start = time.time()
print(f'n = {n}: clever q-form2 = {compute_qform_clever2(n)}')
end = time.time()
print(f'Time: {end - start}')

