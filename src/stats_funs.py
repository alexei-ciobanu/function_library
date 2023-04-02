import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as intgr
import scipy.stats as stats
import itertools
import functools

import sys
import os

from . import general_funs as gf

def covar_to_corr(M):
    M_c = np.copy(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M_c[j, i] = M[j, i]/(np.sqrt(M[j, j] * M[i, i]))        
    return M_c

@functools.wraps(np.histogram)
def histogram(*args, **kwargs):
    '''identical to numpy histogram but instead return the 
    center of the bins instead the edges so you can plot the bloody histogram'''
    f,bins = np.histogram(*args, **kwargs)
    x = (bins[1:] + bins[:-1])/2
    return x,f
    
def kde(X, x, h=None):
    X_sort = np.sort(X)
    X_sort_grad = np.diff(X_sort)
    X_sort_grad_med = np.median(X_sort_grad)
    
    # blame Silverman not me
    magic_constant = 0.382
    
    N = 1/(X_sort_grad_med*magic_constant)
    
    if h is None:
        h = N**(-1/5)
    elif str(h).lower() == 'silverman':
        h = 1.06*len(X)**(-1/5)
    
    phi = lambda x: gf.gaussian(x,0,1)
    kernel = lambda x,h: np.mean(phi((x-X)/h)/h)
    kpdf = np.vectorize(lambda x,h: kernel(x,h))
    f4 = kpdf(x, h)
    
    dx = x[1] - x[0]
    
    f4 = f4/(np.sum(f4)*dx)
    
    return f4
    
def char_fun(X,ts):
    return np.mean(np.exp(-1j*np.multiply.outer(ts, X)), axis=1)
    
def char_to_pdf(X, ts, xs, w=lambda x: np.ones(len(x))):
    char = char_fun(X, ts)
    dt = ts[1] - ts[0]
    
    pdf = 1/(2*np.pi) * np.sum( w(ts)*char*np.exp(1j*np.multiply.outer(xs, ts)), axis=1 ) * dt
    return pdf
    
def char_to_pdf_ifft(X, ts, w= lambda x: np.ones(len(x))):
    char = char_fun(X,ts)
    dt = ts[1] - ts[0]
    
    freq = np.fft.fftshift(scipy.fftpack.fftfreq(len(ts), dt)*2*np.pi)
    df = freq[1] - freq[0]
    
    pdf = np.fft.fftshift(np.abs(np.fft.ifft(w(ts)*char))) / df
    return freq, pdf
    
def phi(d, niter=30, scipy_root_find=False):
    '''
    Returns the solution to the polynomial equation x**d = x + 1.
    Where the solution to the d = 1 case returns the golden ratio.
    This can be thought as a generalization of the golden ratio to the d'th order.
    
    Examples:
    phi(1) = 1.61803398874989484820458683436563
    phi(2) = 1.32471795724474602596090885447809
    '''
    
    if scipy_root_find:
        def root_fun(x):
            return x**(d+1) - x - 1
        return scipy.optimize.root_scalar(root_fun, bracket=[1.0, 1.62], xtol=1e-200)
    
    x = 2.0 
    for i in range(niter): 
        x = pow(1+x, 1/(d+1)) 
    return x

def quasi_random(N, skip=0, init=0.5, alpha=None, normal=True):
    '''
    Generates a 1D sequence of low discrepancy quasi-random numbers using additive recurence
    with alpha = 1/phi where phi is the golden ratio, which has been shown to produce the lowest
    discrepancy of any choice of alpha.
    taken from https://en.wikipedia.org/wiki/Low-discrepancy_sequence#Additive_recurrence
    '''
    if alpha is None:
        alpha = 0.6180339887498948
    N = int(N)
    n = np.arange(skip, N+skip)
    r = (init + alpha*n) % 1
    if normal:
        r = stats.norm.ppf(r)
    return r

def quasi_random_2d(N, skip=0):
    '''
    Generates a 2D sequence of low discrepancy quasi-random numbers using additive recurence
    with alpha = [1/phi, 1/phi**2] where phi is the plastic constant, which has been shown to produce
    the highest 2D packing efficiency of any choice of alpha. The packing efficiency is reported to be
    better than any other 2D quasi-random sequence, including Sobol.
    
    taken from http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    '''
    g = 1.32471795724474602596
    N = int(N)
    n = np.arange(skip, N+skip)
    a1 = 1.0/g
    a2 = 1.0/g**2
    a = np.array([[a1, a2]]).T
    r = (0.5+a*n) % 1
    return r
    
def quasi_random_2d_gen(skip=0):
    g = 1.32471795724474602596
    a1 = 1.0/g
    a2 = 1.0/g**2
    
    i = skip
    while True:
        r1 = (0.5+a1*i) % 1
        r2 = (0.5+a2*i) % 1
        i+=1
        yield r1, r2

def quasi_random_nd(N, m, skip=0, seed=None):
    '''
    Generates quasi-random numbers using additive recurence.
    
    Initial seeds are chosen according to http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    '''
    N = int(N)
    n = np.arange(skip, N+skip)[np.newaxis, :]
    
    if seed is None:
        g = phi(m)
        a = 1/g**(np.arange(1, m+1))
        a = a[:, np.newaxis]
    else:
        a = np.array(seed)[:, np.newaxis]
    
    r = (0.5 + a*n) % 1
    return r
    
def LCG(multiplier=1103515245, modulus=2**31, increment=12345, seed=None):
    '''
    A relatively poor pseudorandom generator based on the Linear Congruential 
    Generator (LCG). Uses the same magic constants as glibc.
    
    Usage:
    LCG returns an infinite generator that continuously spits out numbers.
    Either iterate over it the desired number of times as in the example or use
    the LCGN helper function.
    
    Example:
    gen = LCG()
    out = []
    for i, j in zip(range(10), gen):
        out.append(j)
    '''
    a = multiplier
    c = increment
    m = modulus
    if seed is None:
        seed = int(np.random.rand() * modulus)
    X0 = seed
    while True:
        Xn = (a*X0 + c) % m
        X0 = Xn
        yield Xn
        
def LCGN(N, modulus=2**31, **kwargs):
    '''
    Returns the first N samples of an instance of an LCG pseudorandom generator.
    '''
    N = int(N)
    gen = LCG(modulus=modulus, **kwargs)
    out = np.zeros(N)
    for i,j in zip(range(N), gen):
        out[i] = j
    
    # normalize output
    return out/modulus

def acorn(N, order=120, modulus=2**60, seed=None):
    '''
    A high quality pseudorandom number generator based on the
    Additive Congruential Random Number (ARCORN) generator.
    '''
    
    N = int(N)
    
    if seed is None:
        seed = int(np.random.rand(order) * modulus)
    else:
        order = len(seed)
    
    y = np.zeros([order, N])
    y[:, 0] = seed
        
    for n in range(1, N):
        y[0, n] = y[0, n-1]
        for m in range(1, order):
            y[m, n] = (y[m-1, n] + y[m, n-1]) % modulus
        
    # return the final order
    x = y[-1, :]/modulus
    return x

@np.vectorize
def bernouli_entropy(t):
    '''
    Computes the binary entropy function a.k.a. entropy of a Bernouli process
    
    Domain: 0 < t < 1
    Range: 0 < H <= 1
    '''
    if t == 0 or t == 1:
        H = 0.0
    else:
        H = -t*np.log2(t)-(1-t)*np.log2(1-t)
    return H

@np.vectorize
def xlogx(x, base=np.exp(1)):
    if x == 0:
        return 0.0
    return x*np.log(x)/np.log(base)

def entropy(p, base=np.exp(1), axis=0):
    '''
    Computes the Shannon entropy of a set of probability values.
    
    p :: (N, ...) array
    base :: postive float specifying log base (e.g. base=2 gives entropy in units of bits)
    
    where sum(p, axis=axis) = 1 to be a valid probability
    
    '''
    p = p/np.sum(p, axis=axis)
    H = np.abs(np.sum(xlogx(p, base=base), axis=axis))
    return H

def differential_entropy(p, dx=1, base=np.exp(1), axis=0):
    H = entropy(p, base=base, axis=axis) + np.log(dx)/np.log(base)
    return H

def differential_entropy2(p, base=np.exp(1), axis=0):
    '''
    Only seems to work if np.sum(p)*dx = 1
    '''
    H = np.abs(np.sum(xlogx(p, base=base), axis=axis))/np.sum(p, axis=axis)
    return -H