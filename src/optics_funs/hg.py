import numpy as np
import scipy
import scipy.sparse
import matplotlib.pyplot as plt
import time

def maxhg_mode_list(maxhg):
    '''This one counts up by mode order'''
    modes_list = []
    for i in range(0, maxhg+1):
        for j in range(0, maxhg+1):
            modes_list.append((i,j))
    return modes_list

def herm(n, x):
    c = np.zeros(n+1)
    c[-1] = 1
    return np.polynomial.hermite.hermval(x,c)

def u_n_q(x, q, n=0, lam=1064e-9):
    '''
    Does not include instantaneous Gouy phase
    '''
    zr = np.imag(q)
    z = np.real(q)
    w = np.sqrt(-lam/(np.pi*np.imag(1/q)))
    w0 = np.sqrt(-lam/(np.pi*np.imag(1/(q-z))))
    k = 2*np.pi/lam
    
    t1 = np.sqrt(np.sqrt(2/np.pi))
    t2 = np.sqrt(1.0/(2.0**n*np.math.factorial(n)*w0))
    t3 = np.sqrt(w0/w)
    norm = t1*t2*t3
    
    u = herm(n, np.sqrt(2)*x/w) * np.exp(-1j*k*x**2/(2*q))
    E = norm * u
    
    return E

def HG_projection_sparse_1D(U, xs, qx=1j, maxhg=0):
        
    dx = xs[1] - xs[0]
    Nx = len(xs)
        
    nx_s = np.arange(maxhg+1)
    
    Ei = scipy.sparse.lil_matrix((max(nx_s)+1, Nx), dtype=np.complex128)
    for n in nx_s:
        Ei[n,:] = np.conj(u_n_q(xs, qx, n=n))
    
    out = Ei@U *dx
        
    return out

def iHG_projection_sparse_1D(H, xs, qx=1j):
    
    Nx = len(xs)
    dx = xs[1] - xs[0]
    N = np.size(H)
    nx_s = np.arange(N)
    
    E = scipy.sparse.lil_matrix((Nx, max(nx_s)+1), dtype=np.complex128)
    for n in nx_s:
        E[:,n] = u_n_q(xs, qx, n=n)
    
    out = E@H

    return out

def HG_projection_sparse(U, xs, ys=None, qx=1j, qy=None, mode_list=[(0,0)], maxhg=None):
    if qy is None:
        qy = qx
    if ys is None:
        ys = xs
        
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    
    Ny = len(ys)
    Nx = len(xs)
    
    if maxhg is not None:
        mode_list = maxhg_mode_list(maxhg)
        
    nx_s = list(set(nm[0] for nm in mode_list))
    ny_s = list(set(nm[1] for nm in mode_list))
    
    Exi = scipy.sparse.lil_matrix((Nx, max(nx_s)+1), dtype=np.complex128)
    for n in nx_s:
        Exi[:,n] = np.conj(u_n_q(xs, qx, n=n))
    
    Eyi = scipy.sparse.lil_matrix((max(ny_s)+1, Ny), dtype=np.complex128)
    for m in ny_s:
        Eyi[m,:] = np.conj(u_n_q(ys, qy, n=m))
    
    out = Eyi@U@Exi * dy*dx
    out = out.T
        
    return out

def iHG_projection_sparse(H, xs, ys=None, qx=1j, qy=None):
    if qy is None:
        qy = qx
    if ys is None:
        ys = xs
    
    Ny = len(ys)
    Nx = len(xs)
    
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    
    N,M = np.shape(H)
    
    nx_s = np.arange(N)
    ny_s = np.arange(M)
    
    Ex = scipy.sparse.lil_matrix((max(nx_s)+1, Nx), dtype=np.complex128)
    for n in nx_s:
        Ex[n,:] = u_n_q(xs, qx, n=n)
    
    Ey = scipy.sparse.lil_matrix((Ny, max(ny_s)+1), dtype=np.complex128)
    for m in ny_s:
        Ey[:,m] = u_n_q(ys, qy, n=m)
    
    out = Ey@(H.T)@Ex
        
    return out