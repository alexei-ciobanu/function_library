import numpy as np
import scipy

from . import abcd
from . import CONSTANTS

try:
    import fun_lib.general_funs as general_funs
except ModuleNotFoundError:
    try:
        import general_funs
    except ModuleNotFoundError:
        pass

import numerical_funs as nf

def from_z_w0(z=0, w0=150e-6, lam=None):
    if lam is None:
        lam = complex(CONSTANTS['lambda'])
    return z + 1j*np.pi*w0**2/lam

def from_R_w(R, w, lam=None):
    if lam is None:
        lam = complex(CONSTANTS['lambda'])
    return (1/R - 1j*lam/(np.pi*w**2))**-1

def get_z_zr(q):
    return np.array([q.real, q.imag])

def get_gouy_n(q, n=0):
    n = np.array(n)
    z = np.real(q)
    zR = np.imag(q)
    psi = np.arctan(z/zR)
    return np.exp(1j*((n+1/2))*psi)

def get_gouy_nm(q, n=0, m=0):
    return get_gouy_n(q, n) * get_gouy_n(q, m)

def get_w0(q, lam=CONSTANTS['lambda']):
    '''
    Get waist size from q parameter.
    '''
    lam = float(lam)
    zr = np.imag(q)
    w0 = np.sqrt(zr * lam / np.pi)
    return w0

def get_w(q, lam=CONSTANTS['lambda']):
    '''
    Get beam size from q parameter.

    Alternative forms:
    np.sqrt(-lam/(np.pi*np.imag(1/q)))
    np.sqrt(lam/np.pi) * np.sqrt(q*np.conj(q)/(np.imag(q)))
    np.sqrt(lam/np.pi) * abs(q)/np.sqrt(zr)
    w0 * np.abs(q)/zr
    w0 * np.sqrt(1 + (z/zr)**2)
    '''
    w0 = get_w0(q, lam=lam)
    zr = np.imag(q)
    w = w0 * np.abs(q)/zr
    return w

@np.vectorize
def get_R(q):
    '''
    Alternate forms:
    np.abs(q)**2/np.real(q)
    np.real(1/q)
    '''
    if np.real(q) == 0:
        return np.inf
    else:
        return 1/np.real(1/q)

def get_Theta(q, lam=CONSTANTS['lambda']):
    lam = float(lam)
    w0 = get_w0(q, lam)
    return lam/(np.pi*w0)

def get_q_info(q, lam=CONSTANTS['lambda']):
    out_dict = {}
    out_dict['lam'] = float(lam)
    out_dict['w0'] = get_w0(q, lam)
    out_dict['w'] = get_w(q, lam=lam)
    out_dict['R'] = get_R(q)
    out_dict['Theta'] = get_Theta(q, lam=lam)
    out_dict['Gouy_0_re_im'] = get_gouy_n(q, n=0)
    out_dict['Gouy_0_deg'] = nf.angle(get_gouy_n(q, n=0))
    z,zr = get_z_zr(q)
    out_dict['z'] = z
    out_dict['zr'] = zr
    return out_dict

def propag(q, M, n1=1, n2=1, mode='hadamard'):
    '''
    M.shape = (4,)
    or
    M.shape = M_tail.shape + (2, 2)
    
    if mode == 'hadamard' then q.shape == M_tail.shape
        or M.shape == (4,)
    '''
    M_shape = np.shape(M)
    q_shape = np.shape(q)
    M_ndim = np.size(M_shape)
    q_ndim = np.size(q_shape)
    
    A,B,C,D = abcd.unpack(M)
    
    if mode == 'outer':
        A = general_funs.insert_singletons(A, [0]*q_ndim)
        B = general_funs.insert_singletons(B, [0]*q_ndim)
        C = general_funs.insert_singletons(C, [0]*q_ndim)
        D = general_funs.insert_singletons(D, [0]*q_ndim)
        q = general_funs.insert_singletons(q, [-1]*(M_ndim-2))

    q2 = n2*(A*q/n1 + B)/(C*q/n1 + D)
    return q2

def eig(M):
    '''
    Taken from Siegman Eq. 21.13
    
    Alternative forms:
    q = (-(D-A) - np.lib.scimath.sqrt((D-A)**2 + 4*C*B))/(2*C)
    '''
    A,B,C,D = abcd.unpack(M)
    m = (A+D)/2
    _q = (D-A)/(2*B) + 1j*np.lib.scimath.sqrt(1-m**2)/B
    q = 1/_q
    return q.real + 1j*np.abs(q.imag)