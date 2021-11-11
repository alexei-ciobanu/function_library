import numpy as np
import scipy

from . import metaplectic
from . import symbolic

try:
    import fun_lib.general_funs as general_funs
except ModuleNotFoundError:
    try:
        import general_funs
    except ModuleNotFoundError:
        pass

def abcd2abcdef(abcd, L, n1=1, n2=1, dx=0, dt=0):
    '''
    L is the physical length of the abcd system.
    Taken from Siegman eq 15.62
    '''
    A,B,C,D = abcd.flatten()
    E = (1-A)*dx + (L-n1*B)*dt
    F = -C*dx + (n2-n1*D)*dt
    return np.array([[A, B, E], 
                     [C, D, F], 
                     [0, 0, 1]])

def grin(d,n0,n2):
    # siegman p586 "h"
    return np.array(\
                    [[np.cos(d*np.sqrt(n2/n0)),np.sin(d*np.sqrt(n2/n0))/(np.sqrt(n2*n0))],\
                     [-np.sqrt(n2*n0)*np.sin(d*np.sqrt(n2/n0)),np.cos(d*np.sqrt(n2/n0))]]\
                   )
                                      
def space(d, n=1, dtype=np.float64):
    d = dtype(d)
    N = np.size(d)
    M = np.zeros([N,2,2], dtype=dtype)
    A = M[...,0,0]
    B = M[...,0,1]
    C = M[...,1,0]
    D = M[...,1,1]
    
    A[:] = 1
    B[:] = d/n
    C[:] = 0
    D[:] = 1
    if N == 1:
        M = np.squeeze(M)
    return M

def general_refraction(n1,n2,R=None,p=None):
    if p is None and R is None:
        p = 0
    elif p is None and R is not None:
        p = 1/R
    if isinstance(p, np.ndarray):
        N = np.size(p)
        M = np.zeros([N,2,2])
        A = M[:,0,0]
        B = M[:,0,1]
        C = M[:,1,0]
        D = M[:,1,1]
        
        A.real = 1
        B.real = 0
        C.real = (n1-n2)/n2*p
        D.real = n1/n2
        return M
    return np.array(\
        [[1,0],\
         [(n1-n2)/n2*p,n1/n2]]\
        )

def lens(f, dtype=np.float64):
    f = dtype(f)
    N = np.size(f)
    M = np.zeros([N,2,2], dtype=dtype)
    A = M[...,0,0]
    B = M[...,0,1]
    C = M[...,1,0]
    D = M[...,1,1]
    
    A[:] = 1
    B[:] = 0
    C[:] = -1/f
    D[:] = 1
    if N == 1:
        M = np.squeeze(M)
    return M
                   
def mirror(Rc):
    return np.array(\
                   [[1,0],\
                    [-2/Rc,1]]\
                   )

def beamsplitter(Rc, alpha=0, direction='x', dtype=np.float64):
    '''
    The tilt alpha is the angle in degrees between the beamsplitter and the xaxis
    '''
    N = np.size(Rc)
    M = np.zeros([N,2,2], dtype=dtype)
    A = M[...,0,0]
    B = M[...,0,1]
    C = M[...,1,0]
    D = M[...,1,1]

    A[:] = 1
    B[:] = 0
    if direction == 'x':
        C[:] = -2/(Rc*np.cos(np.radians(alpha)))
    elif direction == 'y':
        C[:] = -2*np.cos(np.radians(alpha))/Rc
    else:
        raise Exception(f'invalid direction {direction}')
    D[:] = 1
    if N == 1:
        M = np.squeeze(M)
    return M
                   
def lens_p(p, dtype=None):
    p = np.asarray(p)
    if dtype is None:
        dtype = p.dtype
    else:
        p = dtype(p)
    N = np.size(p)
    M = np.zeros([N,2,2], dtype=dtype)
    A = M[...,0,0]
    B = M[...,0,1]
    C = M[...,1,0]
    D = M[...,1,1]
    
    A[:] = 1
    B[:] = 0
    C[:] = -p
    D[:] = 1
    if N == 1:
        M = np.squeeze(M)
    return M
                   
def eig(M):
    A,B,C,D = M.flatten()
    one_on_q_eig = (-(A-D) - np.lib.scimath.sqrt((A-D)**2 + 4*B*C))/(2*B)
    return  1/(one_on_q_eig)

def cavity_stability(M, bool=False):
    A = M[..., 0, 0]
    B = M[..., 0, 1]
    C = M[..., 1, 0]
    D = M[..., 1, 1]
    m = ((A+D)/2)**2
    if bool:
        m = m <= 1
    return m

def unpack(Ms):
    Ms_shape = np.shape(Ms)
    Ms_ndim = np.size(Ms_shape)
    
    if Ms_ndim >= 2:
        A = general_funs.get_last_nd(Ms, [0,0])
        B = general_funs.get_last_nd(Ms, [0,1])
        C = general_funs.get_last_nd(Ms, [1,0])
        D = general_funs.get_last_nd(Ms, [1,1])
    elif (Ms_ndim == 1) and (len(Ms_ndim) == 4):
        A,B,C,D = Ms
    else:
        raise Exception(f'unrecognized type {type(Ms)} for M')
    return A,B,C,D

def inverse(M_abcd):
    A,B,C,D = M_abcd.ravel()
    iM = np.array([[D,-B],[-C,A]])
    return iM
    
def reverse(M):
    # from Tovar & Casperson (1994) Table 2
    A,B,C,D = M.flatten()
    return 1/(A*D-B*C) * np.array(\
                                   [[D,B],\
                                    [C,A]]\
                                   )

def cm(a):
    '''
    Chrip multiplication abcd matrix.
    Just alias the lens matrix with negated input
    '''
    return lens_p(-a)

def cc(a):
    '''
    Chrip convolution abcd matrix.
    Just alias the space matrix
    '''
    return space(a)

def frt(a):
    '''
    Used to reconstruct ABCD from Iwasawa decomposition
    '''

    def frt_wrap(a):
        '''
        Wraps continuous FRT order a to -2 <= a < 2
        '''
        return (a-2)%4 - 2

    # setup and initialization
    a = frt_wrap(np.float64(a))
    t = a*np.pi/2
    N = np.size(a)
    M = np.zeros([N,2,2])
    A = M[...,0,0]
    B = M[...,0,1]
    C = M[...,1,0]
    D = M[...,1,1]

    # general case
    A[:] = np.cos(t)
    B[:] = np.sin(t)
    C[:] = -np.sin(t)
    D[:] = np.cos(t)

    # special cases
    A[np.abs(a)==1] = 0.0
    D[np.abs(a)==1] = 0.0
    B[a==1] = 1.0
    C[a==1] = -1.0
    B[a==-1] = -1.0
    C[a==-1] = 1.0

    A[np.abs(a)==2] = -1.0
    B[np.abs(a)==2] = 0.0
    C[np.abs(a)==2] = 0.0
    D[np.abs(a)==2] = -1.0

    if N == 1:
        M = np.squeeze(M)
    return M

def scaling(s, dtype=np.float64):
    '''
    Used to reconstruct ABCD from Iwasawa decomposition
    '''
    s = dtype(s)
    N = np.size(s)
    M = np.zeros([N,2,2], dtype=dtype)
    A = M[...,0,0]
    B = M[...,0,1]
    C = M[...,1,0]
    D = M[...,1,1]
    
    A[:] = s
    B[:] = 0
    C[:] = 0
    D[:] = 1/s
    if N == 1:
        M = np.squeeze(M)
    return M

def fft(N, dx, lam=1064e-9):
    S = scaling(N*dx**2/lam)
    R = frt(1)
    return S@R

def ifft(N, dx, lam=1064e-9):
    S = scaling(N*dx**2/lam)
    R = frt(-1)
    return S@R

def iwasawa_composition(q,s,r):
    '''
    Great for constructing random ABCD matrices
    '''
    M = cm(q)@scaling(s)@frt(r)
    return M

def iwasawa_decomp(M_abcd, return_matrices=True):
    '''
    Valid for any ABCD matrix.
    Reconstructed by
    M_abcd = lens_p(chirp)@scaling(scaling)@frt(frt)
    '''
    A,B,C,D = unpack(M_abcd)
    
    s = np.sqrt(A**2 + B**2)
    g = (A*C+B*D) / (s**2)
    ga = -1j*np.log((A+1j*B)/s)

    r = ga / (np.pi/2)
    s = s
    q = g

    # get rid of the 'zero' imaginary parts
    r = np.real(r)
    s = np.real(s)
    q = np.real(q)

    if return_matrices:
        return cm(q), scaling(s), frt(r)
    else:
        return q, s, r

def qsc_decomp(M, return_matrices=True):
    '''
    Decomposition breaks down if A=0
    '''
    A,B,C,D = M.ravel()
    a3,a2,a1 = C/A, A, B/A
    if return_matrices:
        return cm(a3), scaling(a2), cc(a1)
    else:
        return a3,a2,a1

def qcq_decomp(M, return_matrices=True):
    '''
    Decomposition breaks down if B=0
    '''
    A,B,C,D = M.ravel()
    a3,a2,a1 = (D-1)/B, B, (A-1)/B
    if return_matrices:
        return cm(a3), cc(a2), cm(a1)
    else:
        return a3,a2,a1
    
def cqc_decomp(M, return_matrices=True):
    '''
    Decomposition breaks down if C=0
    '''
    A,B,C,D = M.ravel()
    a3,a2,a1 = (A-1)/C, C, (D-1)/C
    if return_matrices:
        return cc(a3), cm(a2), cc(a1)
    else:
        return a3,a2,a1

def csq_decomp(M, return_matrices=True):
    '''
    Decomposition breaks down if A=0
    '''
    A,B,C,D = M.ravel()
    a3,a2,a1 = B/D, 1/D, C/D
    if return_matrices:
        return cc(a3), scaling(a2), cm(a1)
    else:
        return a3,a2,a1

def lpl_decomp(M, return_matrices=True):
    '''
    Decomposition breaks down if B=0
    '''
    A,B,C,D = M.ravel()
    a3,a2,a1 = -(D-1)/B, B, -(A-1)/B
    if return_matrices:
        return lens_p(a3), space(a2), lens_p(a1)
    else:
        return a3,a2,a1

def plp_decomp(M, return_matrices=True):
    '''
    Decomposition breaks down if C=0
    '''
    A,B,C,D = M.ravel()
    a3, a2, a1 = (A-1)/C, -C, (D-1)/C
    if return_matrices:
        return space(a3), lens_p(a2), space(a1)
    else:
        return a3,a2,a1

def cav_decomp(M, return_matrices=True):
    a,b,c,d = unpack(M)
    d1 = b/(1 + a)
    p1 = (a - d)/b
    p2 = (1 - a**2)/b
    
    r1 = 2/p1
    r2 = 2/p2
    if return_matrices:
        d1 = space(d1)
        r1 = mirror(r1)
        r2 = mirror(r2)
       
    return r1,d1,r2,d1

def iwasawa_random(N=1, qs=1, ss=1):
    s = 10**(np.random.randn(N)/5 * ss)
    q = np.random.randn(N) * qs
    r = scipy.stats.beta(2,2).rvs(N) + np.random.randint(0,2, size=N)*2
    return iwasawa_composition(q,s,r)

def iwasawa_random_stable_cav():
    stab = False
    while not stab:
        m = iwasawa_random()
        stab = cavity_stability(m, bool=True)
    return m

def accumulate_matmul(ms: list):
    '''
    ABCD matrices are given in left-to-right order i.e. component order not matrix order.
    '''
    m1 = np.eye(2)
    acc_ms = []
    for m2 in ms[::-1]:
        m3 = m2@m1
        acc_ms.append(m3)
        m1 = m3
    return np.array(acc_ms)

def metaplectic_phase(M):
    A,B,C,D = unpack(M)
    z = (A + 1j*B)
    return z/np.abs(z)