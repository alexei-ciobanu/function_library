import numpy as np
import scipy.special
import pylops

import numerical_funs as nf
import optics_funs as of
import general_funs as gef
from debug_funs import inplace_print

class LCT_operator:
    def __init__(self,x1s=None,x2s=None,M=None,type=None,lam=1064e-9,op_list=None):
        self.M = M
        self.x1s = x1s
        self.x2s = x2s
        self.lam = lam
        self.scalar = 1
        if op_list is None:
            self.type = type
            self.op_list = [self]
        elif op_list == []:
            self.type = 'LCT'
            self.M = LCT_space(0)
        else:
            self.type = 'compound'
            self.op_list = op_list
        
    def __matmul__(self,other):
        assert(type(self) is type(other))
        if self.type == other.type == 'LCT':
            Op = LCT_operator(type='LCT')
            Op.M = self.M@other.M
            Op.scalar = self.scalar*other.scalar
            return Op
        else:
            Op = LCT_operator(type='compound')
            # print(self.op_list[-1].type,other.op_list[0].type)
            if self.op_list[-1].type == other.op_list[0].type == 'LCT':
                Op.op_list = self.op_list[:-1] + [self.op_list[-1]@other.op_list[0]] + other.op_list[1:]
            else:
                Op.op_list = self.op_list + other.op_list
            return Op
            
    def __mul__(self,other):
        self.scalar *= other
        return self
             
    def __rmul__(self,other):
        return self.__mul__(other)
            
    def build(self):
        if self.type == 'LCT':
            return self.scalar**0.5 * LCT1D(self.x1s,self.x2s,self.M)
        elif self.type == 'map':
            return self.scalar * self.map()
        elif self.type == 'compound':
            return build(self.op_list[0]) @ build(LCT_operator(op_list=self.op_list[1:]))
        else:
            raise Exception(f"don't know how to build operator type: {self.type}")

class LCTNodeParams(object):
    '''
    A class for storing LCT parameters that are needed to build an LCT
    kernel that don't have a physical signinficance
    '''
    def __init__(self, **kwargs):
        N = kwargs.pop('N', None)
        s = kwargs.pop('s', None)

        if N is None:
            self.Nx = kwargs.pop('Nx', 51)
            self.Ny = kwargs.pop('Ny', 50)
        else:
            self.Nx = N
            self.Ny = N
        
        if s is None:
            self.sx = kwargs.pop('sx', None)
            self.sy = kwargs.pop('sy', None)
        else:
            self.sx = s
            self.sy = s
            
        self.wx = kwargs.pop('wx', None)
        self.wy = kwargs.pop('wy', None)
        
        self._xs = kwargs.pop('xs', None)
        self._ys = kwargs.pop('ys', None)

    @property
    def xs(self):
        if self._xs is None:
            out = np.linspace(-1, 1, self.Nx) * self.sx * self.wx
        else:
            out = self._xs
        return out

    @property
    def ys(self):
        if self._ys is None:
            out = np.linspace(-1, 1, self.Ny) * self.sy * self.wy
        else:
            out = self._ys
        return out

    @property
    def dx(self):
        return self.sx * self.wx / (self.Nx - 1) * 2

    @property
    def dy(self):
        return self.sy * self.wy / (self.Ny - 1) * 2

    @property
    def N_points(self):
        return self.Nx * self.Ny

    def __repr__(self):
        return repr(self.__dict__)

class LCTEdgeParams(object):
    '''
    A class for storing LCT parameters that are needed to build an LCT
    kernel that don't have a physical signinficance
    '''
    def __init__(self, **kwargs):
        N = kwargs.pop('N', None)
        s = kwargs.pop('s', None)

        if N is None:
            self.N1x = kwargs.pop('N1x', 51)
            self.N2x = kwargs.pop('N2x', 51)
            self.N1y = kwargs.pop('N1y', 50)
            self.N2y = kwargs.pop('N2y', 50)
        else:
            self.N1x = N
            self.N2x = N
            self.N1y = N
            self.N2y = N
        
        if s is None:
            self.s1x = kwargs.pop('s1x', 4)
            self.s2x = kwargs.pop('s2x', 4)
            self.s1y = kwargs.pop('s1y', 4)
            self.s2y = kwargs.pop('s2y', 4)
        else:
            self.s1x = s
            self.s2x = s
            self.s1y = s
            self.s2y = s
            
        self.w1x = kwargs.pop('w1x', None)
        self.w2x = kwargs.pop('w2x', None)
        self.w1y = kwargs.pop('w1y', None)
        self.w2y = kwargs.pop('w2y', None)
        
        self._x1s = kwargs.pop('x1s', None)
        self._x2s = kwargs.pop('x2s', None)
        self._y1s = kwargs.pop('y1s', None)
        self._y2s = kwargs.pop('y2s', None)
        
        if (self.N1x != self.N2x) or (self.N1y != self.N2y):
            raise ValueError('Hard to make round trip operators when the input/output grid are not the same size')

    @property
    def x1s(self):
        if self._x1s is None:
            out = np.linspace(-1, 1, self.N1x) * self.s1x * self.w1x
        else:
            out = self._x1s
        return out

    @property
    def x2s(self):
        if self._x2s is None:
            out = np.linspace(-1, 1, self.N2x) * self.s2x * self.w2x
        else:
            out = self._x2s
        return out

    @property
    def y1s(self):
        if self._y1s is None:
            out = np.linspace(-1, 1, self.N1y) * self.s1y * self.w1y
        else:
            out = self._y1s
        return out

    @property
    def y2s(self):
        if self._y2s is None:
            out = np.linspace(-1, 1, self.N2y) * self.s2y * self.w2y
        else:
            out = self._y2s
        return out

    @property
    def dx1(self):
        return self.s1x * self.w1x / (self.N1x - 1) * 2

    @property
    def dx2(self):
        return self.s2x * self.w2x / (self.N2x - 1) * 2

    @property
    def dy1(self):
        return self.s1y * self.w1y / (self.N1y - 1) * 2

    @property
    def dy2(self):
        return self.s2y * self.w2y / (self.N2y - 1) * 2

def abcd_space(d,n=1):
    if isinstance(d,np.ndarray):
        N = np.size(d)
        M = np.zeros([N,2,2])
        A = M[:,0,0]
        B = M[:,0,1]
        C = M[:,1,0]
        D = M[:,1,1]
        
        A.real = 1
        B.real = d/n
        C.real = 0
        D.real = 1
        return M
    else:
        return np.array(\
                       [[1,d/n],\
                        [0,1]]\
                       )

def abcd_lens(f):
    if isinstance(f,np.ndarray):
        N = np.size(f)
        M = np.zeros([N,2,2])
        A = M[:,0,0]
        B = M[:,0,1]
        C = M[:,1,0]
        D = M[:,1,1]
        
        A.real = 1
        B.real = 0
        C.real = -1/f
        D.real = 1
        return M
    else:
        return np.array(\
                   [[1,0],\
                    [-1/f,1]]\
                   )

def abcd_general_refraction(n1,n2,Rc=np.inf):
    return np.array(\
        [[1,0],\
         [(n2-n1)/Rc,1]]\
        )

def LCT_space(z,nr=1,lam=1064e-9):
    return abcd_space(z*lam,nr)

def LCT_lens(f,lam=1064e-9):
    return abcd_lens(f*lam)
    
def LCT_mirror(Rc,nr=1,lam=1064e-9):
    return abcd_lens(Rc*lam/(2*nr))

def LCT_general_refraction(n1,n2,Rc=np.inf,lam=1064e-9):
    return abcd_general_refraction(n1,n2,Rc=Rc*lam)

def LCT_inverse(M):
    a,b,c,d = M.flatten()
    return np.array([[d,-b],[-c,a]])

def LCTabcd_to_LCTabg(M=None, M_abcd=None, lam=1064e-9):
    if M is not None:
        a,b,c,d = M.flatten()
    elif M_abcd is not None:
        a,b,c,d = M_abcd.flatten()
        b = b*lam
        c = c/lam

    beta = 1/b
    gamma = a*beta
    alpha = d*beta
    return alpha,beta,gamma

def abcd_to_LCTabg(M_abcd=None, lam=1064e-9):
    a,b,c,d = M_abcd.flatten()
    b = b*lam
    c = c/lam

    beta = 1/b
    gamma = a*beta
    alpha = d*beta
    return alpha,beta,gamma

def abcd2lct(M, lam=1064e-9):
    A,B,C,D = np.ravel(M)
    return np.array([[A, B*lam],[C/lam, D]])
    
def QP_kernel(x,f=None,R=None,lam=1064e-9):
    if not R is None:
        f = R/2        
    f = f*lam
    
    return np.diag(np.exp(1j*np.pi/f*x**2))

def centered_dft(N):
    F = nf.centered_dft(N)
    return np.sqrt(1j)*F
        
def LCT1D(x1s, x2s=None, M=None, M_abcd=None, abg=None, lam=1064e-9, return_arg=False, return_darg=False):
    if x2s is None:
        x2s = x1s
    if M is not None:
        a,b,c,d = M.flatten()
    elif M_abcd is not None:
        a,b,c,d = M_abcd.flatten()

    if b == 0:
        assert(a == d == 1)
        if c == 0:
            return np.eye(len(x1s))
        else:
            return CM_kernel(x1s, C=c)

    if abg is None:
        alpha, beta, gamma = LCTabcd_to_LCTabg(M=M, M_abcd=M_abcd, lam=lam)
    if abg is not None:
        alpha, beta, gamma = abg
    
    N1 = np.size(x1s)
    N2 = np.size(x2s)
    
    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]

    arg = np.zeros([N2,N1],dtype=np.complex128)
    for i,x1 in enumerate(x1s):
        arg[:,i] = (alpha*x2s**2 - 2*beta*x1*x2s + gamma*x1**2)

    if return_arg:
        return arg

    if return_darg:
        x1g,x2g = np.meshgrid(x1s,x2s)
        darg = np.abs(2*gamma*x1g - 2*beta*x2g)*dx1 + np.abs(2*alpha*x2g - 2*beta*x1g)*dx2
        return darg
    
    # DLCT = np.zeros([N2,N1],dtype=np.complex128)
    # for i,x1 in enumerate(x1s):
    #     DLCT[:,i] = dx1*np.sqrt(beta+0j) * np.exp(1j*np.pi/4) * np.exp(-1j*np.pi*(alpha*x2s**2 - 2*beta*x1*x2s + gamma*x1**2))

    DLCT = dx1*np.sqrt(1j*beta) * np.exp(-1j*np.pi*arg)
        
    return DLCT

def LCT1D_windowed(x1s, x2s, M_abcd=None, abg=None, lam=1064e-9):
    if abg is None:
        alpha, beta, gamma = LCTabcd_to_LCTabg(M_abcd=M_abcd, lam=lam)
    if abg is not None:
        alpha, beta, gamma = abg

    N1 = np.size(x1s)
    N2 = np.size(x2s)
    
    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]

    x1g,x2g = np.meshgrid(x1s,x2s)

    arg = alpha*x2g**2 - 2*beta*x1g*x2g + gamma*x1g**2
    darg_x1 = (2*gamma*x1g - 2*beta*x2g)*dx1
    darg_x2 = (2*alpha*x2g - 2*beta*x1g)*dx2
    darg = np.abs(darg_x1) + np.abs(darg_x2)

    DLCT = dx1*np.sqrt(beta+0j) * np.exp(1j*np.pi/4) * np.exp(-1j*np.pi*arg)
    # DLCT[darg>2] = DLCT[darg>2] * np.exp(-8.6*(darg[darg>2]-2)**1)

    c0 = 2
    DLCT[darg>c0] = DLCT[darg>c0] * np.exp(-2*(darg[darg>c0]-c0)**2)
    # DLCT[darg>c0] = 0

    return DLCT

def DLCT(x1s,x2s,M,lam=1064e-9):
    '''Alias for LCT1D'''
    return LCT1D(x1s, x2s, M, lam=1064e-9)

def LCT1D_hq(x1s, x2s=None, M_abcd=None, lam=1064e-9):
    '''
    A higher quality version of the DLCT kernel that performs the subpixel integration analytically.
    
    This significantly reduces kernel aliasing but doesn't entirely remove it.
    This is because while the kernel function is integrated the input field isn't.
    So the field is implicitly assumed to be constant over the subpixel.
    
    This could be improved by constructing the optimal kernel by performing an interpolated analyitcal integration of the kernel times the field in the subpixel.
    Linear interoplation is solvable in wolfram. And so is cubic. In 2D the optimal kernel sounds like a pain.    
    Bilinear is separable, but bicubic is not.
    '''
    if x2s is None:
        x2s = x1s
    
    sqrt = np.sqrt
    A,B,C,D = np.complex128(M_abcd).ravel()
    
    dx1 = x1s[1] - x1s[0]
    
    x1l = x1s - dx1/2
    x1u = x1s + dx1/2
    
    ea = scipy.special.erfi((-1)**(3/4)*np.sqrt(np.pi)*(np.add.outer(-x2s, A*x1l))/(sqrt(A)*sqrt(B*lam)))
    eb = scipy.special.erfi((-1)**(3/4)*np.sqrt(np.pi)*(np.add.outer(-x2s, A*x1u))/(sqrt(A)*sqrt(B*lam)))
    t1 = np.exp((1j*np.pi*x2s[:,None]**2)/(A*B*lam)) 
    t2 = np.exp(-1j*np.pi*D*x2s[:,None]**2/(B*lam)) 
    consts = (-1)**(1/4) * sqrt(B*lam) / (2*sqrt(A)) * sqrt(1j/(B*lam))
    
    L = consts*t1*t2*(ea-eb)
    return L

def CC_kernel(xs, d, lam=1064e-9):
    dx = xs[1] - xs[0]
    beta = 1/(d*lam)
    kernel = dx*np.sqrt(1j*beta)*np.exp(-1j*np.pi*beta*np.subtract.outer(xs,xs)**2)
    return kernel

def F_CC_kernel(x1s, d, lam=1064e-9, centered=True):
    '''
    Returns the 2D Fourier transform of the Chirp Convolution kernel.
    Applying F^(-1)@kernel@F gives back just the CC kernel.
    '''
    N = x1s.size
    if d==0:
        return np.eye(N)
    beta = 1/(d*lam)
    dx = x1s[1] - x1s[0]
    f1s = np.fft.fftfreq(N,dx)
    if centered:
        # use centered dft / xft
        f1s = x1s/(dx**2*N)
    diag = np.exp(1j*np.pi/beta*f1s**2)
    kernel = np.diag(diag)
    return kernel
    
def F_CM_kernel(xs, C, lam=1064e-9, centered=True):
    N = xs.size
    dx = xs[1] - xs[0]
    f1s = np.fft.fftfreq(N, dx)
    f2s = np.fft.fftfreq(N, dx)
    df = f1s[1] - f1s[0]

    if C == 0:
        return np.eye(N)

    if centered:
        f1s = np.fft.fftshift(f1s)
        f2s = np.fft.fftshift(f2s) 
    C = C/lam
    kernel = df*np.exp(-1j*np.pi/C*np.subtract.outer(f1s,f1s)**2)*np.sqrt(1j/C)
    return kernel

def _LCT1D_aa_builder(a,b,c,dx2):
    '''
    Hidden function used by LCT1D_aa() to aid in building its matrix elements.
    The original mathematical expressions were obtained from the partial solution
    to the indefinite integral of the LCT kernel provided by Mathematica.

    The Mathematica expression came from evaluating:
    (1/dy) Sqrt[I] Sqrt[b] Integrate[Exp[-I Pi(a y^2 - 2 b x y + c x^2)], x, y]
    '''
    def g(x2):
        def f(x1):
            a1 = -(1j*np.sqrt(b+0j))/(2*dx2*np.sqrt(a+0j))
            a2 =  np.exp((1j*(b**2 - a*c)*np.pi*x1**2)/a)
            b1 = ((-1)**(3/4)*np.sqrt(np.pi)*(-(b*x1) + a*x2))/np.sqrt(a+0j)
            a3 = scipy.special.erfi(b1)
            return a1*a2*a3
        return f
    return g

def LCT1D_aa(x1s, x2s, M=None, M_abcd=None, lam=1064e-9, verbose=False):
    '''
    An anti-aliased version of the LCT1D kernel. The anti-aliasing is implemented by quad integration
    of each matrix element in the LCT kernel. This causes the matrix elements go to zero in the region where
    the LCT kernel is oscillating faster than the sampling frequency. 
    
    It should converge to the analytical LCT as the sampling frequency tends to infinity, though it 
    doesn't converge as fast as LCT1D(). This appears to be the price you have to pay to obtain a well
    behaved kernel.
    '''
    dx2 = x2s[1] - x2s[0]
    dx1 = x1s[1] - x1s[0]
    h = _LCT1D_aa_builder(*LCTabcd_to_LCTabg(M=M, M_abcd=M_abcd, lam=lam), dx2)
    
    A = np.zeros([x2s.size+1, x1s.size], dtype=np.complex128)
    # loop over matrix elements computing the integral of the LCT kernel in that region
    for i, x1i in enumerate(x1s):
        if verbose:
            if not i%10:
                print(i)
        for j, x2j in enumerate(x2s):
            A[j, i] = nf.quad_complex(h(x2j-dx2/2), x1i-dx1/2, x1i+dx1/2)[0]

    # complete the last column as a special case
    for i,x1i in enumerate(x1s):
        A[-1, i] = nf.quad_complex(h(x2s[-1]+dx2/2), x1i-dx1/2, x1i+dx1/2)[0]

    # a hack to implement fundamental theory of calculus in half the time by reusing nearby columns
    kernel = np.diff(A, axis=0)
    return kernel

def LCT1D_lpl_real(xs, M_abcd, lam=1064e-9):
    '''
    Always samples from real space, should be equivelant to regular LCT1D
    '''
    N = len(xs)
    p1,d1,p2 = of.abcd_lpl_decomp(M_abcd, return_matrices=False)
    x_span = xs[-1] - xs[0]
    dx = xs[1] - xs[0]

    M1 = CM_kernel(xs, p1, lam=lam)
    M2 = CC_kernel(xs, d1, lam=lam)
    M3 = CM_kernel(xs, p2, lam=lam)
        
    return M1@M2@M3

def LCT1D_lpl_fourier(xs, M_abcd, lam=1064e-9, output_domain='real'):
    '''
    Always samples from Fourier space, should be equivelant to regular LCT1D
    '''
    N = len(xs)
    p1,d1,p2 = of.abcd.lpl_decomp(M_abcd, return_matrices=False)
    x_span = xs[-1] - xs[0]
    dx = xs[1] - xs[0]

    F = nf.dft_kernel(N)
    iF = np.conj(F).T

    M1 = F_CM_kernel(xs, p1, lam=lam)
    M2 = F_CC_kernel(xs, d1, lam=lam)
    M3 = F_CM_kernel(xs, p2, lam=lam)

    out = M1@M2@M3

    if output_domain == 'real':
        out = iF@out@F
    elif output_domain == 'fourier':
        out = out
        
    return out

def LCT1D_lpl_cond(xs, M_abcd, lam=1064e-9, return_subkernels=False):
    '''
    Always samples the lenses from real space and the spaces from fourier space.
    This always ensures the condition number is close to 1.
    '''
    N = len(xs)
    x_span = xs[-1] - xs[0]
    dx = xs[1] - xs[0]

    F = nf.centered_dft(N)
    iF = np.conj(F).T

    A,B,C,D = M_abcd.ravel()
    if B != 0:
        p2,d1,p1 = of.abcd_lpl_decomp(M_abcd, return_matrices=False)
        M1 = CM_kernel(xs, p1, lam=lam)
        M2 = iF@F_CC_kernel(xs, d1, lam=lam)@F
        M3 = CM_kernel(xs, p2, lam=lam)
        if return_subkernels:
            return M3, M2, M1
        return M3@M2@M1
    elif B == 0 and C != 0:
        d2,p1,d1 = of.abcd_plp_decomp(M_abcd, return_matrices=False)
        print(d2,p1,d1)
        M3 = iF@F_CC_kernel(xs, d2, lam=lam)@F
        M2 = CM_kernel(xs, p1, lam=lam)
        M1 = iF@F_CC_kernel(xs, d1, lam=lam)@F
        if return_subkernels:
            return M3, M2, M1
        return M3@M2@M1

def LCT1D_lpl2(xs, M_abcd, lam=1064e-9, return_submatrices=False, fast=False):
    '''
    An formulation of the CM-CC-CM LCT kernel using purely CM_kernels and DFT matrices
    by carefully applying the appropriate scaling factors that result from commuting out
    the DFT related scaling operators. 
    '''
    A,B,C,D = M_abcd.ravel()
    N = len(xs)
    dx = xs[1] - xs[0]
    F = nf.centered_dft(N)
    iF = np.conj(F).T
    scale = -lam/(N*dx**2)
    if fast:
        raise NotImplementedError
        q3 = None
        q2 = None
        q1 = None
        return nf.xft(q1)
    else:
        Q3 = CM_kernel(xs, (D-1)/B, lam=lam)
        Q2 = CM_kernel(xs, -B*scale**2, lam=lam)
        Q1 = CM_kernel(xs, (A-1)/B, lam=lam)
        if return_submatrices:
            return Q3, iF@Q2@F, Q1
        else:
            return Q3@iF@Q2@F@Q1

def LCT1D_lpl3(x1s, x2s, M_abcd, lam=1064e-9):
    '''
    An formulation of the CM-CC-CM LCT kernel using purely CM_kernels and DFT matrices
    by carefully applying the appropriate scaling factors that result from commuting out
    the DFT related scaling operators. 
    
    Supports dynamic resizing of input/output grids by applying the appropriate scaling ABCD
    to the output to match the requested output grid.
    
    TODO:
    The scaling ABCD can be applied to either the input or the output (or both), each gives different results
    There is likely and optimal combination.
    '''
    
    N1 = len(x1s)
    N2 = len(x2s)
    if N1 != N2:
        raise ValueError('input and output grid must have the same number of points')
        
    s1 = np.ptp(x1s)
    s2 = np.ptp(x2s)
    scl = s2/s1
    M_scl = of.abcd.scaling(1/scl)
    M_abcd = M_scl@M_abcd
    
    A,B,C,D = M_abcd.ravel()
    dx = x1s[1] - x1s[0]
    F = nf.centered_dft(N1)
    iF = np.conj(F).T
    scale = -lam/(N1*dx**2)
    
    Q3 = CM_kernel(x1s, (D-1)/B, lam=lam)
    Q2 = CM_kernel(x1s, -B*scale**2, lam=lam)
    Q1 = CM_kernel(x1s, (A-1)/B, lam=lam)
    
    out = np.sqrt(1/scl) * Q3@iF@Q2@F@Q1
    return out

def LCT1D_plp2(xs, M_abcd, lam=1064e-9, return_submatrices=False, fast=False):
    '''
    An formulation of the CM-CC-CM LCT kernel using purely CM_kernels and DFT matrices
    by carefully applying the appropriate scaling factors that result from commuting out
    the DFT related scaling operators. 
    '''
    A,B,C,D = M_abcd.ravel()
    N = len(xs)
    dx = xs[1] - xs[0]
    F = nf.centered_dft(N)
    iF = np.conj(F).T
    scale = -lam/(N*dx**2)
    if fast:
        raise NotImplementedError
        q3 = None
        q2 = None
        q1 = None
        return nf.xft(q1)
    else:
        Q3 = CM_kernel(xs, -(A-1)/C*scale**2, lam=lam)
        Q2 = CM_kernel(xs, C, lam=lam)
        Q1 = CM_kernel(xs, -(D-1)/C*scale**2, lam=lam)
        if return_submatrices:
            return iF@Q3@F, Q2, iF@Q1@F
        else:
            return iF@Q3@F@Q2@iF@Q1@F

def LCT1D2(xs, M_abcd, lam=1064e-9):
    A,B,C,D = M_abcd.ravel()
    if np.isclose(B,0):
        return LCT1D_plp2(xs, M_abcd=M_abcd, lam=lam)
    else:
        return LCT1D_lpl2(xs, M_abcd=M_abcd, lam=lam)

def LCT1D_lpl_adaptive(xs, M_abcd, lam=1064e-9):
    '''
    Adaptively samples from subkernels from either fourier or real space to ensure 
    that the kernel is always oversampled.
    '''
    N = len(xs)
    p1,d1,p2 = of.abcd_lpl_decomp(M_abcd, return_matrices=False)
    x_span = xs[-1] - xs[0]
    dx = xs[1] - xs[0]
    
    d_opt = dx*x_span/lam
    p_opt = lam/(dx*x_span)
    
    F = nf.centered_dft(N)
    iF = np.conj(F)
     
    if np.abs(p1) >= p_opt:
        CM = F_CM_kernel(xs, p1, lam=lam)
        M1 = iF@CM@F
    else:
        M1 = CM_kernel(xs, p1, lam=lam)

    if np.abs(d1) <= d_opt:
        CC = F_CC_kernel(xs, d1, lam=lam)
        M2 = iF@CC@F
    else:
        M2 = CC_kernel(xs, d1, lam=lam)

    if np.abs(p2) >= p_opt:
        CM = F_CM_kernel(xs, p2, lam=lam)
        M3 = iF@CM@F
    else:
        M3 = CM_kernel(xs, p2, lam=lam)
        
    return M1@M2@M3

def CM_kernel_pei(N, C):
    n = (np.arange(N) - (N-1)/2)/np.sqrt(N)
    d = np.exp(1j*np.pi * C * n**2)
    return np.diag(d)

def CM_kernel(xs, C, lam=1064e-9, diag=False):
    '''
    Chirp multiplication kernel
    '''
    C = C/lam
    d = np.exp(-1j*np.pi * C * xs**2)
    if not diag:
        d = np.diag(d)
    return d

def scaling_pei(N, M_abcd):
    A,B,C,D = M_abcd.ravel()
    sigma = A
    F = nf.centered_dft(N)
    iF = np.conj(F).T
    if sigma >= 1:
        C1 = CM_kernel_pei(N, sigma)
        C2 = CM_kernel_pei(N, 1/sigma)
        return np.sqrt(-1j)*F@C1@iF@C2@F@C1
    if sigma < 1:
        C1 = CM_kernel_pei(N, -sigma)
        C2 = CM_kernel_pei(N, -1/sigma)
        return np.sqrt(1j)*C2@iF@C1@F@C2@iF

def scaling_pei2(xs, M_abcd, lam=1064e-9):
    N = len(xs)
    dx = xs[1] - xs[0]
    A,B,C,D = M_abcd.ravel()
    sigma = A
    F = nf.centered_dft(N)
    iF = np.conj(F).T
    if sigma >= 1:
        C1 = CM_kernel(xs/dx/np.sqrt(N/lam), sigma, lam=lam)
        C2 = CM_kernel(xs/dx/np.sqrt(N/lam), 1/sigma, lam=lam)
        return np.sqrt(-1j)*F@C1@iF@C2@F@C1
    if sigma < 1:
        C1 = CM_kernel(xs/dx/np.sqrt(N/lam), -sigma, lam=lam)
        C2 = CM_kernel(xs/dx/np.sqrt(N/lam), -1/sigma, lam=lam)
        return np.sqrt(1j)*C2@iF@C1@F@C2@iF

def RS1D_kernel(x1s, x2s=None, d=1, lam=1064e-9):
    '''
    1D version of Rayleigh-Sommerfeld (RS) diffraction integral (without obliquity factors) 
    to compare against the LCT.
    
    Comments: This may be improper as RS seems to be defined for 2D in a what seems to be a
    non-separable form. So rewritting as a separable 1D integral may not be the right thing to do.
    '''
    if x2s is None:
        x2s = x1s
    N1 = np.size(x1s)
    N2 = np.size(x2s)
    
    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]
    
    r = np.zeros([N2,N1],dtype=np.complex128)
    for i,x1 in enumerate(x1s):
        r[:,i] = np.sqrt((x2s-x1)**2 + d**2)
        
    return dx1*np.sqrt(1j/lam) * np.exp(-1j*2*np.pi/lam*r)/np.sqrt(r) * np.exp(1j*2*np.pi/lam*d)

def generator_FRT(U,D):
    G = np.pi**2*(U@U+D@D)/2
    return G

def generator_CM(U,D):
    G = np.pi*(U@U)
    return G

def generator_CC(U,D):
    G = np.pi*(D@D)
    return G

def generator_scaling(U,D):
    G = np.pi*(U@D+D@U)
    return G

def hyperdifferential_xs_scale(xs, lam=1064e-9):
    N = len(xs)
    xs_span = np.ptp(xs)
    dx = xs[1] - xs[0]
    # xs_scale = lam*(N-1)**2/(N * xs_span**2)
    xs_scale = lam/(N * dx**2)
    return xs_scale

def hyperdifferential_FRT(U,D,a):
    G = generator_FRT(U,D)
    FRT = scipy.linalg.expm(1j*a*G)
    return FRT

def hyperdifferential_CM(U,D,q):
    G = generator_CM(U,D)
    Q = scipy.linalg.expm(-1j*q*G)
    return Q

def hyperdifferential_CC(U,D,z):
    G = generator_CC(U,D)
    C = scipy.linalg.expm(1j*z*G)
    return C

def hyperdifferential_scaling(U,D,m):
    G = generator_scaling(U,D)
    SC = scipy.linalg.expm(1j*(np.log(m))*G)
    return SC

def U_structural_bak(N):
    n = np.arange(N)
    u = np.sin(2*np.pi/N*(n-(N-1)/2))
    U = np.diag(u)
    U = (np.sqrt(N)/(np.pi))*U
    return U

def U_structural_alt(N):
    D = D_structural_alt(N)
    F = nf.centered_dft(N)
    iF = np.conj(F).T
    U = F@D@iF
    return U

def U_structural(N):
    ns = np.arange(N) - (N-1)/2
    u = np.sin(2*np.pi/N*ns)
    U = np.diag(u)
    U = U/np.pi
    return U

def D_structural(N):
    U = U_structural(N)
    F = nf.centered_dft(N)
    iF = np.conj(F).T
    D = iF@U@F
    return D

def D_structural_alt(N):
    D = np.zeros([N,N], dtype=complex)
    D[nf.diag_indices_circulant(N, k=1)] = -1j/(2*np.pi)
    D[nf.diag_indices_circulant(N, k=-1)] = 1j/(2*np.pi)
    return D

def FRT_structural(N,a):
    U = U_structural(N)
    D = D_structural(N)
    return hyperdifferential_FRT(U,D,a)

def CM_structural(N,q):
    U = U_structural(N)
    D = D_structural(N)
    return hyperdifferential_CM(U,D,q)

def CC_structural(N,q):
    U = U_structural(N)
    D = D_structural(N)
    return hyperdifferential_CC(U,D,q)

def scaling_structural(N,m):
    U = U_structural(N)
    D = D_structural(N)
    return hyperdifferential_scaling(U,D,m)

def LCT_structural(xs, M_abcd, lam=1064e-9):
    A,B,C,D = M_abcd.ravel()
    N = len(xs)
    xs_scale = hyperdifferential_xs_scale(xs, lam=lam)
    s = of.abcd_scaling(np.sqrt(xs_scale))
    si = of.abcd_scaling(1/np.sqrt(xs_scale))
    # M2 = np.array([[A,B*xs_scale],[C/xs_scale,D]])
    M2 = s@np.array([[A,B],[C,D]])@si
    q,m,a = of.abcd_to_iwasawa(M2, return_matrices=False)
    FRT = FRT_structural(N, a)
    Q = CM_structural(N, q)
    SC = scaling_structural(N, m)
    return Q@SC@FRT

def U_numerical(N):
    n = (np.arange(N)-(N-1)/2)
    u = n/np.sqrt(N)
    U = np.diag(u)
    return U

def D_numerical(N):
    U = U_numerical(N)
    F = nf.centered_dft(N)
    iF = np.conj(F).T
    D = iF@U@F
    return D

def FRT_numerical(N,a):
    U = U_numerical(N)
    D = D_numerical(N)
    return hyperdifferential_FRT(U,D,a)

def CM_numerical(N,q):
    U = U_numerical(N)
    D = D_numerical(N)
    return hyperdifferential_CM(U,D,q)

def CC_numerical(N,q):
    U = U_numerical(N)
    D = D_numerical(N)
    return hyperdifferential_CC(U,D,q)

def scaling_numerical(N,m):
    U = U_numerical(N)
    D = D_numerical(N)
    return hyperdifferential_scaling(U,D,m)

def LCT_numerical(xs, M_abcd, lam=1064e-9):
    A,B,C,D = M_abcd.ravel()
    N = len(xs)
    xs_scale = hyperdifferential_xs_scale(xs, lam=lam)
    s = of.abcd.scaling(np.sqrt(xs_scale))
    si = of.abcd.scaling(1/np.sqrt(xs_scale))
    # M2 = np.array([[A,B*xs_scale],[C/xs_scale,D]])
    # M2 = s@np.array([[A,B],[C,D]])@si
    M2 = M_abcd
    q,m,a = of.abcd.iwasawa_decomp(M2, return_matrices=False)
    R = FRT_numerical(N, a)
    Q = CM_numerical(N, q)
    Sc = scaling_numerical(N, m)
    return Q@Sc@R

def DFRT(N, a):
    '''
    Computes the centered discrete fractional Fourier transform (DFRT) by computing the 
    matrix power of the centered DFT using the eigenvector decomposition into DHG modes.

    This DFRT is unitary and obeys index additivity.

    The following identites hold up to float precision for any integer k
    a == 4*k   (identity operator)
    a == 1+4*k (DFT)
    a == 2+4*k (parity operator)
    a == 3+4*k (inverse DFT)
    '''
    Fa = nf.DFRT(N, a) * np.exp(1j*np.pi/4*a)
    return Fa

def FRT_(N, a=1):
    '''
    This implementation of the FRT is only valid for fractional order
    a \in [-1,1].
    
    This is sufficient and can be extended to all fractional orders by
    using a combination of wrapping around fourth orders, using FT 
    offset, and parity offset.
    
    The planewave LCT phase is taken out of here, because it's more
    convenient to add it back in after extending it to all fractional
    orders.
    '''
    def CM_kernel(N, c):
        n = (np.arange(N) - (N-1)/2)/np.sqrt(N)
        d = np.exp(1j*np.pi * c * n**2)
        return np.diag(d)

    Na = N
    if Na % 2 == 0:
        Nb = 2*Na
    else:
        Nb = 2*Na - 1
    
    Nzf = (Nb - Na)/2
    Nz = int(Nzf)
    
    Zp = pylops.Pad(Na, (Nz, Nz)).todense()
    Fa = nf.xft_kernel(Na)
    Fb = nf.xft_kernel(Nb)
    iFb = np.conj(Fb).T
    
    Ksu = iFb@Zp@Fa
    Ksd = Ksu.T
    
    # print(Ksu.shape)

    phi = a*np.pi/2
    tan_phi = np.tan(phi/2)
    sin_phi = np.sin(phi)
    
    s = Nb/Na # s == 2 if Na even and very close to 2 if Na odd
    
    # print(Nb, Na, s)

    Q1 = CM_kernel(Nb, tan_phi/s)
    Q2 = CM_kernel(Nb, sin_phi*s)
    # Q3 = CM_kernel(Nb, tan_phi/s)
    Q3 = Q1
    
    sclr = np.exp(-1j*a*np.pi/4)
    # print(a)
    
    R = Ksd@Q3@iFb@Q2@Fb@Q1@Ksu * sclr
    
    return R

def FRT(N, a=1, phase='LCT'):
    def sign_mod(x, a):
        x = np.asarray(x)
        s = (x > 0).astype(int)*2 - 1
        y = np.abs(x) % a
        y *= s
        return y
    
    a = sign_mod(a, 4)
    
    if phase == 'LCT':
        sclr = np.exp(1j*a*np.pi/4)
    else:
        sclr = 1
    
    if a >= 2:
        Po = pylops.Flip(N)
        a = a - 2
    elif a <= -2:
        Po = pylops.Flip(N)
        a = a + 2
    else:
        Po = pylops.Identity(N)
        
    if a >= 1:
        Fo = np.conj(nf.xft_kernel(N)).T
        a = a - 1
    elif a <= -1:
        Fo = nf.xft_kernel(N)
        a = a + 1
    else:
        Fo = pylops.Identity(N)
        
    R = FRT_(N, a) * sclr
    
    return Po@Fo@R

def pylops_scaling(N, s=1, kind='sinc'):
    ix1 = np.linspace(-1, 1, N)
    ix2 = (ix1/s + 1)/2 * (N-1)
    Op,_ = pylops.signalprocessing.Interp(N, ix2, kind='sinc')
    return (Op).todense() / np.sqrt(s)

def LCT_iwasawa(xs, M_abcd, lam=1064e-9):
    A,B,C,D = M_abcd.ravel()
    N = len(xs)
    xs_scale = hyperdifferential_xs_scale(xs, lam=lam)
    s = of.abcd.scaling(np.sqrt(xs_scale))
    si = of.abcd.scaling(1/np.sqrt(xs_scale))
    M2 = s@np.array([[A,B],[C,D]])@si
    q,m,a = of.abcd.iwasawa_decomp(M2, return_matrices=False)
    R = DFRT(N, a)
    Sc = scaling_numerical(N, m)
    Q = CM_numerical(N, q)
    LCT = Q@Sc@R
    return LCT

def LCT_iwasawa2(x1s, x2s, M_abcd, lam=1064e-9, return_suboperators=False):
    N1 = len(x1s)
    N2 = len(x2s)
    if N1 != N2:
        raise ValueError('input and output grid must have the same number of points')
    N = N1
        
    s1 = np.ptp(x1s)
    s2 = np.ptp(x2s)
    scl = s2/s1
    M_scl = of.abcd.scaling(1/scl)
    
    A,B,C,D = M_abcd.ravel()
    xs_scale = hyperdifferential_xs_scale(x1s, lam=lam)
    s = of.abcd.scaling(np.sqrt(xs_scale))
    si = of.abcd.scaling(1/np.sqrt(xs_scale))
    M2 = s@np.array([[A,B],[C,D]])@si
    M3 = M_scl@M2
    q,m,a = of.abcd.iwasawa_decomp(M3, return_matrices=False)
    R = DFRT(N, a)
    Sc = scaling_numerical(N, m)
    Q = CM_numerical(N, q)

    if return_suboperators:
        LCT = Q, Sc * np.sqrt(1/scl), R
    else:
        LCT = Q@Sc@R * np.sqrt(1/scl)
    
    return LCT

def LCT_iwasawa3(x1s, x2s, M_abcd, lam=1064e-9, return_suboperators=False):
    '''The most accurate of the Iwasawa kernels
    '''
    N1 = len(x1s)
    N2 = len(x2s)
    if N1 != N2:
        raise ValueError('input and output grid must have the same number of points')
    N = N1
    
    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]
        
    scl = dx2/dx1
    M_scl = of.abcd.scaling(1/scl)
    
    A,B,C,D = M_abcd.ravel()
    xs_scale = hyperdifferential_xs_scale(x1s, lam=lam)
    s = of.abcd.scaling(np.sqrt(xs_scale))
    si = of.abcd.scaling(1/np.sqrt(xs_scale))
    M2 = s@np.array([[A,B],[C,D]])@si
    M3 = M_scl@M2
    q,m,a = of.abcd.iwasawa_decomp(M3, return_matrices=False)
    # print(q,m,a)
    R = FRT(N, a)
    Sc = pylops_scaling(N, m)
    Q = CM_kernel_pei(N, -q)

    if return_suboperators:
        LCT = Q, Sc * np.sqrt(1/scl), R
    else:
        LCT = Q@Sc@R * np.sqrt(1/scl)
    
    return LCT

def LCT_cav_scan_1D(D_rt, u_inc, r, phi=0):
    N = len(u_inc)
    g = np.exp(-1j*phi/90*np.pi)
    I = np.eye(N)
    u_circ = np.linalg.solve(I-r*g*D_rt, u_inc)
    return u_circ

def LCT_cav_scan_1D_eig(D_rt, u_inc, r=0.999, phi=[0]):
    phi = np.array(phi)
    g = np.exp(-1j*phi/90*np.pi)

    eh,ev = np.linalg.eig(D_rt)
    evi = np.linalg.inv(ev)

    eh *= np.conj(nf.phase(eh[0]))
    
    ec = evi@u_inc
    ec2 = ec[:,None]/(1-r*np.outer(eh,g))
    u_out = ev@ec2
    
    return u_out

def LCT_cav_scan_2D_separable_sylvester(D_rt_x, D_rt_y_inv, U_inc, r=0.999, phis=[0], debug=1):
    phis = np.array(phis)
    K = np.size(phis)
    N,M = np.shape(U_inc)
    out = np.zeros([K,N,M], dtype=np.complex128)
    
    for k, phi in enumerate(phis):
        if debug > 0:
            inplace_print(k)
        g = np.exp(-1j*phi/90*np.pi)
        A = 1/np.sqrt(r*g)*D_rt_y_inv
        B = -np.sqrt(r*g)*D_rt_x.T
        Q = A@U_inc
        X = scipy.linalg.solve_sylvester(A, B, Q)
        out[k,:,:] = X
        
    return out

def compute_operator_eigenmodes(D_rt, zero_fundamental_phase=True, sortby='hamiltonian'):
    N,N2 = np.shape(D_rt)
    assert(N==N2)
    
    eh,ev = np.linalg.eig(D_rt)
    
    if sortby == 'hamiltonian':
        dhg = nf.DHG_modes(N)
        dhg_hamiltonian_eigvals = np.diag(np.sqrt(np.arange(N)+1))

        ev_hamiltonian_eigvals = np.sum(dhg_hamiltonian_eigvals@np.abs(dhg@ev)**2,axis=0)
        ii = np.argsort(ev_hamiltonian_eigvals)

        eh = eh[ii]
        ev = ev[:,ii]
    
    if zero_fundamental_phase:
        eh *= np.conj(nf.phase(eh[0]))
    
    return eh,ev

def LCT_cav_scan_2D_separable_eig(D_rt_x=None, D_rt_y=None, U_inc=None, phis=[0], r=0.999, return_powers=False, zero_fundamental_phase=True, debug=0):
    ehy, evy = compute_operator_eigenmodes(D_rt_y, zero_fundamental_phase=zero_fundamental_phase)
    ehx, evx = compute_operator_eigenmodes(D_rt_x, zero_fundamental_phase=zero_fundamental_phase)
    evxi = np.linalg.inv(evx)
    evyi = np.linalg.inv(evy)

    phis = np.asarray(phis)
    N,M = np.shape(U_inc)

    e_inc = evyi@U_inc@evxi.T

    ehc = np.reshape(np.kron(ehy, ehx), (N, M))
    g = np.exp(-1j*phis/90*np.pi)
    circ_e = e_inc[:,:,None]/(1-r*np.multiply.outer(ehc, g))
    
#     to convert back to original coordinates
    circ_u = evx@np.swapaxes(evy@circ_e, 0, 1)
    out = circ_u
        
    if return_powers:
        out = of.compute_power(out, sum_axes=(0,1))
        
    return out

def LCT_2D_eig_decomp(f_rt, shape=None, k=50, sortby='hamiltonian', eigs_kwargs={}):
    if shape is None:
        shape = f_rt.shape
    N,M = shape
    L = scipy.sparse.linalg.LinearOperator(matvec=f_rt, shape=[N*M, N*M])
    eh, ev = scipy.sparse.linalg.eigs(L, k=k, **eigs_kwargs)
    
    # default sort by eigval amplitude
    if sortby is None or sortby == 'eigval_abs':
        sk = np.argsort(np.abs(eh))[::-1]
        
    elif sortby == 'hamiltonian':
        em = np.reshape(ev, (N, M, k))
        ny = np.arange(N) + N
        nx = np.arange(M) + M
        dhg = nf.DHG_modes(N)
        dem = dhg@np.swapaxes(dhg@em,0,1)
        dpem = np.abs(dem)**2
        dpehx = np.sum(nx[:,None,None]*dpem, axis=(0,1))
        dpehy = np.sum(ny[None,:,None]*dpem, axis=(0,1))
        dpeh = np.sqrt(dpehx**2+dpehy**2)
        sk = np.argsort(dpeh)
        
    eh = eh[sk]
    ev = ev[:, sk]
    
    return eh,ev

def hybrid_fox_li(f_rt, shape=None, u_init=None, Nki=200, Nk=100, Nkf=10, debug=1, tol=1e-9, solver=scipy.sparse.linalg.gmres):
    '''
    Finds the lowest order cavity eigenmode using an alternating method of Fox-Li and a sparse solver.
    Generally converges very fast.
    
    Example f_rt
    
    def f_rt(v):
        n = np.size(v)
        N = int(np.sqrt(n))
        X = np.reshape(v, [N, N])
        X = B1y@X@B1x.T
        X = X*circ_aperture
        X = B2y@X@B2x.T
        X = X*circ_aperture
        return np.ravel(X)
    '''
    if shape is None:
        shape = f_rt.shape

    N,M = shape
    
    if u_init is None:
        u_init = np.ones(N*M)
    
    def fox_li_iter(f_rt, un, Nk=100, debug=1):
        ut = un
        for i in range(Nk):
            ut = un
            if debug >=1:
                inplace_print(i)
            un = f_rt(ut)
        return un, ut

    def sparse_iter(f_rt, un, solver=solver, debug=1):
        out = LCT_cav_scan_2D(f_rt, U_inc=np.reshape(un,[N,M]), phis=[0], r=1, solver=solver, debug=debug, zero_fundamental_phase=False)
        out = out[0,:]
        out_p = of.compute_power(out)
        out /= np.sqrt(out_p)
        return out
    
    un, _ = fox_li_iter(f_rt, u_init, Nk=Nki, debug=debug)
    
    norm0 = np.inf
    converged = False
    while not converged:
        un2, un1 = fox_li_iter(f_rt, un, Nk=Nk, debug=debug)
        un3 = sparse_iter(f_rt, un2, debug=debug)
        norm1 = np.sum(np.abs(un-un3)**2)
        if debug >= 1:
            inplace_print(np.abs(norm1 - norm0))
        if np.abs(norm1 - norm0) < tol:
            converged = True
        un = un3
        norm0 = norm1
    
    u1, u0 = fox_li_iter(f_rt, un, Nk=Nkf, debug=debug)
    
    # g0 is the eigenvalue of the fox-li eigenmode
    g0 = u1@np.conj(u0) / (u0@np.conj(u0)) # Siegman eq 14.15
    return g0, u1

def LCT_cav_solve_2D(f_rt, U_inc, r=0.999, solver=scipy.sparse.linalg.gmres, solver_kwargs={}):
    '''
    Solve circulating field for a single point.
    '''
    def f_cav(v,r):
        x = f_rt(v)
        return v - x*r

    f2 = lambda v: f_cav(v, r)
    N,M = np.shape(U_inc)
    L_linop = scipy.sparse.linalg.LinearOperator(matvec=f2, shape=[N*M,N*M])
    soln, _ = solver(L_linop, np.ravel(U_inc), **solver_kwargs)
    return soln

def LCT_cav_scan_2D(f_rt, U_inc, r=0.999, phis=[0], debug=1, zero_fundamental_phase=True, solver=scipy.sparse.linalg.gmres, solver_kwargs={}):
    '''
    Adding planewave preconditioning does bugger all and breaks the sparse iter in hybrid fox li

    example f_rt
    
    def f_rt(v):
        n = np.size(v)
        N = int(np.sqrt(n))
        X = np.reshape(v, [N, N])
        X = B1y@X@B1x.T
        X = X*circ_aperture
        X = B2y@X@B2x.T
        X = X*circ_aperture
        return np.ravel(X)
    '''

    def f_cav(v,g,r):
        x = f_rt(v)
        return v - x*g*r
    
    phis = np.asarray(phis)
    gs = np.exp(-1j*phis/90*np.pi)
    
    N,M = np.shape(U_inc)

    g0 = 1
    if zero_fundamental_phase:
        g0, u0 = hybrid_fox_li(f_rt, (N,M))

    # def g_op(v,g,r):
    #     v2 = 1/(1 - g*r) * v
    #     return v2
    
    out = []
    codes = []
    x0 = U_inc.ravel()
    for i,g in enumerate(gs):
        if debug >= 1:
            inplace_print(i)
        f2 = lambda v: f_cav(v, g, r*np.conj(nf.phase(g0)))
        # g2 = lambda v: g_op(v, g, r*np.conj(nf.phase(g0)))
        L_linop = scipy.sparse.linalg.LinearOperator(matvec=f2, shape=[N*M,N*M])
        # M_linop = scipy.sparse.linalg.LinearOperator(matvec=g2, shape=[N*M,N*M])
        soln = solver(L_linop, np.ravel(U_inc), x0=x0, **solver_kwargs)
        x0 = soln[0]
        out.append(soln[0])
        codes.append(soln[1])
    
    out = np.asarray(out)
    
    if debug >= 2:
        return out, codes
    else:
        return out

def LCT_cav_scan_2D_eig(f_rt, U_inc, phis=[0], r=0.999, k=50, zero_fundamental_phase=True):
    N,M = np.shape(U_inc)
    phis = np.asarray(phis)
    print(k)
    eh, ev = LCT_2D_eig_decomp(f_rt, shape=(N, M), k=k)
    evi = np.linalg.pinv(ev)
    
    if zero_fundamental_phase:
        eh *= np.conj(nf.phase(eh[0]))
    
    g = np.exp(-1j*phis/90*np.pi)
    e_inc = evi@np.ravel(U_inc)
    circ_e = e_inc[:,None]/(1-r*np.multiply.outer(eh, g))
    
    out = ev@circ_e
    return out