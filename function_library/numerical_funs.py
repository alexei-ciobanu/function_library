import functools
import random
from itertools import chain

import numpy as np
import scipy
import scipy.linalg
import scipy.signal
import scipy.integrate

def fromiter_2d(iterable, dtype, shape=(-1,1)):
    '''numpy.fromiter() only works on 1D iterables (ones that return a number 
    each iteration). This one works on iterables that return another iterable of
    numbers (the sub-iterables are assumed to be of constant length).
    Default shape consumes entire iterator, resizing the array each step. Very
    inefficient

    Example:
    d = {(i,j):k for i,j,k in np.random.randn(5,3)}
    a = nf.fromiter_2d(d.keys(), dtype=float, shape=(5,2))
    b = np.fromiter(d.values(), dtype=float, count=5)
    '''
    n,m = shape
    a = np.fromiter(chain.from_iterable(iterable), dtype=dtype, count=n*m)
    if n != -1:
        a.shape = shape
    return a

def eye_like(x):
    shape = x.shape
    N = shape[0]
    if not np.all(np.array(shape) == shape[0]):
        raise Exception
    return np.eye(N)

def minmax(x):
    '''
    Why does this not exist in numpy?
    '''
    return np.min(x), np.max(x)

def phase(x):
    x = np.asarray(x)
    out = np.zeros_like(x)
    mask = np.abs(x) == 0
    out[mask] = 0
    out[~mask] = x[~mask]/np.abs(x[~mask])
    return out

def angle(a, *args, deg=False, **kwargs):
    '''
    numpy angle won't let me change where the discontinuity is.
    So I made my own that is discontinuous at 0/360 instead of 180/-180
    '''
    theta = np.angle(a, *args, deg=deg, **kwargs)
    theta = np.asarray(theta)
    if deg is False:
        theta[theta<0] += 2*np.pi
    elif deg is True:
        theta[theta<0] += 360
    return theta.item() if theta.ndim == 0 else theta

@functools.wraps(np.linspace)
def linspace(*args, **kwargs):
    xs = np.linspace(*args, **kwargs)
    dx = np.nan
    if len(xs) >=2:
        dx = xs[1] - xs[0]
    return np.array([xs, dx], dtype=object)

def linspace_like(xs, N=101):
    xl = xs[0]
    xu = xs[-1]
    return linspace(xl, xu, N)

def centered_linspace(c=0, w=1, N=21):
    '''
    Alternate constructor for a linspace using a center and width.
    '''
    lb, ub = c-w, c+w
    return linspace(lb, ub, N)

def symlog(x, p=1, norm='max'):
    '''
    Symmetric log. Useful for compressing dynamic range of data for plotting 
    without losing sign information. Works on complex numbers without distorting
    the phase, it's purely an amplitude transformation.
    p : (0 > p > inf) symlog corner. Smaller numbers have a bigger dynamic range 
    compression. 
    As p tends to infinity symlog(x) tends to x.
    As p tends to 0 symlog(x) tends to log(x) + offset

    -----------------------------------
    import numpy as np
    import matplotlib.pyplot as plt
    from numerical_funs import symlog

    r = np.linspace(0, 1, 301)
    x = r * np.exp(1j*r*2*np.pi)
    y = nf.symlog(x, 0.01)

    plt.plot(x.real, x.imag, label='x')
    plt.plot(y.real, y.imag, label='symlog(x)')
    plt.axis('equal')
    plt.legend()
    '''
    x = np.asarray(x, dtype=complex)
    y = np.zeros_like(x)
    x_phase = phase(x)
    y = np.log(np.abs(x) + p) - np.log(p)
    y = y * x_phase
    if norm=='max':
        y_max = np.max(np.abs(y))
        x_max = np.max(np.abs(x))
        y = y/y_max*x_max
    return y

def rwd(a,b):
    'relative weighted difference'
    return (a-b)/(np.abs(a+b)/2)

def rwad(a,b):
    'relative weighted absolute difference'
    return np.abs(a-b)/(np.abs(a+b)/2)

def rwsd(a,b):
    'relative weighted squared difference'
    return rwad(a,b)**2

def find_nearest_ind(array, value):
    array = np.asarray(array)
    idx = np.nanargmin(np.abs(array - value))
    return idx

def diag_indices(N, k=0):
    M = N - np.abs(k)
    if k >= 0:
        # upper diag
        ix = np.arange(k, N)
        iy = np.arange(0, M)
    elif k < 0:
        # lower diag
        ix = np.arange(0, M)
        iy = np.arange(-k, N)
    return ix,iy
        
def diag_indices_circulant(N, k=0):
    ix = np.mod(np.arange(N) + k, N)
    iy = np.arange(N)
    return ix,iy

def assign_diag(M, v, k=0, circulant=False):
    N = len(M)
    if circulant:
        M[diag_indices_circulant(N, k)] = v
    else:
        M[diag_indices(N, k)] = v
    return M

def D_bilinear(N):
    B = np.zeros([N,N])
    A = np.zeros([N,N])

    assign_diag(B, 1, 0, circulant=True)
    assign_diag(B, 1, 1, circulant=True)
    
    assign_diag(A, 1, 0, circulant=True)
    assign_diag(A, -1, 1,circulant=True)
    
    D = np.linalg.solve(B,A)*np.sqrt(N)/np.pi*1j
    return D

def U_bilinear(N):
    D = D_bilinear(N)
    F = centered_dft(N)
    iF = np.conj(F).T
    U = F@D@iF
    return U

def dft_scaling_commutator(N, dx=1, lam=1):
    '''Scaling factor that turns the continous Fourier transfrom into the DFT'''
    s = N*dx**2/lam
    return s

def dft_xs_scaling(N, dx=1, lam=1):
    '''Scaling factor that turns the continous Fourier transfrom into the DFT'''
    s = 1/np.abs(dft_scaling_commutator(N, dx=dx, lam=lam))
    return s

def dft_kernel(N, centered=True, balanced_norm=True):
    '''
    The standard DFT/FFT kernel. The centering is done by rolling the array with np.fft.fftshift.
    Note that for even N the kernel can't be centered exactly. If exact centering is required use 
    centered_dft and xft.
    '''
    # F = scipy.linalg.dft(N)
    F = np.fft.fft(np.eye(N))
    if centered:
        F = np.fft.fftshift(F)
    if balanced_norm:
        F = F/np.sqrt(N)
    return F

def fft(x, sign_convention='optics'):
    N = np.shape(x)[0]
    if sign_convention == 'optics':
        out = np.fft.ifft(x) * np.sqrt(N)
    elif sign_convention in ['signal', 'engineering', 'math']:
        out = np.fft.fft(x) / np.sqrt(N)
    return out

def ifft(x, sign_convention='optics'):
    N = np.shape(x)[0]
    if sign_convention == 'optics':
        out = np.fft.fft(x) / np.sqrt(N)
    elif sign_convention in ['signal', 'engineering', 'math']:
        out = np.fft.ifft(x) * np.sqrt(N)
    return out

def fft_kernel(N, sign_convention='optics'):
    '''
    Mostly for debugging
    '''
    return fft(np.eye(N), sign_convention=sign_convention)

def ifft_kernel(N, sign_convention='optics'):
    '''
    Mostly for debugging
    '''
    return ifft(np.eye(N), sign_convention=sign_convention)

def dft_kernel2(Nx, Ny):
    ny = (np.arange(Ny)-Ny//2)/np.sqrt(Ny)
    nx = (np.arange(Nx)-Nx//2)/np.sqrt(Nx)
    arg = np.outer(ny,nx)
    F = np.exp(-1j*2*np.pi*arg)
    return F

def ft_kernel(xs, b=1):
    arg = np.outer(xs,xs)
    dx = xs[1] - xs[0]
    F = np.exp(-1j*2*np.pi*b*arg) * dx
    return F

def centered_dft(N, b=1, norm=True, sign_convention='optics'):
    '''
    Centered version of the DFT kernel by symmetrically sampling the normalized frequency vector.
    '''
    if sign_convention == 'optics':
        fsgn = 1
    elif sign_convention in ['signal', 'engineering', 'math']:
        fsgn = -1
    n = np.arange(N)-(N-1)/2
    arg = np.outer(n,n)/N
    F = np.exp(fsgn*1j*2*np.pi*b*arg)
    if norm:
        F /= np.sqrt(N)
    return F

def centered_dft2(N, b=1, norm=True, sign_convention='optics'):
    '''
    Centered around nyquist frequency
    '''
    n = np.arange(N)
    arg = np.outer(n,n)/N
    F = np.exp(-1j*2*np.pi*b*arg)
    if norm:
        F /= np.sqrt(N)
    return F

def xft(x, axis=0, norm=True):
    '''A fast way to compute the centered DFT by using the FFT and the Fourier shift 
    theorem to center the FFT kernel.
    '''
    # parameters for expanding the Fourier shift mask for broadcasting
    # to work when taking 1D FFT of N-D arrays
    x_shape = np.shape(x)
    n_dim = len(x_shape)
    N = x_shape[axis]
    new_dims = np.arange(n_dim-1)
    new_dims[axis:] += 1

    # Fourier shift mask
    n = np.arange(N)
    a0 = np.exp(-1j*np.pi*(N-1)**2/2/N)
    S = np.exp(1j*np.pi*(N-1)*n/N)
    S = np.expand_dims(S, new_dims.tolist())

    # compute the XFT
    X = a0*S*np.fft.fft(S*x, axis=axis)
    if norm:
        X /= np.sqrt(N)
    return X

def ixft(x, norm=True):
    '''
    Inverse of xft.
    '''
    N = len(x)
    n = np.arange(N)
    a0i = np.exp(1j*np.pi*(N-1)**2/2/N)
    Si = np.exp(-1j*np.pi*(N-1)*n/N)
    X = a0i*Si*np.fft.ifft(Si*x)
    if norm:
        X *= np.sqrt(N)
    return X
    
def xft_kernel(N, norm=True):
    '''
    Computes the centered DFT kernel by applying the Fourier shift theorem.
    '''
    return xft(np.eye(N), norm=norm)

def ixft_kernel(N, norm=True):
    '''
    Inverse of xft_kernel by using the fact that the inverse of the Fourier kernel is just
    the complex conjugate.
    '''
    return ixft(np.eye(N), norm=norm)

def xft2(x, norm=True):
    '''
    A fast way to compute the centered DFT by using the FFT and the Fourier shift 
    theorem to center the FFT kernel.
    '''
    x = np.asarray(x)
    M,N = np.shape(x)
    ns = np.arange(N)
    ms = np.arange(M)
    a0n = np.exp(-1j*np.pi*(N-1)**2/2/N)
    a0m = np.exp(-1j*np.pi*(M-1)**2/2/M)
    Sn = np.exp(1j*np.pi*(N-1)*ns/N)[None, :]
    Sm = np.exp(1j*np.pi*(M-1)*ms/M)[:, None]
    X = a0n*a0m*Sm*Sn*np.fft.fft2(Sm*Sn*x)
    if norm:
        X /= np.sqrt(N*M)
    return X

def ixft2(x, norm=True):
    '''
    A fast way to compute the centered DFT by using the FFT and the Fourier shift 
    theorem to center the FFT kernel.
    '''
    x = np.asarray(x)
    M,N = np.shape(x)
    ns = np.arange(N)
    ms = np.arange(M)
    a0n = np.exp(1j*np.pi*(N-1)**2/2/N)
    a0m = np.exp(1j*np.pi*(M-1)**2/2/M)
    Sn = np.exp(-1j*np.pi*(N-1)*ns/N)[None, :]
    Sm = np.exp(-1j*np.pi*(M-1)*ms/M)[:, None]
    X = a0n*a0m*Sm*Sn*np.fft.ifft2(Sm*Sn*x)
    if norm:
        X *= np.sqrt(N*M)
    return X

def logF_kernel(N=None, F=None):
    '''
    Computes matrix logarithm of dft kernel using Aristidou (2007)
    https://doi.org/10.1155/2007/20682

    This method is preferred over scipy.linalg.logm(F) since the matrix logarithm
    is not unique for complex matrices (branch cut stuff).
    
    Can either specify the number of samples N or pass a custom DFT kernel e.g. (XFT kernel).
    '''
    if N is None and F is None:
        raise Exception('specify something')
    if N is None:
        N = len(F)
    if F is None:
        F = dft_kernel(N)
    I = np.eye(N)
    G1 = 1/4 * (I - 1j*F - F@F + 1j*F@F@F)
    G2 = 1/4 * (I - F + F@F - F@F@F)
    G3 = 1/4 * (I + 1j*F - F@F - 1j*F@F@F)
    logF = 1/2*1j*np.pi*G1 + 1j*np.pi*G2 - 1/2*1j*np.pi*G3
    return logF

def fourier_shift_1D(x, k=1):
    '''
    Perform a periodic sample shift on an array x using the Fourier shift theorem

    k: amount to shift in units of samples (can be noninteger)
    '''
    N = np.size(x)
    n = np.arange(N)
    shift = np.exp(-1j*k*n/N*np.pi*2)
    
    return np.fft.ifft(np.fft.fft(x)*shift)

def fourier_shift_2D(z, kx=1, ky=1):
    Ny, Nx = np.shape(z)
    ny = np.fft.fftshift(np.arange(Ny) - Ny/2)//1
    nx = np.fft.fftshift(np.arange(Nx) - Nx/2)//1
        
    shifty = np.exp(-1j*ky*ny/Ny*np.pi*2)
    shiftx = np.exp(-1j*kx*nx/Nx*np.pi*2)
    shift = np.outer(shifty,shiftx)
    
    return np.fft.ifft2(np.fft.fft2(z)*shift)

def wigner(ys, use_hilbert=True, verbose=False):
    '''
    Compute wigner distribution function (WDF) of a time series array ys.
    
    use_hilbert: flag for whether to compute the WDF using the analytic 
    signal (Hilbert transform) of a real signal ys
    
    Where on earth did I get this from?
    It's actually really good.
    '''
    ys = np.array(ys)
    assert(np.ndim(ys) == 1)
    if use_hilbert:
        try:
            ys = scipy.signal.hilbert(ys)
        except ValueError:
            if verbose:
                print('Hilbert transform skipped because ys is complex')
    
    N = ys.size
    F = centered_dft(N)
    iF = np.conj(F).T

    F2 = centered_dft(N, b=0.5)
    iF2 = np.conj(F2).T
    
    # slower (but more clear)
    Fy = np.diag(F@ys)
    Y1 = F2 @ Fy @ iF # scaled fourier dual
    Y2 = np.flipud(np.conj(Y1)) # parity conjugate of Y1
    W = F@(Y1*Y2)

    # # faster (but less clear)
    # Fy = F@ys
    # Y1 = (Fy[None,:] * F2)@iF
    # Y2 = (Fy[None,:] * iF2)@iF
    # W = F@(np.conj(Y1)*Y2)
    
    return np.real(W)

def centered_affine_transform(im, mat, order=3):
    '''Performs the scipy.ndimage.affine_transform with respect to
    the center of the image.

    Uses the forward affine matrix instead of the inverse that ndimage uses.
    '''
    assert(np.ndim(im) == 2)
    imat = np.linalg.inv(mat)
    center = -0.5*np.array(im.shape)
    offset = imat@(center-mat@center)
    im2 = scipy.ndimage.affine_transform(im, imat, offset=offset, order=order)
    return im2

def wigner2(x, use_analytic_signal=True, order=3):
    '''Computes the Wigner-Ville distribution in 3 steps
    This produces similar results to pytftb's tftb.processing.WignerVilleDistribution
    And was derived by reverse engineering it.
    
    1. Compute outer product of input and its reversed complex conjugate
    2. Apply an affine transformation of a -45 degree rotation and scaling by 1/sqrt(2)
    3. Compute a centered fourier transform (e.g. xft) along the first axis
    '''
    assert(np.ndim(x) == 1)
    if use_analytic_signal and not np.iscomplexobj(x):
        x = scipy.signal.hilbert(x)
    A = np.outer(np.flipud(np.conj(x)),x)

#     # old method with two separate affine transformations
#     Ar = scipy.ndimage.rotate(A, -45, order=order)
#     Ar = scipy.ndimage.interpolation.zoom(Ar, 1/np.sqrt(2), order=order)

    theta = -np.pi/4
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    scl = np.eye(2)/np.sqrt(2)
    mat = scl@rot
    Ar = centered_affine_transform(A, mat, order=order)
    W = xft(Ar, axis=0)
    return np.real(W)

def spwvd(x, sigma_f=None, sigma_t=None):
    '''smoothed psuedo wigner ville distribution
    
    Smoothing is done in both time and frequency domains in accordance with the 
    uncertainty principle to ensure that probabilities are strictly positive.
    
    The remaining negative probabilities are either due to floating point or they are at 
    around 0 Hz, which I still don't have an explanation for
    
    TODO:
    * why are there still negative probabilities around 0 Hz
    * proper rescaling filter (e.g. linear interpolation)
    * variable time-frequency resolution (eg. 1024 time samples and 256 freq samples)
    '''
    N = x.size
    
    if sigma_f is None:
        sigma_f = np.sqrt(2/np.pi) * np.sqrt(1/N)
        
    if sigma_t is None:
        sigma_t = np.sqrt(2/np.pi) * np.sqrt(1/N)
    
    Gf = gef.gaussian_wn(N, sigma=sigma_f)
    Gt = gef.gaussian_wn(N, sigma=sigma_t)
    F = nf.centered_dft(N)
    iF = np.conj(F).T
    S_F = nf.centered_dft(N, b=0.5) # this should be an FFT with a rescaling filter
    
    Fy = scipy.sparse.diags(F@x) # makes it slightly faster
    Y1 = S_F @ Fy @ iF # scaled fourier dual
    Y2 = np.flipud(np.conj(Y1)) # parity conjugate of Y1

    A = (Y1*Y2)@iF # Ambiguity function 
    Ag = (A * Gf[:,None]) * Gt[None,:] # apply time-frequency filter
    W = F@Ag@F # rotate to wigner domain
    
    return np.real(W)

def DHG_modes(N, S=None):
    '''
    Computes the complete orthonormal set of discrete Hermite-Gaussian modes for R^N.
    Based on Otzakas (2001). I've modified it to work with the centered DFT instead of 
    the standard one. This removes a lot of the modulo 2 arithmetic from the original 
    implementation as well as a sign ambiguity on the final mode when N was even.
    
    Each DHG mode is an eigenvector of the N-point centered DFT matrix.
    
    Can be used to compute the discrete fractional Fourier transform (DFRT).
    
    The overall plus/minus sign of each individual DHG mode appears to be arbitrary
    '''
    
    def P_matrix(N):
        '''
        A matrix that splits up vectors into even components (symmetric around central index) on the left 
        and odd components (antisymmetric around central index) on the right
        '''
        P = np.eye(N)
        idx, idy = np.array(np.diag_indices_from(P))
        # create cross diagonal indices
        idy2 = idy[::-1]
        idx2 = idx
        P[idx2,idy2] = 1
        P[idx[N//2:], idy[N//2:]] = -1
        # if N is odd then central index is the special case
        if N%2 == 1:
            P[N//2,N//2] = np.sqrt(2)
        return P/np.sqrt(2)
    
    def B_matrix(N):
        '''
        Computes the fourier transform of the second derivative operator
        '''
        n = (np.arange(N) - (N-1)/2)
        # b = 2*(np.cos(np.pi*2*n/N) - 1)
        b = -4*np.pi**2*(n/N)**2
        # b = -np.cosh(n/np.sqrt(N)*4)
        B = np.diag(b)
        return B
    
    # Define matrices
    B = B_matrix(N)
    F = centered_dft(N)
    iF = np.conj(F).T
    D2 = F@B@iF
    if S is None:
        S = B + D2
    P = P_matrix(N)

    # return D2
    
    PSP = P@S@P
    Nk = int((N+1)/2)
    # split up the PSP matrix into "even" and "odd" submatrices
    Ev = PSP[:Nk,:Nk]
    Ov = PSP[Nk:,Nk:]
    
    # compute eigenvectors of the Ev submatrix
    eh,ev = np.linalg.eig(Ev)
    # sort by decreasing eigenvalue
    ehi = np.argsort(eh)[::-1]
    ev = ev[:,ehi]
    # zero-pad the eigenvector back to N
    epad = np.zeros([N-Nk,Nk])
    ev = np.vstack([ev,epad])

    # repeat for the Ov submatrix
    oh,ov = np.linalg.eig(Ov)
    ohi = np.argsort(oh)[::-1]
    ov = ov[:,ohi]
    opad = np.zeros([Nk,N-Nk])
    ov = np.vstack([opad,ov])
    
    # the DHG modes are given by "completing" the zero padded eignevectors 
    # with the P matrix
    us = []
    for i in range(N):
        if i%2 == 0 and i//2 < Nk:
            us.append(P@ev[:,i//2])
        elif i%2 == 1 and i//2 < N-Nk:
            us.append(P@ov[:,i//2])
    us = np.array(us)
    return us

def DHG_bilinear(N):
    D = D_bilinear(N)
    U = U_bilinear(N)
    S = D@D + U@U
    return DHG_modes(N,-S)

def DFRT(N, a, S=None):
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
    us = DHG_modes(N, S=S)
    Fa = np.zeros([N,N],dtype=np.complex128)
    for i in range(len(us)):
        Fa += np.outer(us[i],us[i])*np.exp(1j*np.pi/2*i*a)
    return Fa

def fourier_projectors(N):
    '''
    A set of 4 idemponent matrices derived from the DFT matrix that map DFT eigenvectors to a unique eigenvalue or the zero vector.

    Taken from Candan 2011
    '''
    F = centered_dft(N)
    F1 = F
    F0 = np.eye(N)
    F2 = F@F1
    F3 = F@F2    
    
    P1 = 1/4*(F3+F2+F1+F0)
    Pm1 = 1/4*(-F3+F2-F1+F0)
    Pj = 1/4*(1j*F3-F2-1j*F1+F0)
    Pmj = 1/4*(-1j*F3-F2+1j*F1+F0)

    return P1,Pm1,Pj,Pmj

def kong_eigenvector(N):
    '''Eigenvector of N-point DFT with eigenvalue 1 that approaches
    the HG_0 function as N goes to infinity.
    The eigenvector in a sense is unique in that it has the most consecutive zero 
    elements that is possible while remaining a DFT eigenvector.
    
    First discovered in (Kong 2008) in "Analytic Expressions of Two 
    Discrete Hermite-Gauss Signals"   
    https://ieeexplore.ieee.org/document/4400111/
    
    This function is only valid if (N-1)/4 is an integer i.e.
    N = {5,9,13,17,21,25,29,...}
    '''
    L = (N-1)/2
    K = L/2
    if not np.isclose(K%1,0):
        raise ValueError(f'Invalid N. (N-1)/4 must be an integer')
    
    n = np.arange(N)-(N-1)/2

    s = np.arange(K+1, L+1)
    a = 0.5
    B = np.subtract.outer(np.cos(2*np.pi/N*n),np.cos(2*np.pi/N*s))
    u = np.product(B,axis=1)
    u /= np.sum(np.abs(u))
    return u

def kong_eigenvector_zpk(N):
    '''zpk representation of the first Kong eigenvector.

    Remarkably the Kong eigenvector has a simple representation in zpk
    where the zeros are all the N'th roots of unity with negative real part.
    There are half as many poles as zeros which are all stacked at x=0
    i.e. in the center of the unit circle.

    First discovered in (Kong 2008) in "Analytic Expressions of Two 
    Discrete Hermite-Gauss Signals"   
    https://ieeexplore.ieee.org/document/4400111/
    '''
    L = (N-1)/2
    K = L/2
    if not np.isclose(K%1,0):
        raise ValueError(f'Invalid N. (N-1)/4 must be an integer')
    
    n = np.arange(N)-(N-1)/2
    e = np.exp(1j*np.pi*2*n/N)

    zeros = e[np.real(e)<=0]

    a = np.polynomial.polynomial.polyfromroots(zeros)[::-1]
    a /= np.sum(np.abs(a))

    b = np.zeros(len(z)//2+1)
    b[0] = 1

    poles = np.zeros(len(z)//2)
    b = np.polynomial.polynomial.polyfromroots(poles)[::-1]
    return zeros, poles

###############################################################################

def arbitrary_finite_difference_stencil(order, offsets):
    '''Doesn't work for many stencil points because the matrix
    calculation becomes unstable due to really high powers.
    
    Example
    --------
    offsets = [-4,-3,-2,-1,0,1,2]
    order = 2
    coeffs = arbitrary_finite_difference_stencil(offsets, order)
    '''
    offsets = np.asarray(offsets)
    N = len(offsets)
    rhs = np.zeros(N)
    rhs[order] = scipy.special.factorial(order)
    M = np.zeros([N, N])
    for i in range(N):
        M[i, :] = offsets**i # this is unstable for large i
    coeffs = np.linalg.solve(M, rhs)
    return coeffs

def central_finite_difference(M, N, k=None):
    '''
    Computes N-point order central finite difference coefficients of an M order derivative
    using a recursive algorithm published by Fornberg 1988. It is more numerically stable 
    for large M and N than the matrix inversion algorithm.
    
    If k is specified then it is used to specify the size of the finite difference stencil and 
    N is used for zero padding. User has to make sure if N is even then k should be even, similarly
    if N is odd k should be odd.
    '''
    def generate_alpha_vec(N):
        '''
        Generates indices for the N-point central difference coefficients.
        The order of the indices is significant for the algorithm.
        e.g. 
        if N=5
        alpha = [0,1,-1,2,-2]
        if N=4
        alpha = [1/2,-1/2,3/2,-3/2]
        '''
        alpha = []
        if N%2 == 1:
            k = (N+1)//2
            alpha.append(0)
            for i in range(1,k):
                alpha.extend([i,-i])
        elif N%2 == 0:
            k = N//2
            alpha = []
            for i in range(1,k+1):
                alpha.extend([i-1/2,-i+1/2])
        return alpha
    
    def permute_output_indices(delta_coeffs):
        '''
        Computes the indices to undo the ordering of the alpha vector to return the 
        finite difference coefficients in a more recognizable form.
        '''
        delta_coeffs = np.array(delta_coeffs)
        N = len(delta_coeffs)
        k = (N+1)//2 
        inds = []
        if N%2 == 0:
            for i in range(0,k):
                inds.append(N-1-i*2)
            for i in range(0,k):
                inds.append(i*2)
        if N%2 == 1:
            for i in range(0,k):
                inds.append(N-1-i*2)
            for i in range(0,k-1):
                inds.append(i*2+1)
        inds = np.array(inds)
        return inds

    delta = np.zeros([M+1, N, N])
    a = generate_alpha_vec(N)
    delta[0,0,0] = 1
    x0 = 0
    c1 = 1
    for n in range(1, N):
        c2 = 1
        for v in range(0, n):
            c3 = a[n] - a[v]
            c2 = c2 * c3
            for m in range(0, np.min([n,M])+1):
                delta[m,n,v] = ((a[n] - x0)*delta[m,n-1,v] - m*delta[m-1,n-1,v])/c3
    #             print(n,m,v, delta[m,n,v])
        for m in range(0, np.min([n,M])+1):
            delta[m,n,n] = c1/c2 * (m*delta[m-1,n-1,n-1] - (a[n-1] - x0)*delta[m,n-1,n-1])
        c1 = c2
    inds = permute_output_indices(delta[M,N-1])
    if k is None:
        return delta[M,N-1][inds]
    else:
        return delta[M,k-1][inds]

def jacobian(fun, x0, step_size=None):
    x0 = np.array(x0)
    if step_size is None:
        step_size = np.maximum(np.abs(x0) * 1e-7, 1e-8)
    N = len(x0)
    dx = step_size
    f0 = fun(x0)
    J = np.zeros(N)

    # compute first derivatives
    for i in range(N):
        xv = x0.copy()
        xv[i] += dx[i]
        J[i] = (fun(xv) - f0)/dx[i]

    return J

def hessian(fun, x0, step_size=None):
    '''
    Computes the second derivative matrix (Hessian) of function fun(x) at x=x0
    using the first order finite difference method.
    '''
    
    if step_size is None:
        step_size = np.maximum(np.abs(x0) * 1e-5, 1e-7)
    N = len(x0)
    dx = step_size
    f0 = fun(x0)
    H = np.zeros([N,N])
    
    # f1 = np.zeros(N)
    # # compute first order steps
    # for i in range(N):
    #     xv = x0.copy()
    #     xv[i] += dx[i]
    #     f1[i] = fun(xv)
    
    J = jacobian(fun, x0, step_size)

    # compute the second derivative matrix
    for i in range(0,N):
        for j in range(i,N):
            xv = x0.copy()
            xv[i] += dx[i]
            xv[j] += dx[j]
            f2 = fun(xv)

            # first order finite difference for second derivative
            # h = (f2 - f1[i] - f1[j] + f0)/(dx[i]*dx[j])
            # print(i,j,f2,f1[i],f1[j],f0)

            # using jacobian
            h = ((f2-f0) - J[i]*dx[i] - J[j]*dx[j])/(dx[i]*dx[j])
            # print(i,j,f2-f0,f1[i]*dx[i],f1[j]*dx[j])

            H[j,i] = h
            H[i,j] = h
            
    return H

def gradient_descent(f, x0, step_size=None, gtol=1e-9):
    x0 = np.array(x0)
    f0, j = jacobian(f, x0, return_f0=True)
    jstep = j*step_size
    jnorm = np.sqrt(np.sum(jstep)**2)

    xp = x0 + jstep
    fj = f0 + jnorm
    fp = f(xp)
    print(fp,fj)
    return fp, (fj-fp)/jnorm

def antialiased_function(fun, xs, *args, **kwargs):
    '''
    Computes fun(xs) but the fun is integrated between (x-dx/2, x+dx/2) for each x in xs.
    Can be used to obtain a more representative sampling of a function if it behaves too 
    wildly between nearing x points (particularly in the case of aliasing).
    '''
    out = []
    dx = xs[1] - xs[0]
    for i,x in enumerate(xs):
        out.append(quad_complex(fun, x-dx/2, x+dx/2)[0]/dx, *args, **kwargs)
    return np.array(out)

quad_wrapper = lambda x: functools.update_wrapper(x, scipy.integrate.quad)
dblquad_wrapper = lambda x: functools.update_wrapper(x, scipy.integrate.dblquad)

@quad_wrapper
def quad_complex(func, a, b, *args, **kwargs):
    '''A simple extension for scipy's quad to work for complex func'''
    fr = lambda x: np.real(func(x))
    fi = lambda x: np.imag(func(x))
    sol_r = scipy.integrate.quad(fr, a, b, *args, **kwargs)
    sol_i = scipy.integrate.quad(fi, a, b, *args, **kwargs)
    return sol_r[0] + 1j*sol_i[0], sol_r[1] + 1j*sol_i[1]

@dblquad_wrapper
def dblquad_complex(func, a, b, gfun, hfun, *args, **kwargs):
    '''A simple extension for scipy's dblquad to work for complex func'''
    fr = lambda x,y: np.real(func(x,y))
    fi = lambda x,y: np.imag(func(x,y))
    sol_r = scipy.integrate.dblquad(fr, a, b, gfun, hfun, *args, **kwargs)
    sol_i = scipy.integrate.dblquad(fi, a, b, gfun, hfun, *args, **kwargs)
    return sol_r[0] + 1j*sol_i[0], sol_r[1] + 1j*sol_i[1]


####################################################################################

def vec_trick(A, B):
    '''
    Uses the 'vec' function (aka np.ravel()) to compute the 
    Hadamard product of A and B using only matrix multiplication.
    This method results in a single nonseparable matrix-vector multiplication.
    '''
    n1,m1 = A.shape
    n2,m2 = B.shape
    D = np.diag(A.ravel())
    vec_C = D @ B.ravel()
    C = np.reshape(vec_C, (n1,m2))
    return C

def svd_trick(A, B):
    '''
    Uses the SVD to compute the Hadamard product of A and B using only 
    matrix multiplication. This method results in a sum of separable matrix products.
    '''
    U,s,V = np.linalg.svd(A)
    N = len(s)
    acc = 0
    for i in range(N):
        acc += s[i] * np.diag(U.T[i,:]) @ B @ np.diag(V[i,:])
    
    return acc

def complex_to_R2(z):
    zr = np.real(z)
    zi = np.imag(z)
    return np.squeeze(np.vstack([zr,zi]).T)

def construct_simplex(d_points):
    '''
    Constructs a N-simplex from a set of points. 
    The simplex with the largest volume chosen by Monete-Carlo.
    
    d_points: first index is sample number, second index is the dimension
    '''
    N, dim = d_points.shape
    max_vol = 0
    verts = []
    for i in range(200):
        si = np.array(random.sample(range(N), k=dim+1))
        s = d_points[si,:]
        h = scipy.spatial.ConvexHull(s)
        vol = h.volume
        if vol > max_vol:
            max_vol = vol
            verts = s
    return verts
