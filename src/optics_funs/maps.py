import numpy as np
import scipy

import general_funs as gef
import numerical_funs as nf
from . import CONSTANTS

def tilt_map_1D(xs, xbeta=0, lam=CONSTANTS['lambda']):
    lam = complex(lam)
    tx = np.exp(1j*4*np.pi*xs/lam*xbeta)
    return tx

def tilt_map_2D(xs, ys=None, xbeta=0, ybeta=0, lam=CONSTANTS['lambda']):
    if ys is None:
        ys = xs
    tx = tilt_map_1D(xs, xbeta)
    ty = tilt_map_1D(ys, ybeta)
    return np.multiply.outer(ty, tx)

def tophat_1D(xs, x0=0, w=1):
    N = len(xs)
    wn_fn = gef.gaussian_wn(N)
    dx = xs[1] - xs[0]
    fxsc = nf.dft_xs_scaling(N,dx)

    ft_shift = np.exp(-1j*xs*fxsc*x0*np.pi*2)

    norm = 2*w/(dx*np.sqrt(N))
    ys = wn_fn*np.sinc(xs*fxsc*2*w)*ft_shift*norm
    return np.real(nf.ixft(ys))

def tophat2_1D(xs, x0=0, w=1):
    return gef.rect(xs, mu=x0, w=w)

def rect_tophat_2D(xs, ys=None, x0=0, y0=0, wx=1, wy=None):
    if ys is None:
        ys = xs
    if wy is None:
        wy = wx
    rt = np.multiply.outer(tophat_1D(ys, y0, wy), tophat_1D(xs, x0, wx))
    return rt

def rect_tophat2_2D(xs, ys=None, x0=0, y0=0, wx=1, wy=None):
    if ys is None:
        ys = xs
    if wy is None:
        wy = wx
    rt = np.multiply.outer(tophat2_1D(ys, y0, wy), tophat2_1D(xs, x0, wx))
    return rt

def circ_tophat_2D(xs, ys=None, r=1, x0=0, y0=0):
    '''
    computes the analytical solution for the fft of circ tophat.
    Inverting the fft gives back the circ tophat.
    This is a way better way of defining a circ tophat than the 
    implicit equation since this one has infinite support for r,x0,y0.
    
    r : radius of top hat
    x0 : offset of tophat center to the right
    y0 : offset of tophat center up
    
    offsets can be made negative to go in the other direction
    '''

    if ys is None:
        ys = xs

    Nx = len(xs)
    Ny = len(ys)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    fxsc = nf.dft_xs_scaling(Nx,dx)
    fysc = nf.dft_xs_scaling(Ny,dy)
    wn_fn = gef.gaussian_wn2(Nx,Ny)

    rs = np.sqrt(np.add.outer((ys*fysc)**2, (xs*fxsc)**2))

    ft_xshift = np.exp(-1j*xs*fxsc*x0*np.pi*2)
    ft_yshift = np.exp(-1j*ys*fysc*y0*np.pi*2)
    ft_shift = np.multiply.outer(ft_yshift, ft_xshift)

    # the norm makes tophat height unity
    norm = np.pi*r**2/(dx*dy*np.sqrt(Nx*Ny))

    somb2 = gef.jinc(rs*np.pi*r*2)*ft_shift*norm
    
    return np.real(nf.ixft2(wn_fn*somb2))

def circ_tophat2_2D(xs, ys=None, r=1, x0=0, y0=0):
    '''
    computes the analytical solution for the fft of circ tophat.
    Inverting the fft gives back the circ tophat.
    This is a way better way of defining a circ tophat than the 
    implicit equation since this one has infinite support for r,x0,y0.
    
    r : radius of top hat
    x0 : offset of tophat center to the right
    y0 : offset of tophat center up
    
    offsets can be made negative to go in the other direction
    '''

    if ys is None:
        ys = xs

    Nx = len(xs)
    Ny = len(ys)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    rs = np.sqrt(np.add.outer(ys**2, xs**2))

    circ = np.ones([Ny,Nx])
    circ[ rs > r] = 0
    
    return circ

def point_absorber_1D(r, w, h=30e-2, zero_min=False):
    '''
    w is point absorber radius
    '''
    r = np.abs(np.asarray(r))
    c0 = -1/2 - np.log((h**2+np.sqrt(w**2+h**2))/w)
    
    b1 = np.abs(r) <= w
    b2 = np.abs(r) > w
    
    out = np.zeros_like(r)
    out[b1] = -1/2 * (r[b1]/w)**2
    out[b2] = c0 + np.log((h**2 + np.sqrt(r[b2]**2+h**2))/r[b2])
    
    if zero_min:
        out -= np.min(out)
    
    return out

def point_absorber_2D(xs, ys, w, h=30e-2, zero_min=False):
    '''
    w is point absorber radius
    '''
    r = np.sqrt(np.add.outer(ys**2, xs**2))
    c0 = -1/2 - np.log((h**2+np.sqrt(w**2+h**2))/w)
    
    b1 = np.abs(r) <= w
    b2 = np.abs(r) > w
    
    out = np.zeros_like(r)
    out[b1] = -1/2 * (r[b1]/w)**2
    out[b2] = c0 + np.log((h**2 + np.sqrt(r[b2]**2+h**2))/r[b2])
    
    if zero_min:
        out -= np.min(out)
    
    return out

def curvature_2D_quadratic(xs, ys=None, RoC_x=None, RoC_y=None, lam=1064e-9):
    '''
    Quadratic approximation to spherical surface.
    '''
    if ys is None:
        ys = xs
    if RoC_y is None:
        RoC_y = RoC_x
    cx = -1/(2*RoC_x)
    cy = -1/(2*RoC_y)
    h = np.add.outer(ys**2*cy, xs**2*cx) # the height map
    C = np.exp(-1j*np.pi/lam * 4 * h)
    return C

def curvature_2D_spherical(xs, ys=None, RoC=None, lam=1064e-9):
    if ys is None:
        ys = xs
    h = np.sqrt(RoC**2-np.add.outer(ys**2, xs**2)) - RoC
    C = np.exp(-1j*np.pi/lam * 4 * h)
    return C