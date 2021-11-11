import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as intgr
import scipy.linalg as linalg
import scipy.special
import itertools
import os
import pathlib
from pathlib import Path
import zipfile
import bz2
import gzip
import lzma
import pickle as pl
import sys
import functools
import collections

import numerical_funs as nf

import datetime
import time

home_dir = Path(os.path.expanduser("~"))
cwd = Path(os.path.realpath(__file__)).parent

from .strings import *

class SentinelValue():
    '''
    Make a unique object for comparison with the `is` statement.
    Useful if checking strings and you don't want to use a string as a 
    sentinel value.
    '''
    def __init__(self, name=None):
        # the name is only for printing
        self.__name = name

    def __repr__(self):
        if self.__name is None:
            name = ""
        else:
            name = f"\"{self.__name}\""
        return f"<SentinelValue({name}) at {hex(id(self))}>"

NOTPASSED = SentinelValue('NOTPASSED')
_getattr = getattr

def getattr(object, name, default=NOTPASSED):
    '''
    Allows for name to be a dotted string to be able to pull nested attributes
    '''
    name_split = name.split('.')
    ob = object
    for attr in name_split:
        if default is not NOTPASSED:
            ob = _getattr(ob, attr, default)
        else:
            ob = _getattr(ob, attr)
    return ob

def getitem(obj, x, default=NOTPASSED):
    '''
    Same as __getitem__ but admits a default value
    '''
    if default is NOTPASSED:
        out = obj.__getitem__(x)
    else:
        try:
            out = obj.__getitem__(x)
        except IndexError:
            out = default
    return out

def arithmetic_mean(x, axis=None):
    return np.mean(x, axis=axis)

def geometric_mean(x, axis=None):
    return np.exp(np.mean(np.log(x), axis=axis))

def harmonic_mean(x, axis=None):
    return np.size(x, axis=axis)/np.sum(1/x, axis=axis)

def passed_kwargs():
    """
    Stolen from https://stackoverflow.com/a/1409284/13161738
    
    Decorator that provides the wrapped function with an attribute 'passed_kwargs'
    containing just those keyword arguments actually passed in to the function.

    Useful for creating a more friendly function interfaces.
    
    Example:
    @passed_kwargs()
    def test(a=1, b=2, c=3, **kwargs):
        print(locals())
        print(test.passed_kwargs)
    
    test(b=6, d=9)
    """
    def decorator(function):
        def inner(*args, **kwargs):
            inner.passed_kwargs = kwargs
            return function(*args, **kwargs)
        return inner
    return decorator

def get_list_shape(lst):
    arr = np.array(lst, dtype=object)
    return arr.shape

def polled_sleep(t, t1=None, interval=0.1, debug=False):
    '''
    Looks like you can't interrupt normal time.sleep() inside a jupyter notebook.
    So this is a hack around that.
    '''
    if t1 is None:
        t1 = time.time()
    if t < interval:
        return polled_sleep(t, t1=t1, interval=interval/10, debug=debug)
    t2 = time.time()
    dt = t2-t1
    if debug:
        print('start : ', t, t1, dt, interval, dt < t)
    while dt < t:
        if debug:
            print('sleeping for ', interval)
        time.sleep(interval)
        t2 = time.time()
        dt = t2-t1
        if debug:
            print(t-dt)
        if t-dt < 0:
            return dt
        if t-dt < interval:
            if debug:
                print('recursing')
            return polled_sleep(t, t1=t1, interval=(t-dt)/10, debug=debug)

def try_remove(path, silent=False):
    '''
    Same as os.remove but doesn't throw an error if the
    path doesn't exist.
    '''
    try:
        os.remove(path)
        if not silent:
            print('removed : ', path)
    except FileNotFoundError:
        if not silent:
            print('warning : couldn\'t find ', path)

def default_eq(x, b, default_x=None, default_b=None):
    if x is default_x or b is default_b:
        return True
    else:
        return x == b

def default_key(d, key, default=None):
    try:
        return d[key]
    except KeyError:
        return default
    
def default_getattr(obj, attr, default=None):
    try:
        return getattr(obj, attr)
    except AttributeError:
        return default

def default_call(obj, default=None):
    try:
        return obj()
    except TypeError:
        if default == 'self':
            return obj
        else:
            return default

def filter_dict_by_key(dd, filter_key):
    return {key: dd[key] for key in dd if key != filter_key}

def filter_dict_by_value(dd, filter_value):
    return {key: dd[key] for key in dd if dd[key] != filter_value}

def merge_dict(d1, d2, merge_op=lambda x,y: y):
    '''
    Merges two dictionaries d1 and d2. If d1 and d2 have common keys then 
    merge_op is used to resolve how the values should be merged. merge_op should 
    be a function of two values. By default merge_op selects the value from d2, 
    which is the same behaviour as python's dict union operator.

    Overlapping keys are inserted in d1 order. The remaining d2 keys are inserted 
    at the end in d2 order.
    '''
    overlapping_keys = set(d1.keys()) & set(d2.keys())
    d1_keys = [k for k in d1.keys()]
    d2_keys = [k for k in d2.keys() if k not in overlapping_keys]
    dd = {}
    for k in d1_keys:
        if k in overlapping_keys:
            dd[k] = merge_op(d1[k], d2[k])
        else:
            dd[k] = d1[k]
    for k in d2_keys:
        dd[k] = d2[k]
    return dd

def flip_dict(dd, bijective=False):
    """Swap kay-value in a dictionary"""
    if bijective:
        outdd = {v:k for k,v in dd.items()}
    else:
        outdd = collections.defaultdict(list)
        for k,v in dd.items():
            outdd[v].append(k)
    return outdd
    
def extract_pickles_from_zip(zipname=None, file=None):
    if zipname is not None:
        name, ext = os.path.splitext(zipname)
        if ext == '':
            ext = '.zip'
        elif ext == '.zip':
            pass
        else:
            raise Exception('unknown extension')

    if file is None:
        file = open(f"{name}{ext}", 'rb')
        
    file.__enter__()
    
    data = []
    with zipfile.ZipFile(file) as z: 
        # printing all the contents of the zip file 
        infolist = z.infolist()
        infolist = sorted(infolist, key=lambda x: x.filename)
        for info in infolist:
            if os.path.splitext(info.filename)[-1] == '.pl':
                with z.open(info.filename) as f:
                    b = pl.load(f)
                data.append(b)

    file.__exit__()
    if len(data) == 1:
        return data[0]
    else:
        return data

def get_date(timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y_%m_%d')

def get_time(timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    return datetime.datetime.fromtimestamp(timestamp).strftime('%H_%M_%S')

def get_timestamp(timestamp=None):
    return get_date(timestamp) + "_" + get_time(timestamp)
    
class mytime:
    '''
    A small library for converting between HH:MM:SS and seconds and 
    other useful manipulations. Made it for adding up hours on 
    timesheets.
    '''
    def __init__(self):
	    pass
	    
    def t2s(self, h=0, m=0, s=0):
        ''' HH:MM:SS to seconds '''
        return h*60*60 + m*60 + s

    def s2t(self, s):
        ''' seconds to HH:MM:SS '''
        s_rem = s
    #     d = s_rem//(24*60*60)
    #     s_rem = s%(24*60*60)
        h = s_rem//(60*60)
        s_rem = s_rem%(60*60)
        m = s_rem//60
        s_rem = s_rem%(60)
        return (h, m, s_rem)

    def td(self, t1, t2):
        '''
        h1=[14,14,10,10,12,14,17,10,14,10,11,16,10,14,16]
        m1=[45,50,10,25,0,20,55,20,10,0,0,40,50,35,30]

        h2=[16,16,11,11,13,17,18,13,17,10,13,18,13,15,18]
        m2=[20,15,0,45,15,25,45,15,25,30,25,40,30,20,40]
        
        acc_s = 0
        acc_t = 0
        for i in range(len(h1)):
            t = td([h1[i],m1[i]],[h2[i],m2[i]])
            acc_s += np.array(t2s(*t))
            acc_t += np.array(t)
            print(t,acc_s,s2t(acc_s))
        print(acc_t,s2t(acc_s))
        '''
        return self.s2t(abs(self.t2s(*t1)-self.t2s(*t2)))

def get_data_in_quantile(x, q1=0, q2=1, return_quantiles=True, sort_output=False):
    n = len(x)
    x_argsort = np.argsort(x)
    x_iargsort = np.argsort(x_argsort)
    X = x[x_argsort]

    q = np.arange(n)/n
    s1 = int(np.round(q1*n))
    s2 = int(np.round(q2*n))
    if sort_output:
        Y = X[s1:s2]
        Q = q[s1:s2]
    else:
        Y = np.zeros_like(x) + np.nan
        Y[x_argsort[s1:s2]] = x[x_argsort][s1:s2]
        Y = Y[~np.isnan(Y)]

        Q = np.zeros_like(x) + np.nan
        Q[x_argsort[s1:s2]] = q[s1:s2]
        Q = Q[~np.isnan(Q)]
    if return_quantiles:
        return Q, Y
    else:
        return Y

def unit_arange(N):
    return np.arange(N)/N

def symmetric_linspace(lb, ub=None, N=30, symm_point=0):
    '''Useful for constructing a linspace symmetric around some point'''
    # assert(lb < symm_point)
    # assert(ub > symm_point)
    if ub is None:
        ub = lb
    b = np.max([symm_point-lb, ub-symm_point])
    return np.linspace(symm_point-b, symm_point+b, N)

def float_mag(x):
    'returns position of the floating point'
    return np.floor(np.log10(x))

def ceil_nsigfig(x, n):
    assert(n>0)
    sign = np.sign(x)
    x = np.abs(x)
    f = 10**(np.floor(np.log10(x)) - (n - 1))
    return np.ceil(sign*x/f)*f

def round_nsigfig(x, n):
    assert(n>0)
    sign = np.sign(x)
    x = np.abs(x)
    f = 10**(np.floor(np.log10(x)) - (n - 1))
    return np.round(sign*x/f)*f

def floor_nsigfig(x, n):
    assert(n>0)
    sign = np.sign(x)
    x = np.abs(x)
    f = 10**(np.floor(np.log10(x)) - (n - 1))
    return np.floor(sign*x/f)*f

def ceil_nsigfig_complex(x, n):
    x = np.complex128(x)
    xr, xi = ceil_nsigfig(np.real(x), n), ceil_nsigfig(np.imag(x), n)
    return xr + xi*1j

def round_nsigfig_complex(x, n):
    x = np.complex128(x)
    xr, xi = round_nsigfig(np.real(x), n), round_nsigfig(np.imag(x), n)
    return xr + xi*1j

def floor_nsigfig_complex(x, n):
    x = np.complex128(x)
    xr, xi = floor_nsigfig(np.real(x), n), floor_nsigfig(np.imag(x), n)
    return xr + xi*1j

def dB2ampl_ratio(dBs):
    return 10**(dBs/20)
    
def ampl_ratio2dB(ratios):
    return 20*np.log10(ratios)
    
def power_ratio2dB(ratios):
    return 10*np.log10(ratios)
    
def dB2power_ratio(dBs):
    return 10**(dBs/10)

def nd_argmax(array):
    return np.unravel_index(array.argmax(), array.shape)

def binary_search(search_limits, test_fun, verbose=False):
    # test_fun has to return [True,...,True,False,...,False]
    # on the search range, where the binary search will find the
    # last interger in the search range for which test_fun returns
    # true, if it's all True then it will return the upper search 
    # limit and lower limit if it's all False
    L, R = search_limits
    while True:
        m = np.floor((L+R)/2).astype(int)
        if verbose:
            print(L, R ,m)
        if L > R:
            return m
        else:
            if test_fun(m):
                L = m + 1
            else:
                R = m - 1 

def append_zip(zipped_list, new_list):
    # unzip zipped_list and pass the individual lists 
    # as a tuple into zip() together with the new list
    unzipped_list = zip(*zipped_list)
    return zip(*unzipped_list, new_list)

def matmul_reduce(mat_list, initial=None):
    '''
    Apparently np.matmul.reduce is not defined.
    So I made my own.
    '''
    if mat_list == []:
        if initial is None:
            raise TypeError('reduce() of empty sequence with no initial value')
        else:
            return initial
    if initial is None:
        N = np.shape(mat_list[0])[0]
        initial = np.eye(N)
    return functools.reduce(np.matmul, mat_list, initial)

def unzip_selector(zipped_list, index):
    unzip = zip(*zipped_list)
    return next(itertools.islice(unzip, index, None))
    
def ring_append(ring_arr, apnd_array):
    '''Append an nd_array to another nd_array in a
    ring buffer fashion'''
    N = apnd_array.size
    assert(N < ring_arr.size)
    rold_arr = np.roll(ring_arr, -N)
    rold_arr[-N:] = apnd_array
    return rold_arr
    
def polyfit2d(x, y, z, order=3, mode_list=None):
    # x :: (1,N) array
    # y :: (1,M) array
    # z :: (1,N*M) or (1,N=M) array
    # m :: (1,(order+1)**2) array
    
    N = np.size(x)
    M = np.size(y)
    if np.size(z) == N*M:
        outer = True
    else:
        outer = False
    
    # print(outer)

    if mode_list is None:
        ij = list(itertools.product(range(order+1), range(order+1)))
    else:
        ij = mode_list
        
    ncols = np.shape(np.array(ij))[0]
    
    G = np.zeros((z.size, ncols))
    for k, (i,j) in enumerate(ij):
        if outer:
            G[:, k] = np.multiply.outer(x**i, y**j).ravel()
        else:
            G[:, k] = np.multiply(x**i, y**j).ravel()
        
    #print(G.shape,z.shape)
    m, _, _, _ = np.linalg.lstsq(G, z, rcond=-1)
    return m

def polyval2d(x, y, m, outer=True, mode_list=None):
    # x :: (1,N) array
    # y :: (1,M) array
    # m :: (1,(order+1)**2) array
    # z :: (1,N*M) array
    order = int(np.sqrt(len(m))) - 1
    
    if mode_list is None:
        ij = list(itertools.product(range(order+1), range(order+1)))
    else:
        ij = mode_list
    
    if outer:
        z = np.zeros([y.size, x.size])
        for a, (i,j) in zip(m, ij):
            z += a * np.multiply.outer(y**j, x**i)
    else:
        z = np.zeros_like(x)
        for a, (i,j) in zip(m, ij):
            z += a * x**i * y**j
    return z
    
def polyfit3d(x, y, z, f, order=3):
    # x :: (1,N) array
    # y :: (1,M) array
    # z :: (1,N*M) array
    # m :: (1,(order+1)**2) array
    ncols = (order + 1)**3
    G = np.zeros((x.size, ncols))
    ijk = itertools.product(range(order+1), range(order+1), range(order+1))
    for n, (i,j,k) in enumerate(ijk):
        G[:,n] = x**i * y**j * z**k
    m, _, _, _ = np.linalg.lstsq(G, f, rcond=-1)
    return m
    
def polyval3d(x, y, z, m):
    # x :: (1,N) array
    # y :: (1,M) array
    # m :: (1,(order+1)**2) array
    # z :: (1,N*M) array
    order = int(np.sqrt(len(m))) - 1
    ijk = itertools.product(range(order+1), range(order+1), range(order+1))
    f = np.zeros_like(x)
    for a, (i,j,k) in zip(m, ijk):
        f += a * x**i * y**j * z**k
    return f

def inplace_print(self, *args):
    # clear the previous line
    clear_s = ' '*0
    print(clear_s, end='\r')
    new_s = ''
    for arg in args:
        new_s += str(arg) + ' '
    new_s = new_s[:-1]
    print(new_s, end='\r')
    return None
    
############################### NUMPY STUFF #######################################
    
def numpy_outer(A, B, op=np.add):
    '''
    Performs outer operation on two ND numpy arrays.
    The outer operation is done on the first indices of A and B.
    op(A[N, ...], B[M, ...]) = C[N, M, ...]
    Assumes op is a binary operation that can apply numpy broadcasting rules.
    '''
    Anew = np.expand_dims(A, 0)
    Bnew = np.expand_dims(B, 1)
    return op(Anew, Bnew)

def numpy_outer2(A, B, op=np.add):
    '''
    '''
    Ashape = np.shape(A)
    Bshape = np.shape(B)
    Da = len(Ashape)
    Db = len(Bshape)
    Anew = np.expand_dims(A, tuple(np.arange(Da,Db+Da)))
    Bnew = np.expand_dims(B, tuple(np.arange(Da)))
    return op(Anew, Bnew)

def outer_matmul(A, B):
    '''
    assumes A and B are stacks of matrices where the elements reside in the last two indices
    '''
    Ashape = np.shape(A)
    Bshape = np.shape(B)
    Da = len(Ashape) - 2
    Db = len(Bshape) - 2
    Anew = np.expand_dims(A, tuple(np.arange(Da,Db+Da)))
    Bnew = np.expand_dims(B, tuple(np.arange(Da)))
    return np.matmul(Anew, Bnew)

def get_last_nd(data, last_ind=[0,0]):
    nd = len(last_ind)
    slc = [slice(None)] * (data.ndim - nd)
    slc += [slice(x,x+1) for x in last_ind]
    # squeeze is needed to remove singleton dimesnions
    return np.squeeze(data[tuple(slc)])

def get_first_nd(data, first_ind=[0,0]):
    nd = len(first_ind)
    slc = [slice(x,x+1) for x in first_ind] 
    slc += [slice(None)] * (data.ndim - nd)
    # squeeze is needed to remove singleton dimesnions
    return np.squeeze(data[tuple(slc)])    
    
def insert_singletons(X, axes):
    '''
    insert_singletons(X, [0,0,0]).shape = (1,1,1) + X.shape
    insert_singletons(X, [-1,-1]).shape =  X.shape + (1,1)
    '''
    if isinstance(axes, int):
        axes = [axes]
    Xnew = X
    for axis in axes:
        Xnew = np.expand_dims(Xnew, axis)
    return Xnew
    
def matmul_transpose(X):
    '''
    Swaps the last two axes in an ND array.
    
    The matmul operator works over the last two axes for ND arrays. However the 
    numpy transpose reverses the order of all of the axes in an ND array. This is typically
    not what you want to do when doing matrix multiplication on an ND stack of matrices.
    '''
    axes = np.arange(len(X.shape))
    axes[-2:] = axes[-2:][::-1]
    return np.transpose(X, axes)

def atleast_nd(X, n):
    m = len(X.shape) - n
    if m >= 0:
        return X
    else:
        return insert_singletons(X, [0]*m)

############################### FILTER STUFF #######################################



####################################################################################

def apply_filt(u, v):
    return np.real(np.fft.ifft(np.fft.fft(u)*v))

def rect(x, mu=0, w=1, norm=False):
    dx = x[1] - x[0]
    r = np.ones_like(x, dtype=np.float64)
    r[x < mu - w] = 0
    r[x > mu + w] = 0
    r[x - mu + w == 0] = 0.5
    r[x - mu - w == 0] = 0.5
    if norm:
        r = r/np.sqrt(np.sum(r**2)*dx)
    return r
	
def jinc(x):
    if not x:
        return 1.0
    else:
        return 2*scipy.special.jv(1, x)/x
jinc = np.vectorize(jinc)

def lorentzian(fwhm, x, x0=0):
    '''maximum of lorentzian is 2/(pi*fwhm)'''
    return 0.5/np.pi*fwhm/((x-x0)**2+(0.5*fwhm)**2)
    
def lorentzian_fft(fwhm, x, x0=0):
    '''maximum of lorentzian is 2/(pi*fwhm)'''
    dx = x[1]-x[0]
    xf = np.fft.fftfreq(len(x))
    l_fft = np.exp(-np.abs(xf)*(fwhm/dx)*np.pi) * np.exp(-1j*(x0/dx)*(xf*2*np.pi) )
    return np.real(np.fft.ifft(l_fft))
    
def lorentzian2(fwhm, x, x0=0):
    '''maximum of lorentzian is 1.0'''
    return (fwhm/2)**2/((x-x0)**2+(fwhm/2)**2)
    
def airy(x, fwhm, fsr, x0=0):
    '''maximum of airy is 2/(pi*fwhm)'''
    return 0.5/np.pi*fwhm/(np.sin((x-x0)*np.pi/fsr)**2+(0.5*fwhm)**2)

def airy2(x, fwhm, fsr, x0=0):
    '''maximum of airy is 1.0'''
    return (1/4)*(fwhm*np.pi/fsr)**2/(np.sin((x-x0)*np.pi/fsr)**2+(0.5*(fwhm*np.pi/fsr))**2)

def gaussian_deriv_poly(x, order):
    if order%2 == 0:
        return np.polynomial.hermite_e.hermeval(x,np.r_[np.zeros(order),1])
    else:
        return np.polynomial.hermite_e.hermeval(x,np.r_[np.zeros(order),-1])

def gaussian_assym(x, mu=0, sl=1, sr=None, norm=False):
    '''
    Assymetric gaussian

    Example:

    N = 501
    s = 1
    sl = s
    sr = s + 1.4
    xs = np.linspace(-5*sl,5*sr,N)
    dx = xs[1] - xs[0]
    ys = gaussian_assym(xs, sl=sl, sr=sr, norm=True)

    plt.plot(xs, ys)
    '''
    if sr is None:
        sr = sl
    ys = np.zeros_like(x)
    xli = x < mu
    xri = np.logical_not(xli)
    ys[xli] = np.exp(-(x[xli]-mu)**2/sl**2)
    ys[xri] = np.exp(-(x[xri]-mu)**2/sr**2)
    if norm:
        ys = ys / ((sl+sr)/2) / np.sqrt(np.pi)
    return ys

def gaussian(x, mu=0, sigma=1, order=0):
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2) * gaussian_deriv_poly(x,order)

def gaussian2(x, mu=0, sigma=1, order=0):
    return np.exp(-0.5*((x-mu)/sigma)**2) * gaussian_deriv_poly(x,order)

def gaussian0(x, mu=0, sigma=1):
    g = np.exp(-((x-mu)/sigma)**2) * 1/(sigma*np.sqrt(np.pi))
    return g

def gaussian_p(x, mu=0, sigma=1):
    '''
    Gaussian discrete probability mass
    
    Defined to satisfy:
        np.sum(gaussian_p(xs,mu,sigma)) == 1
        
    '''   
    g = np.exp(-((x-mu)/sigma)**2)
    g = g / np.sum(g)
    return g

def log_gaussian(x, mu, sigma):
    return -0.5*((x-mu)/sigma)**2 - 0.5*np.log(2*np.pi*sigma**2)
    
def log_likelihood(x, mu, sigma):
    n1 = len(x)
    n2 = len(mu)
    assert n1 == n2
    return -0.5/sigma**2*np.sum((x-mu)**2) - 0.5*n1*np.log(2*np.pi*sigma**2)
    
def lorentzian_filt(fwhm, N):
    # if used as a conv filter then the fft spectrum is 
    # exp(-fwhm/(1*N/pi) * range(N))
    x = np.arange(N)
    x0 = N/2
    return lorentzian(fwhm,x,x0)
    
def sinc_filt(b, N):
    # if used as a conv filter then the fft spectrum is 
    # rect(N/b) ie b is the fraction of the frequencies kept
    x = np.arange(N)-N/2
    return 1/b * np.sinc(x/b)
    
def myconv(u, v, mode='same', centered=True):
    # numpy.convolve is time-domain for some stupid reason
    u_fft = np.fft.fft(u)
    v_fft = np.fft.fft(v)
    conv = np.fft.ifft(u_fft*np.conj(v_fft))
    if centered:
        conv = np.fft.fftshift(conv)
    return conv
    
def myconv2(u, v, mode='same'):
    # numpy.convolve is time-domain for some stupid reason
    u_fft = np.fft.fft2(u)
    v_fft = np.fft.fft2(v)
    return np.fft.fftshift(np.fft.ifft2(u_fft*v_fft))   
    
def mycorr(u, v, mode='same', centered=True):
    # numpy.convolve is time-domain for some stupid reason
    purely_real = False
    u = np.array(u)
    v = np.array(v)
    if np.max(u.imag) == 0 and np.max(v.imag) == 0:
        purely_real = True
    u_fft = np.fft.fft(u)
    v_fft = np.fft.fft(v)
    corr = np.fft.ifft(u_fft*np.conj(v_fft))
    if centered:
        corr = np.fft.fftshift(corr)
    if purely_real:
        corr = np.real(corr)
    return corr
    
# def myconv(u,v):
    # return scipy.convolve(u,v,'valid')
    
def hann_wn(N):
    n = np.arange(N)
    return 0.5 - 0.5*np.cos(2*np.pi*n/(N-1))

def hann_wn2(N, M=None):
    if M is None:
        M = N
    nx = np.arange(N)/(N-1) - 0.5
    ny = np.arange(M)/(M-1) - 0.5
    nr = np.sqrt(np.add.outer(ny**2, nx**2))
    out = 0.5 + 0.5*np.cos(2*np.pi*nr)
    out[nr >= 0.5] = 0
    return out
    
def hamming_wn(N, alpha=25/46):
    beta = 1 - alpha
    n = np.arange(N)
    return alpha - beta*np.cos(2*np.pi*n/(N-1))

def confined_gaussian_wn(N, sigma_t=0.1):
    ns = np.arange(N)
    N = N - 1
    L = N + 1
    def G(x):
        out = np.exp(-((x-N/2)/(2*L*sigma_t))**2)
        return out
    
    out = G(ns) - G(0.5)*(G(ns+L) + G(ns-L))/(G(-0.5 + L) + (G(-0.5 - L)))
    return out

def gaussian_wn(N, sigma=0.17):
    ns = np.arange(N)/(N-1) - 0.5
    out = np.exp(-(ns/sigma)**2)
    return out

def gaussian_wn2(N, M, sigma=0.17):
    g1 = gaussian_wn(N, sigma)
    g2 = gaussian_wn(M, sigma)
    return np.outer(g2, g1)
    
def sinc_filt_fiir(b, N, M=100, wn_fun=hamming_wn):
    # if used as a conv filter then the fft spectrum is 
    # rect(N/b) ie b is the fraction of the frequencies kept
    full_filt = np.zeros(N)
    wn = wn_fun(M)
    filt = sinc_filt(b,M)*wn
    filt = filt / np.sum(filt) # renormalize filter for unity gain
    full_filt[0:M] = filt
    return np.roll(full_filt, np.ceil(N/2-M/2).astype(int))

def brick_lowpass(signal, b=30, M=100, wn_fun=hamming_wn, plot_response=False, remove_edge=True):
    '''
    b is the fraction of the low frequencies kept
    1 keeps all freqs, 1/2, keeps half of freqs, 1/10 keeps a tenth ...
    M is the length of the filter as M approaches N the filter approaches an ideal brick_lowpass
    in practice M should be in range of [10,100]
    '''
    N = len(signal)
    full_filt = sinc_filt_fiir(b,N,M,wn_fun)
    if plot_response:
        plt.figure()
        plt.semilogy(np.abs(np.fft.fft(full_filt))[0:N//2])
        plt.show()
    
    return myconv(signal,full_filt)

def kernel_triangle(x, dx):
    y = -np.abs(x)/dx + 1
    y[y<0] = 0
    return y

def kernel_quadratic(x, dx):
    dx = dx/3
    y = 3*(dx-x)**2*np.sign(x-dx) - (x-3*dx)**2*np.sign(x-3*dx) - 3*(dx+x)**2*np.sign(dx+x) + (3*dx+x)**2*np.sign(3*dx+x)
    return y/(12*dx**2)

def kernel_lanczos(x, dx, a=2):
    xp = x/dx
    y = np.sinc(xp) * np.sinc(xp/a)
    y[np.abs(xp) > a] = 0
    return y

def kernel_sinc(x, dx):
    xp = x/dx
    y = np.sinc(xp)
    return y

def kernel_tps(x):
    import scipy.special
    r = np.sqrt(x**2)
    g = scipy.special.xlogy(r**2,r)
    return g

def construct_toeplitz_convolution(y, kernel_size=3, valid=True, return_x=False):
    '''
    If kernel is a vector h then convolution is computed by y = h@T
    Should be numerically identical to np.convolve(y, h, valid=valid)
    '''
    m = kernel_size-1
    n = len(y)
    if valid:
        print(m,n)
        x = np.arange(m, n) - kernel_size/2
        c = y[:m+1][::-1]
        r = y[m:]
    else:
        x = np.arange(0, n)
        c = np.r_[y[0], [0.]*m]
        r = np.r_[y, [0.]*m]
    T = linalg.toeplitz(c,r)
    if return_x:
        return x, T
    else:
        return T

def nearest_interp_matrix(xold, xnew):
    '''
    Computes the nearest neighbour interpolation resampling matrix for transforming
    yold to ynew given xold and xnew.
    
    xold and xnew need not be same size or even sorted.
    '''

    # assume xold and xnew aren't sorted
    xold_as  = np.argsort(xold)
    xnew_as  = np.argsort(xnew)

    xold = xold[xold_as]
    xnew = xnew[xnew_as]

    # get the inverse sort so we can get back to the
    # original order at the end
    xold_ias = np.argsort(xold_as)
    xnew_ias = np.argsort(xnew_as)

    M = np.zeros([len(xnew), len(xold)])
    B = np.subtract.outer(xnew, xold)

    nearest_xold = np.argmin(np.abs(B), axis=1)
    for i in range(len(xnew)):
        M[i, nearest_xold[i]] = 1

    # revert to the original (possibly unsorted) order
    M = M[:, xold_ias]
    M = M[xnew_ias, :]
    
    return M

def linear_interp_matrix(xold, xnew, debug=False):
    '''
    Computes the linear interpolation resampling matrix for transforming
    yold to ynew given xold and xnew.
    
    xold and xnew need not be same size or even sorted.
    
    If a point in xnew lies outside of the range of xold then the corresponding
    row of matrix elements will be nan.
    
    Example:
    
    xold, yold = np.hstack([np.random.rand(2,6),[[0,1],[0,1]]])
    xnew = np.linspace(-1, 2, 301) # points outside range [0, 1] will not be interpolated
    M = linear_interp_matrix(xold, xnew)
    ynew = M@yold

    plt.plot(xold, yold, 'x')
    plt.plot(xnew, ynew, '.')
    
    '''
    # assume xold and xnew aren't sorted
    xold_as  = np.argsort(xold)
    xnew_as  = np.argsort(xnew)
    
    xold = xold[xold_as]
    xnew = xnew[xnew_as]
    
    # get the inverse sort so we can get back to the
    # original order at the end
    xold_ias = np.argsort(xold_as)
    xnew_ias = np.argsort(xnew_as)

    M = np.zeros([len(xnew), len(xold)])
    B = np.subtract.outer(xnew, xold)
    sgB = np.sign(B)

    # hack for any points in xnew that exactly equal
    # any points in xold, which is common for the endpoints
    M[np.isclose(B,0)] = 1
    sgB[np.isclose(B,0)] = 1

    dsgB = np.diff(sgB, axis=1)

    ix, iy = np.where(dsgB<0)
    ix2 = np.hstack([ix, ix])
    iy2 = np.hstack([iy, iy+1])

    M[ix2, iy2] = np.abs(B[ix2, iy2])

    # compute normalization weights
    w = np.sum(M, axis=1)[:, None]
    
    # if normalization is zero then we are extrapolating
    # make it nan to avoid dividing by zero
    w[w==0] += np.inf # used to be np.nan
    M /= w    
    M[ix2, iy2] = 1 - M[ix2, iy2]
    
    # revert to the original (possibly unsorted) order
    M = M[:, xold_ias]
    M = M[xnew_ias, :]
    
    if debug:
        fig, ax = plt.subplots(2, 2, figsize=[8*2, 8])

        im = ax[0,0].pcolormesh(B)
        fig.colorbar(im, ax=ax[0,0])
        ax[0,0].set_ylabel('xnew')
        ax[0,0].set_title('B: xnew - xold')

        im = ax[0,1].pcolormesh(sgB)
        fig.colorbar(im, ax=ax[0,1])
        ax[0,1].set_title('sgB: sign(xnew - xold)')

        im = ax[1,0].pcolormesh(dsgB)
        fig.colorbar(im, ax=ax[1,0])
        ax[1,0].set_xlabel('xold')
        ax[1,0].set_ylabel('xnew')
        ax[1,0].set_title('diff(sgB, axis=1): [sign(xnew - xold)]/d[xold]')

        im = ax[1,1].pcolormesh(M)
        fig.colorbar(im, ax=ax[1,1])
        ax[1,1].set_xlabel('xold')
        ax[1,1].set_title('M: Resampling Matrix')
        
    
    return M
    
def gauss_linear_interp_matrix(xold, xnew, scale=1, debug=False):
    '''
    Computes the averaged linear interpolation resampling matrix for transforming
    yold to ynew given xold and xnew. The kernel of this transformation is just 
    the convolution of the linear interpolation and gaussian kernels.
    
    xold and xnew need not be same size or even sorted.
    
    '''
    Nold = np.size(xold)
    xold_ptp = np.ptp(xold)
    
    # assume xold and xnew aren't sorted
    xold_as  = np.argsort(xold)
    xnew_as  = np.argsort(xnew)
    
    xold = xold[xold_as]
    xnew = xnew[xnew_as]
    
    # get the inverse sort so we can get back to the
    # original order at the end
    xold_ias = np.argsort(xold_as)
    xnew_ias = np.argsort(xnew_as)

    # make the gaussian kernel matrix
    gauss_mask = np.exp(-np.subtract.outer(xold, xold)**2/(xold_ptp/Nold)**2/scale)
    # normalize it
    mask_norm = np.sum(gauss_mask, axis=1)
    gauss_mask = gauss_mask/mask_norm[:, None]
    
    # get the linear interpolation matrix
    M = linear_interp_matrix(xold, xnew)
    
    # convolve the two kerenels
#     comb = np.sum((M[..., None]*gauss_mask[None, ...]), axis=1) # this is a very inefficient matmul
    comb = (M @ gauss_mask)
    # renormalize again
    norm = np.sum(comb, axis=1)[:, None]
    comb = comb/norm
    
    if debug:
        nrows = 2
        ncols = 2
        fig, ax = plt.subplots(nrows, ncols, figsize=[5*1.8*ncols,5*nrows])
        im = ax[0,0].pcolormesh(gauss_mask)
        fig.colorbar(im, ax=ax[0,0])
        im = ax[0,1].pcolormesh(M)
        fig.colorbar(im, ax=ax[0,1])
        im = ax[1,0].pcolormesh(comb)
        fig.colorbar(im, ax=ax[1,0])
        
    # restore original order
    comb = comb[:, xold_ias]
    comb = comb[xnew_ias, :]
    
    return comb

def rbf_resample_matrix(xold, xnew, scaling_method='old', scale=1, return_distance=False, debug=False):
    '''
    xold : N x 1 array
    xnew : M x 1 array
    weights : M X N array
    norm : M x 1 array
    '''
    xold_ptp = np.ptp(xold)
    Nold = np.size(xold)
    grad_xnew = np.gradient(xnew)[:, None]
    grad_xold = np.gradient(xold)
    distance = np.subtract.outer(xnew, xold)
    if return_distance:
        return distance
    if scaling_method == 'old':
        grad_x = np.maximum(grad_xnew, np.min(grad_xold)/np.sqrt(2))
        weights = np.exp(-distance**2/grad_x**2/scale)
    else:
        weights = np.exp(-distance**2/(xold_ptp/Nold)**2/scale)
    norm = np.sum(weights, axis=1)[:, None]
    return weights/norm

def resample_matrix(xold, xnew, scale=1, interp='linear', mixing_strength=1):
    dist_matrix = np.subtract.outer(xnew, xold)
    d_xnew = np.diff(xnew)
    d_xnew = np.r_[d_xnew[0],d_xnew,d_xnew[-1]]
    
    # build averaging matrix
    W1 = np.zeros_like(dist_matrix)
    for i in range(len(xnew)):
        # get forwards and backwards x derivatives at xnew[i]
        df_xnew = d_xnew[i+1]
        db_xnew = d_xnew[i]
        W1[i,:] = gaussian_assym(-dist_matrix[i,:], mu=0, sl=db_xnew/3*scale, sr=df_xnew/3*scale)

    # get interpolation matrix
    if interp == 'linear':
        W2 = linear_interp_matrix(xold, xnew)
    elif interp == 'nearest':
        W2 = nearest_interp_matrix(xold, xnew)
    elif interp is None:
        W2 = np.zeros_like(W1)
    else:
        raise NotImplementedError
        
    W = W1 + W2*mixing_strength

    # renormalize resampling matrix
    W = W / np.sum(W,axis=1)[:,None]
    
    return W

def linear_resampling_matrix(xold, xnew, scale=1):

    d_xnew = np.diff(xnew)
    d_xnew = np.r_[d_xnew[0],d_xnew,d_xnew[-1]]

    M = linear_interp_matrix(xold, xnew)
    Mi = np.argsort(M,axis=1)

    # build averaging matrix
    W1 = np.zeros([xnew.size, xold.size])
    for i in range(len(xnew)):
        # get forwards and backwards x derivatives at xnew[i]
        df_xnew = d_xnew[i+1]
        db_xnew = d_xnew[i]
        xbi, xfi = Mi[i,-2:]
        W1[i,:] += M[i,xbi] * gaussian_p(xold, mu=xold[xbi], sigma=db_xnew/3*scale)
        W1[i,:] += M[i,xfi] * gaussian_p(xold, mu=xold[xfi], sigma=df_xnew/3*scale)
    
    return W1

def resample_data(x_old, y_old, x_new, scale=1):
    W = rbf_resample_matrix(x_old, x_new, scale)
    y_new = W@y_old
    return y_new

def stitch_data(xs_old, ys_old, sort=True, debug=False):
    x_old = np.hstack(xs_old)
    y_old = np.hstack(ys_old)
    if debug:
        print('x.shape:', x_old.shape, 'y.shape:', y_old.shape)
    if sort:
        inds = np.argsort(x_old)
        x_old = x_old[inds]
        y_old = y_old[..., inds]
    return x_old, y_old

def stitch_resample_data(xs_old, ys_old, x_new, scale=1, interp='linear', mixing_strength=1):
    '''
    Example:
    
    import general_funs as gf
    import numpy as np
    import matplotlib.pyplot as plt

    intervals = [[-10,-6],[-4,4],[6,10]]
    xs_old = np.array([np.linspace(np.min(x), np.max(x), 800) for x in intervals])
    ys_old = xs_old**2 + np.random.randn(*xs_old.shape)*5

    x_new = np.linspace(-10,10,501)
    y_new = gf.stitch_resample_data(xs_old, ys_old, x_new)

    for i in range(len(xs_old)):
        plt.plot(xs_old[i], ys_old[i])    
    plt.plot(x_new, y_new)  
    '''
    x_old, y_old = stitch_data(xs_old, ys_old)
    W = resample_matrix(x_old, x_new, scale=scale, interp=interp, mixing_strength=mixing_strength)
    y_new = W@y_old
    return y_new

def forward_difference(x, full=False):
    diff = np.diff(x)
    if full:
        diff = np.r_[diff[0], diff]
    return diff

def backward_difference(x, full=False):
    diff = np.diff(x)
    if full:
        diff = np.r_[diff, diff[-1]]
    return diff

def central_difference(x, full=False):
    diff = forward_difference(x, full=True) + backward_difference(x, full=True)
    return diff/2

def welch_spectra_stitch(xs, fs, fnew, scale=1, drop=0, nperseg=None, debug=False):
    '''
    xs : N x ... x M array
    fs : N x 1 array
    f_axis : P x 1 array
    
    W : P x M array
    XS_stitched : ... x M array   
    XS_new : ... x P array
    '''
    
    XS = []
    f_axes = []
    for i,f in enumerate(fs):
        x = xs[i]
        f_axis, X = scipy.signal.welch(x, fs=f, nperseg=nperseg)
        f_axes.append(f_axis[drop:])
        XS.append(X[..., drop:])

    if debug:
        print(np.shape(XS))

    f_stitched, XS_stitched = stitch_data(f_axes, XS, debug=debug)
    if debug:
        print('f_stitched.shape:', f_stitched.shape, 'XS_stitched.shape', XS_stitched.shape)
    W = rbf_resample_matrix(f_stitched, fnew, scale=scale)
    # W = gauss_linear_interp_matrix(f_stitched, fnew, scale=scale)
    
    XS_new = (W@XS_stitched.T).T
    
    return XS_new
    
####### GEOMETRY STUFF #######

def sind(x):
    theta = x/180*np.pi
    return np.sin(theta)

def arcsind(x):
    theta = x/180*np.pi
    return np.arcsin(theta)

def cosd(x):
    theta = x/180*np.pi
    return np.cos(theta)

def arccosd(x):
    theta = x/180*np.pi
    return np.arccos(theta)

def scale_spread(x, s=1, f_center=np.median):
    x = np.array(x)
    xbar = f_center(x)
    xp = (x-xbar)*s + xbar
    return xp

def scale_std(x, s=1):
    return scale_spread(x, s=s, f_center=np.mean)
    
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)    
    
def norm_vec(v):
    return v/np.sqrt(np.sum(v**2))

def random_bool():
    b = not random.getrandbits(1)
    return b

def random_phase(*args, rad=False, **kwargs):
    r = np.exp(1j * 2*np.pi * np.random.rand(*args, **kwargs))
    if rad:
        r = np.angle(r)
    return r

def randn_complex(*args, **kwargs):
    return np.random.randn(*args, **kwargs) * random_phase(*args, **kwargs)

def rand_complex(*args, **kwargs):
    return np.random.randn(*args, **kwargs) + 1j*np.random.randn(*args, **kwargs)

def gen_rand_norm_vec():
    r = np.random.rand(3)-0.5
    return norm_vec(r)
    
def gen_rand_norm_mat():
    # random matrix where every column is normalized
    r1 = gen_rand_norm_vec()
    r2 = gen_rand_norm_vec()
    r3 = gen_rand_norm_vec()
    return np.vstack([r1,r2,r3]).T

def gen_rand_rot_mat():
    # random matrix where every colum is normalized and 
    # orthogonal to each other
    v1 = gen_rand_norm_vec()
    
    r2 = gen_rand_norm_vec()    
    # subtract off the component of r2 parallel to v1
    v2 = norm_vec(r2 - r2.dot(v1)*v1)
    
    # generate the last vector via cross product
    v3 = np.cross(v1,v2)
    
    return np.vstack([v1,v2,v3]).T
    
def solve_for_euler_angles(R):
    # Follows R = R_z(phi)*R_y(theta)*R_x(psi) convention.
    # Note that there are always two sets of euler angles for
    # any non-degenerate (cos(theta) != 0) rotation matrix.
    # Code taken from Slabaugh's article
    if abs(R[2,0]) != 1:
        theta1 = - np.arcsin(R[2,0])
        theta2 = np.pi - theta1
        psi1 = np.arctan2(R[2,1]/np.cos(theta1),R[2,2]/np.cos(theta1))
        psi2 = np.arctan2(R[2,1]/np.cos(theta2),R[2,2]/np.cos(theta2))
        phi1 = np.arctan2(R[1,0]/np.cos(theta1),R[0,0]/np.cos(theta1))
        phi2 = np.arctan2(R[1,0]/np.cos(theta2),R[0,0]/np.cos(theta2))
    else:
        # abs(R[2,0]) == 1 -> degenerate case
        phi = 0 # could be anything
        if R[2,0] == -1:
            theta = np.pi/2
            psi = phi + np.arctan2(R[0,1],R[0,2])
        else:
            theta = - np.pi/2
            psi = -phi + np.arctan2(-R[0,1],-R[0,2])
            
    return (psi1,theta1,phi1)
    
def R2(theta):
    '''
    2D rotation matrix
    '''
    return np.matrix([\
                    [np.cos(theta),-np.sin(theta)],\
                    [np.sin(theta),np.cos(theta)]\
                    ])
    
def R_x(psi):
    return np.matrix([\
                     [1,0,0],\
                     [0,np.cos(psi),-np.sin(psi)],\
                     [0,np.sin(psi),np.cos(psi)]\
                    ])
def R_y(theta):
    return np.matrix([\
                     [np.cos(theta),0,np.sin(theta)],\
                     [0,1,0],\
                     [-np.sin(theta),0,np.cos(theta)]\
                    ])

def R_z(phi):
    return np.matrix([\
                     [np.cos(phi),-np.sin(phi),0],\
                     [np.sin(phi),np.cos(phi),0],\
                     [0,0,1]\
                    ])

def find_all_factors(n):
    '''stolen from https://stackoverflow.com/a/19578818/13161738'''
    assert(float(n).is_integer())
    n = int(n)
    from math import sqrt
    from functools import reduce
    step = 2 if n%2 else 1
    fac_set = set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(sqrt(n))+1, step) if n % i == 0)))
    return sorted(list(fac_set))

def find_most_multiple_factor(n):
    fac = find_all_factors(n)
    l = len(fac)
    return fac[l//2 if l%2 else l//2-1]

def cantor_pairing_function(n,m):
    '''useful for mapping two integers to a single integer sequence'''
    return (n+m)*(n+m+1)//2 + m  

def pickle_copy(obj):
    '''like deepcopy but works in more cases, like copying a matplotlib figure'''
    obj_bytes = pl.dumps(obj)
    new_obj = pl.loads(obj_bytes)
    return new_obj

def save_compressed(myobj, filename, compresslevel=1):
    name, ext = os.path.splitext(filename)
    # print(ext)
    if ext == '.bz2':
        CFile = bz2.BZ2File
    elif ext in ['.gz', '.zip']:
        CFile = gzip.GzipFile
    elif ext in ['.xz', '.lzma']:
        CFile = lzma.LZMAFile
    else:
        raise NameError(f'extension {ext} not found')
    try:
        f = CFile(filename, 'wb')
    except IOError:
        sys.stderr.write('File ' + filename + ' cannot be written\n')
        raise

    pl.dump(myobj, f)
    f.close()

def load_compressed(filename):
    name, ext = os.path.splitext(filename)
    if ext == '.bz2':
        CFile = bz2.BZ2File
    elif ext in ['.gz', '.zip']:
        CFile = gzip.GzipFile
    elif ext in ['.xz', '.lzma']:
        CFile = lzma.LZMAFile
    try:
        f = CFile(filename, 'rb')
    except IOError:
        sys.stderr.write('File ' + filename + ' cannot be read\n')
        raise

    myobj = pl.load(f)
    f.close()
    return myobj

def save_uncompressed(obj, filename):
    import pickle as pl
    with open(filename, 'wb') as f:
        pl.dump(obj, f)
        
def load_uncompressed(filename):
    import pickle as pl
    with open(filename, 'rb') as f:
        obj = pl.load(f)
    return obj

def groupby(d, key=lambda x: x):
    out = collections.defaultdict(list)
    for i in d:
        out[key(i)].append(i)
    return out