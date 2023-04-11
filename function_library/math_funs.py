import numpy as np
import scipy

def complex_cross_product(z1, z2):
    '''Treat complex numbers as vectors in R^2
    '''
    a,b = z1.real, z1.imag
    c,d = z2.real, z2.imag
    return a*d - b*c

def complex_dot_product(z1, z2):
    '''Treat complex numbers as vectors in R^2
    '''
    a,b = z1.real, z1.imag
    c,d = z2.real, z2.imag
    return a*c + b*d

def mobius_transform(xs, M_abcd):
    '''
    Parametrizes every possible rational involution (functions that are their own inverse).
    Provided that M_abcd@M_abcd = +/- np.eye(2)

    I found it very intersting that it has the same form as the formula 
    for q parameter propagation through ray optics ABCD matrices.
    '''
    A,B,C,D = M_abcd.ravel()
    out = (A*xs + B)/(C*xs + D)
    return out

def log_exp_involution(xs):
    '''
    A function I found on Wikipedia that is its own inverse (an involution).
    Doesn't seem to have a name.
    '''
    ys = np.log((np.exp(xs)+1)/(np.exp(xs)-1))
    return ys

def cot(x):
    return np.cos(x)/np.sin(x)

def sec(x):
    return 1/np.cos(x)

def csc(x):
    return 1/np.sin(x)

def gaussian_bump(x, s=1, norm=True, dtype=np.float64):
    xs = np.asarray(x)
    bump = np.zeros_like(xs, dtype=dtype)
    mask = np.abs(xs)<1
    bump[mask] = np.exp(-s/(1-xs[mask]**2)) * np.exp(s)
    if norm:
        c,e = scipy.integrate.quad(lambda x: gaussian_bump(x, s=s, norm=False), -1, 1)
        if e > 1e-6:
            print('large error in quad')
        bump = bump/c
    return bump
    
def bessel_bump(x, s=1, norm=True, dtype=np.float64):
    '''
    Taken from https://math.stackexchange.com/a/101484
    '''
    xs = np.asarray(x)
    bump = np.zeros_like(xs, dtype=dtype)
    mask = np.abs(xs)<1
    bump[mask] = 1/np.i0(s/(1-xs[mask]**2))
    if norm:
        c,e = scipy.integrate.quad(lambda x: bessel_bump(x, s=s, norm=False), -1, 1)
        if e > 1e-6:
            print('large error in quad')
        bump = bump/c
    return bump

@np.vectorize
def erfcxinv(y, debug=False):
    '''
    N = 300
    xs = np.logspace(-7,7,N)
    residual = (scipy.special.erfcx(erfcxinv(xs)) - xs)/xs
    '''
    erfcx = scipy.special.erfcx
    if y > 5:
        # initial guess for large y
        a,b = np.array([-0.13167196, -1.80620422])
        x = a*np.log(y) + b
    elif y <= 5:
        # initial guess for small y
        x = 1/(y*np.sqrt(np.pi))
    n = 0
    prev = -np.inf
    while prev != x:
        # newton's iterations
        n+=1
        prev = x
        x = x - (erfcx(x) - y)/(2*x*erfcx(x) - 2/np.sqrt(np.pi))
        if n > 60:
            break
    if debug:
        return x,n
    return x