import numpy as np
import pandas as pd
import scipy
import scipy.integrate as intgr

try:
    import fun_lib.general_funs as general_funs
except ModuleNotFoundError:
    try:
        import general_funs
    except ModuleNotFoundError:
        pass

from . import CONSTANTS
from . import maps
from .q import propag as q_propag

def lam_to_k(lam=CONSTANTS['lambda']):
    return 2*np.pi/lam

def compute_power(fields, sum_axes=None):
    fields = np.asarray(fields)
    intensity = np.abs(fields*np.conj(fields))
    powers = np.sum(intensity, axis=sum_axes)
    return powers

def HG_projection(U, xs, ys=None, qx=1j, qy=None, gx=0, gy=0, mode_list=[(0,0)], include_gouy=False):
    if qy is None:
        qy = qx
    if ys is None:
        ys = xs
        
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    
    nx_s = list(set(nm[0] for nm in mode_list))
    ny_s = list(set(nm[1] for nm in mode_list))
    exi = {n: np.conj(u_n_q(xs, qx, n=n, gamma=gx, include_gouy=include_gouy)) for n in nx_s}
    eyi = {m: np.conj(u_n_q(ys, qy, n=m, gamma=gy, include_gouy=include_gouy)) for m in ny_s}
    
    out = {}
    for nm in mode_list:
        n,m = nm
        out[n,m] = exi[n]@(eyi[m]@U) * np.sqrt(dx*dy)
        
    return out

def iHG_projection(H, xs, ys=None, qx=1j, qy=None, gx=0, gy=0, include_gouy=False):
    '''
    stuff
    '''
    if qy is None:
        qy = qx
    if ys is None:
        ys = xs

    mode_list = list(H.keys())
    nx_s = list(set(nm[0] for nm in mode_list))
    ny_s = list(set(nm[1] for nm in mode_list))
    ex = {n: u_n_q(xs, qx, n=n, gamma=gx, include_gouy=include_gouy) for n in nx_s}
    ey = {m: u_n_q(ys, qy, n=m, gamma=gy, include_gouy=include_gouy) for m in ny_s}

    out = np.zeros([len(ys), len(xs)], dtype=np.complex128)
    for nm in mode_list:
        n,m = nm
        out += np.outer(ey[n], ex[n]) * H[(n,m)]
    return out

def unpack_abcd(Ms):
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

def abcd_self_consistency_cm1(M):
    '''
    Only valid for B != 0, C != 0.
    
    Should return abcd_cm(C)
    
    Based on equating the CQC and QCQ decompositions
    '''
    A,B,C,D = M.ravel()
    assert(B != 0)
    assert(C != 0)
    return abcd_cc((1-A)/C)@abcd_cm((D-1)/B)@abcd_cc(B)@abcd_cm((A-1)/B)@abcd_cc((1-D)/C)

def abcd_self_consistency_cm2(a):
    '''
    Only valid for a != 0

    Based on equating the two scaling operator decompositions in Pei 2016 Eq 40.
    
    Should return abcd_cm(a)
    '''
    assert(a != 0)
    F = abcd_frt(1)
    iF = abcd_frt(-1)
    return iF@abcd_cm(-1/a)@F@abcd_cm(-a)@iF@abcd_cm(-1/a)@iF@abcd_cm(-a)@F@abcd_cm(-1/a)@iF

def accum_gouy_plp(q, M, n=0, m=0):
    '''
    Doesn't work for C=0
    '''
    m3,m2,m1 = abcd_plp_decomp(M)
    q1 = q_propag(q, m1)
    q2 = q_propag(q1, m2)
    q3 = q_propag(q2, m3)
    
    phi1 = q2gouy(q1, n=n, m=m)*np.conj(q2gouy(q, n=n, m=m))
    phi3 = q2gouy(q3, n=n, m=m)*np.conj(q2gouy(q2, n=n, m=m))
    
    exp_j_psi = phi1*phi3
    return exp_j_psi

def accum_gouy_lpl(q, M, n=0, m=0):
    '''
    Doesn't work for B=0
    '''
    m3,m2,m1 = abcd_lpl_decomp(M)
    q1 = q_propag(q, m1)
    q2 = q_propag(q1, m2)
    q3 = q_propag(q2, m3)
    
    exp_j_psi = q2gouy(q2, n=n, m=m)*np.conj(q2gouy(q1, n=n, m=m))
    return exp_j_psi

def accum_gouy_lpl_reduced(q, M):
    A,B,C,D = np.ravel(M)
    z = np.real(q)
    zr = np.imag(q)
    t = (B+A*np.conj(q))*q / np.sqrt(np.abs(q)**2*((B+A*z)**2 + A**2*zr**2))
    return t

def accum_gouy_Erden(q, M, lam=1064e-9):
    '''
    Agrees with Siegman and Finesse if you make sure to use Erden's definition 
    of ABCD matrices where B_erden = B_siegman*lam, and C_erden = C_siegman/lam
    Paper: https://doi.org/10.1364/JOSAA.14.002190
    eq 15

    Need to use atan2 to remove arctan sign ambiguity.
    '''
    A,B,C,D = M.ravel()
    
    r = q2R(q)
    w = q2w(q, lam=lam)
    B = B*lam # Erden's weird abcd definition
    
    tan_xeta = B/((A+B/(lam*r))*np.pi*w**2)
    xeta = np.arctan2(B,((A+B/(lam*r))*np.pi*w**2))
    return np.exp(1j*xeta)

def accum_gouy_Siegman(q, M):
    '''
    Siegman Chp 20.2 eq 26
    
    Note on sign conventions:
    The q in this expression has to be conjugated if the q in u_n_q is not 
    conjugated and vice versa.
    '''
    A,B,C,D = unpack_abcd(M)
    exp_j_psi = (A+B/np.conj(q))/np.abs(A+B/q)
    return exp_j_psi

def accum_gouy_Siegman_n(q, M, n=0):
    '''
    1D accumulated Gouy phase of a mode of order n and beam parameter q through
    abcd matrix M.
    '''
    n = np.array(n)
    exp_j_psi = accum_gouy_Siegman(q, M)
    return exp_j_psi**(n+1/2)

def accum_gouy_Siegman_nm(qx, Mx, n=0, qy=None, My=None, m=0):
    '''
    2D accumulated Gouy phase of a mode of order n, m and beam parameter qx, qy 
    through abcd matrix Mx, My.
    '''
    n = np.array(n)
    m = np.array(m)
    if My is None:
        My = Mx
    if qy is None:
        qy = qx

    exp_j_psi_x = accum_gouy_Siegman_n(qx, Mx, n=n)
    exp_j_psi_y = accum_gouy_Siegman_n(qy, My, n=m)
    return exp_j_psi_x * exp_j_psi_y
    
def make_cavity(q1,d=1):
    '''
    This will try to find the mirror curvatures of a fabry perot
    whos eigenmode q2 matches the closest for a particular input q1.
    
    Not guaranteed to converge for a particular choice of d, though if it does the solution should be unique.
    '''
    import scipy.optimize
    L1 = abcd_space(d)
    def iterfun(Rs):
        R1,R2 = Rs
        M1 = abcd_mirror(R1)
        M2 = abcd_mirror(R2)
        Mtot = np.matmul(np.matmul(np.matmul(M2,L1),M1),L1)
        q2 = abcd_eig(Mtot)
#         print(q2,mode_mismatch(q2,q1))
        if q2.imag == 0:
            return np.inf
        else:
            return mode_mismatch(q2,q1)
    soln = scipy.optimize.minimize(iterfun,(2,2),method='Nelder-Mead',options={'xatol' : 1e-7, 'ftol' : 1e-7})   
    return soln
    
def make_cavity2(q1):
    '''
    This will try to find the mirror curvatures and separation of a fabry perot
    whos eigenmode q2 matches the closest for a particular input q1.
    
    Guaranteed to converge for any physical q1, though the solution is not unique.
    '''
    import scipy.optimize
    def iterfun(Rs):
        R1,R2,d = Rs
        M1 = abcd_mirror(R1)
        L1 = abcd_space(d)
        M2 = abcd_mirror(R2)
        Mtot = np.matmul(np.matmul(np.matmul(M1,L1),M2),L1)
        q2 = abcd_eig(Mtot)
#         print(q2,mode_mismatch(q2,q1))
        if q2.imag == 0:
            return np.inf
        else:
            return mode_mismatch(q2,q1)
    soln = scipy.optimize.minimize(iterfun,(4,4,2),method='Nelder-Mead',options={'xatol' : 1e-7, 'ftol' : 1e-7, 'adaptive':True})   
    return soln

def rt_gouy(M):
	A,B,C,D = M.flatten()
	xi = np.sign(B)*np.arccos((A+D)/2)
	return xi
	
def rt_gouy2(M):
	A,B,C,D = M.flatten()
	xi = 2*np.arccos(np.sign(B)*np.sqrt((A+D+2)/4))
	return xi

def generate_continuous_optical_system(N=None, s=0.05, merge_lens_space=True):
    '''
    N is the number of (space,lens) pairs in the system.
    
    One Gouy-phase period corresponds to N = 12.5/s
    
    Useful as a test ABCD system
    '''
    if N is None:
        N = int(np.round(12.5/s))
    ms = []
    for i in range(N):
        if merge_lens_space:
            ms.append(abcd_lens_p(s)@abcd_space(s))
        else:
            ms.append(abcd_lens_p(s),abcd_space(s))
    return ms

def accum_gouy_optical_system(q, ms, ndims=1, n=0, partials=False):
    '''
    Calculates the accumulated Gouy-phase through a sequence of ABCD matrices
    component-by-component to account for the metaplectic sign flips.
    '''
    ms = ms[::-1] # first optical component is the last ABCD matrix
    gs = []
    g = 1
    for m in ms:
        if ndims == 1:   
            g = g * accum_gouy_Siegman_n(q, m, n)
        elif ndims == 2:
            g = g * accum_gouy_Siegman(q, m)
        gs.append(g)
        q = q_propag(q, m)
        
    if partials:
        return np.array(gs)
    else:
        return g

def accum_abcd(ms):
    m1 = np.eye(2)
    acc_ms = []
    for m2 in ms[::-1]:
        m3 = m2@m1
        acc_ms.append(m3)
        m1 = m3
    return np.array(acc_ms)

def sign_to_generator(signs):
    '''
    sign = (-1)**generator
    '''
    generators = (signs + 2)%3
    return generators

def abcd_metaplectic_phase(M):
    A,B,C,D = unpack_abcd(M)
    z = (A + 1j*B)
    return z/np.abs(z)

def metaplectic_sign_flip_matmul(M2, M1, debug=False):
    M3 = M2@M1
    A1,B1,C1,D1 = unpack_abcd(M1)
    A2,B2,C2,D2 = unpack_abcd(M2)
    A3,B3,C3,D3 = unpack_abcd(M3)
    b1 = (A1 + 1j*B1)
    b2 = (A2 + 1j*B2)
    b3 = (A3 + 1j*B3)
    # print(b1,b2,b3)
    g1 = np.angle(b1)
    g2 = np.angle(b2)
    g3 = np.angle(b3)
    
    if debug:
        print(np.array([g1,g2,g3])/np.pi, end=', ')

    # M2 cannot rotate M1 by more than 180 degrees
    # with large ABCDs (elements > 10e10) one can occasionally
    # end up with M3 slightly behind M1 due to float error, which looks like 
    # M2 rotated M1 by 359.9999... degrees which flips the metaplectic sign
    safety = np.pi

    # make the safety less than exactly 180 degrees otherwise it breaks the 
    # parity operators which rotate by exactly 180 degrees
    safety = safety*0.99

    # the slow way but more verbose
    if g2 >= 0:
        if g3 < g1 - safety:
            out = 1.0
        else:
            out = 0.0
    elif g2 < 0:
        if g3 > g1 + safety:
            out = 1.0
        else:
            out = 0.0
    
    # fast but not numerically stable
    # c1 = g3 < g1
    # c2 = g2 < 0
    # out = float(np.logical_xor(c1,c2))
    
    if debug:
        print(out)
    return out

def accum_abcd_metaplectic_signs(ms, debug=False):
    '''
    Computes the meteplectic sign through a sequence of ABCD matrices using the 
    accumulated metaplectic winding number.
    
    The metaplectic sign flips whenever the winding number crosses an odd 
    multiple of pi.
    '''
    m1 = np.eye(2)
    s1 = 1.0
    s2 = 1.0
    signs = []
    for m2 in ms[::-1]:
        g3 = metaplectic_sign_flip_matmul(m2, m1, debug=debug)
        s3 = (-1)**g3
        m3 = m2@m1
        sign = s1*s2*s3
        signs.append(sign)
        m1 = m3
        s1 = sign

    signs = np.array(signs)
    return signs

def accum_abcd_metaplectic_winding_number(ms, debug=False):
    '''
    Computes the accumulated metaplectic winding number through a sequence
    of ABCD matrices.
    
    The metaplectic sign flips whenever the winding number crosses an odd 
    multiple of pi.
    '''
    ms = ms[::-1]
    phis = [0]
    M = np.eye(2)
    for m in ms:
        A1,B1,C1,D1 = unpack_abcd(M)
        A2,B2,C2,D2 = unpack_abcd(m)
        A3,B3,C3,D3 = unpack_abcd(m@M)
        b1 = A1 + 1j*B1
        b2 = A2 + 1j*B2
        b3 = A3 + 1j*B3
        p1 = np.angle(b1)
        p2 = np.angle(b2)
        p3 = np.angle(b3)
        
        wrap = metaplectic_sign_flip_matmul(m,M)
        
        dphi = p3 - p1
        if wrap:
            # unwrap phase using the sign of next abcd
            dphi += 2*np.pi*np.sign(p2)

        # dphi = np.angle(b3*np.conj(b1))
        phis.append(phis[-1] + dphi)
        M = m@M
        
        if debug:
            print(p1, p2, p3, dphi, wrap)
    return np.array(phis[1:])
    
def herm(n,x):
    from numpy.polynomial import hermite
    
    if n == -1:
        np.exp(x**2)
        return 0.5 * np.sqrt(np.pi) * np.exp(x**2) * (-np.vectorize(np.math.erf)(x) + 1)
    elif n == -2:
        return 0.25 * np.sqrt(np.pi) * ((2*(np.sqrt(np.pi)*np.exp(x**2)*x*np.math.erf(x) + 1))/np.sqrt(np.pi) - 2*np.exp(x**2)*x)
    
    c = np.zeros(n+1)
    c[-1] = 1
    return hermite.hermval(x,c)
    
def herm_q(n, x, q, lam=1064e-9):
    w = q2w(q, lam=lam)
    return herm(n, np.sqrt(2)*x/w)
    
def laguerre_coeff(n,alpha):
    '''
    stole this one from a mathworks post
    
    Geert Van Damme (4 Aug 2008)
    
    I'd like to propose the following elegant alternative: determine the coefficients of the associated Laguerre polynomial of order n, by determining the coefficients of the characteristic polynomial of its companion matrix:

    function [c] = Laguerre_coeff(n, alpha)

    i = 1:n;
    a = (2*i-1) + alpha;
    b = sqrt( i(1:n-1) .* ((1:n-1) + alpha) );
    CM = diag(a) + diag(b,1) + diag(b,-1);

    c = (-1)^n/factorial(n) * poly(CM);
    
    '''

    i = np.arange(1,n+1)
    a = (2*i-1) + alpha
    b = np.sqrt( i[0:n-1] * (np.arange(1,n) + alpha) )
    CM = np.diag(a) + np.diag(b,1) + np.diag(b,-1)

    if CM.size < 1: # CM is empty
        return np.array([1])
    else:
        c = (-1)**n/np.math.factorial(n) * np.poly(CM)
        return c
        
def laguerre(p,l,x):
    return np.polyval(laguerre_coeff(p,l),x)

def u_n(n,x,z,w0,lam=1064e-9):
    import numpy as np
    
    zR = np.pi*w0**2/lam
    k = 2*np.pi/lam
    w = w0*np.sqrt(1+(z/zR)**2)
    
    if z == 0:
        R = np.inf
    else:
        R = (z+np.spacing(0.0))*(1+(zR/z)**2)
    psi = np.arctan(z/zR)
    
    t1 = np.sqrt(np.sqrt(2/np.pi))
    t2 = np.sqrt(np.exp(1j*(2*n+1)*psi)/(2**n*np.math.factorial(n)*w))
    t3 = herm(n,np.sqrt(2)*x/w)
    a1 = 0 # -1j*k*z
    a2 = 1j*k*x**2/(2*R)
    a3 = x**2/w**2
    t4 = np.exp(a1 - a2 - a3)
    E = t1 * t2 * t3 * t4
    
    return E

def u_nm(x,y,z,w0,n=0,m=0,lam=1064e-9,gamma_x=0):
    return np.outer(u_n(m,y,z,w0,lam),u_n(n,x,z,w0,lam))
    
# def u_n_q(n, x, q, lam=1064e-9):
#     # siegmann eq 16.54
#     import numpy as np
#     factorial = np.math.factorial
    
#     q0 = q - np.real(q)
#     w0 = np.sqrt(lam*np.imag(q0)/np.pi)
#     w = np.sqrt(-lam/(np.pi*np.imag(1/q)))
#     k = 2*np.pi/lam
    
#     t1 = np.sqrt(np.sqrt(2/np.pi))
#     t2 = np.sqrt(1.0/(2.0**n*factorial(n)*w0))
#     t3 = np.sqrt(q0/q)
#     t4 = (q0/np.conj(q0) * np.conj(q)/q)**(n/2)
#     t5 = herm(n,np.sqrt(2)*x/w) * np.exp(-1j*k*x**2/(2*q))
    
#     E = t1 * t2 * t3 * t4 * t5
    
#     return E
    
def u_n_q(x, q, n=0, lam=CONSTANTS['lambda'], gamma=0, include_gouy=True):
    '''
    1D HG electric field amplitude taken from Siegmann eq 16.54.

    If include_gouy=False then the Gouy phase from Siegmann's formula is not computed and 
    the phase of the beam at the origin x=0 should always be 0 (i.e. it is purely real).
    Useful if you want to do your own Gouy phase propagation.
    '''
    # 
    lam = complex(lam)
    w = q2w(q, lam=lam)
    k = 2*np.pi/lam
    tilt_map = maps.tilt_map_1D(x, xbeta=gamma, lam=lam)
    
    norm = gauss_norm(n, q, lam=lam, include_gouy=include_gouy)
    u = herm(n, np.divide.outer(np.sqrt(2)*x,w)) * np.exp(-1j*k*np.divide.outer(x**2,(2*q)))

    # print(norm.shape, u.shape, tilt_map.shape)

    E = (u.T * tilt_map).T * norm
    
    return E
    
def u_nm_q(x, y=None, qx=1j, qy=None, n=0, m=0, lam=1064e-9, include_gouy=True):
    '''
    2D HG electric field amplitude obtained by computing the outer product of the 1D
    HG distributions in the x and y axes.

    If include_gouy=False then the Gouy phase from Siegmann's formula is not computed and 
    the phase of the beam at the origin x=0 should always be 0 (i.e. it is purely real).
    Useful if you want to do your own Gouy phase propagation.
    '''
    if y is None:
        y = x
    if qy is None:
        qy = qx
    if np.ndim(x) == 1 and np.ndim(x) == 1:
        return np.outer(u_n_q(y, qy, m, lam, include_gouy=include_gouy), u_n_q(x, qx, n, lam, include_gouy=include_gouy))       
    else:
        return u_n_q(y, qy, m, lam, include_gouy=include_gouy) * u_n_q(x, qx, n, lam, include_gouy=include_gouy)

def i_n_w(x, w, n=0):
    '''
    1D normalized HG intensity distribution
    '''
    norm = np.sqrt(2/np.pi) * 1.0/(2.0**n*np.math.factorial(n)*w)
    I = norm * herm(n, np.sqrt(2)*x/w)**2 * np.exp(-2*x**2/w**2)
    return np.real(I)

def i_nm_w(x, y=None, wx=None, wy=None, n=0, m=0):
    '''
    2D normalized HG intensity distribution using the separable HG definition
    '''
    if y is None:
        y = x
    if wy is None:
        wy = wx
    ix = i_n_w(x, wx, n)
    iy = i_n_w(y, wy, m)
    i2 = np.outer(iy,ix)
    return i2
        
def u_pl(p,l,r,phi,z,w0,lam=1064e-9):
    import numpy as np
    factorial = np.math.factorial

    N = 2*p + np.abs(l)
    psi_z = (N+1)*np.arctan(z/zR(w0,lam))
    w = w_z(w0,z,lam)
    k = 2*np.pi/lam
    C_LG_lp = np.sqrt(2*factorial(p)/(np.pi*factorial(p+np.abs(l))))
    
    t1 = C_LG_lp/w
    t2 = (r*np.sqrt(2)/w)**(np.abs(l))
    t3 = np.exp(-r**2/w**2)
    t4 = laguerre(p,np.abs(l),2*r**2/w**2)
    t5 = np.exp(-1j*k*r**2/(2*R_z(w0,z,lam)))
    t6 = np.exp(1j*(-k*z + psi_z))
    
    return np.multiply.outer(t1*t2*t3*t4*t5*t6,np.exp(-1j*l*phi))
    
def zR(w0, lam=1064e-9):
    return np.pi * w0**2 / lam
    
def w_z(w0, z, lam=1064e-9):
    return w0*np.sqrt(1+(z/zR(w0,lam))**2)
    
def R_z(w0, z, lam=1064e-9):
    return z*(1+(zR(w0,lam)/z)**2)
     
def q_create(z, w0, lam=1064e-9):
    return z + 1j*np.pi*w0**2/lam
    
def q_create2(R, w, lam=1064e-9):
    return (1/R - 1j*lam/(np.pi*w**2))**-1
    
def q2gouy(q, n=0, m=0):
    z = np.real(q)
    zR = np.imag(q)
    psi = np.arctan(z/zR)
    return np.exp(1j*((n+1/2))*psi) * np.exp(1j*((m+1/2))*psi)
    # return np.sqrt(np.exp(1j*(2*n+1))*psi) * np.sqrt(np.exp(1j*(2*m+1))*psi)

def q2gouy_n(q, n=0):
    n = np.array(n)
    z = np.real(q)
    zR = np.imag(q)
    psi = np.arctan(z/zR)
    return np.exp(1j*((n+1/2))*psi)

def q2gouy_nm(q, n=0, m=0):
    z = np.real(q)
    zR = np.imag(q)
    psi = np.arctan(z/zR)
    return np.exp(1j*((n+1/2))*psi) * np.exp(1j*((m+1/2))*psi)

def q2guoy(*args, **kwargs):
    '''
    THIS IS NOT HOW YOU SPELL GOUY.
    This only kept here to prevent old code from breaking.
    '''
    print('use q2gouy instead')
    return q2gouy(*args, **kwargs)
    
def q2w(q, lam=1064e-9):
    '''
    Get beam size from q parameter.

    Alternative forms:
    np.sqrt(-lam/(np.pi*np.imag(1/q)))
    np.sqrt(lam/np.pi) * np.sqrt(q*np.conj(q)/(np.imag(q)))
    np.sqrt(lam/np.pi) * abs(q)/np.sqrt(zr)
    w0 * np.abs(q)/zr
    w0 * np.sqrt(1 + (z/zr)**2)
    '''
    w0 = q2w0(q, lam=lam)
    zr = np.imag(q)
    return w0 * np.abs(q)/zr

def q2w0(q, lam=1064e-9):
    '''
    Get waist size from q parameter.
    '''
    zr = np.imag(q)
    return np.sqrt(zr * lam / np.pi)

@np.vectorize
def q2R(q):
    '''
    Alternate forms:
    np.abs(q)**2/np.real(q)
    np.real(1/q)
    '''
    if np.real(q) == 0:
        return np.inf
    else:
        return 1/np.real(1/q)

def q2zzr(q):
    return np.array([q.real, q.imag])

def q2Theta(q, lam=1064e-9):
    w0 = q2w0(q, lam)
    return lam/(np.pi*w0)
    
def accum_guoy(q,d):
    '''
    Calculates the amount of guoy phase that is accumilated by a TEM00
    Gaussian beam with basis q as it propagates a length d. Originally written
    to calculate guoy phase shifts in LCT cavity scans.
    '''
    p0 = q2guoy(q)
    p1 = q2guoy(q+d)
    p2 = np.conj(p0)*p1
    return p2

def DHT(F, x, y, z, w0, lam=1064e-9, maxorder=None, mode_list=None, type='Riemann'):
    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])

    # legacy
    if not maxorder is None:
        N = maxorder+1
        H = np.zeros([N,N],dtype=np.complex128)

        for ii in range(N):
            for jj in range(N):
                herm = u_nm(x,y,z,w0,ii,jj,lam)
                if type == 'Simpson':
                    H[ii,jj] = intgr.simps(intgr.simps(F*np.conj(herm)))*dx*dy
                elif type == 'Riemann':
                    H[ii,jj] = np.sum((F*np.conj(herm)))*dx*dy           
        return H
        
    # new: mode_lists
    assert(not mode_list is None)
    mode_arr = np.array(mode_list)
    arr_shape = mode_arr.shape
    
    if len(arr_shape) == 3: # N x M x 2
        H = np.zeros([arr_shape[0],arr_shape[1]],dtype=np.complex128)
        for ii in mode_arr:
            for jj in ii:
                n,m = jj
                herm = u_nm(x,y,z,w0,n,m,lam)
                if type == 'Simpson':
                    H[n,m] = intgr.simps(intgr.simps(F*np.conj(herm)))*dx*dy
                elif type == 'Riemann':
                    H[n,m] = np.sum((F*np.conj(herm)))*dx*dy 
    elif len(arr_shape) == 2: # N x 2
        H = np.zeros([arr_shape[0]],dtype=np.complex128)
        for j,ii in enumerate(mode_arr):
            n,m = ii
            herm = u_nm(x,y,z,w0,n,m,lam)
            if type == 'Simpson':
                H[j] = intgr.simps(intgr.simps(F*np.conj(herm)))*dx*dy
            elif type == 'Riemann':
                H[j] = np.sum((F*np.conj(herm)))*dx*dy  
                
    return H              
    
    
def DHT_q(F, x, y, q, lam=1064e-9, maxorder=None, mode_list=None, type='Riemann'):
    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])

    # legacy
    if not maxorder is None:
        N = maxorder+1
        H = np.zeros([N,N],dtype=np.complex128)

        for ii in range(N):
            for jj in range(N):
                herm = u_nm_q(x,y,q,ii,jj,lam)
                if type == 'Simpson':
                    H[ii,jj] = intgr.simps(intgr.simps(F*np.conj(herm)))*dx*dy
                elif type == 'Riemann':
                    H[ii,jj] = np.sum((F*np.conj(herm)))*dx*dy           
        return H
        
    # new: mode_lists
    assert(not mode_list is None)
    mode_arr = np.array(mode_list)
    arr_shape = mode_arr.shape
    
    if len(arr_shape) == 3: # N x M x 2
        H = np.zeros([arr_shape[0],arr_shape[1]],dtype=np.complex128)
        for ii in mode_arr:
            for jj in ii:
                n,m = jj
                herm = u_nm_q(x,y,q,n,m,lam)
                if type == 'Simpson':
                    H[n,m] = intgr.simps(intgr.simps(F*np.conj(herm)))*dx*dy
                elif type == 'Riemann':
                    H[n,m] = np.sum((F*np.conj(herm)))*dx*dy 
    elif len(arr_shape) == 2: # N x 2
        H = np.zeros([arr_shape[0]],dtype=np.complex128)
        for j,ii in enumerate(mode_arr):
            n,m = ii
            herm = u_nm_q(x,y,q,n,m,lam)
            if type == 'Simpson':
                H[j] = intgr.simps(intgr.simps(F*np.conj(herm)))*dx*dy
            elif type == 'Riemann':
                H[j] = np.sum((F*np.conj(herm)))*dx*dy  
    return H  

def BH_DHT_1D(q1,q2,delta=0,gamma=0,maxtem=None):
    # Bayer-Helms DHT (mostly for debugging Bayer-Helms)
     
    N = maxtem+1
    H = np.zeros([N,N],dtype=np.complex128)

    for ii in range(N):
        for jj in range(N):
            H[jj,ii] = k_fun(ii,jj,q1,q2,dx=delta,gamma=gamma)
            
    return H
    
def BH_DHT(q1,q2,n1=0,m1=0,dx=0,dy=0,gammax=0,gammay=0,maxtem=None,mode_list=None):
    # Bayer-Helms DHT (mostly for debugging Bayer-Helms)
     
    # legacy
    if not maxtem is None: 
        N = maxtem+1
        H = np.zeros([N,N],dtype=np.complex128)

        for ii in range(N):
            for jj in range(N):
                H[jj,ii] = k_fun(n1,jj,q1,q2,dx=dx,gamma=gammax)*k_fun(m1,ii,q1,q2,dx=dy,gamma=gammay)
                
        return H
    else:
        assert(not mode_list is None)
        mode_arr = np.array(mode_list)
        arr_shape = mode_arr.shape
        
        H = np.zeros([arr_shape[0]],dtype=np.complex128)
        for j,ii in enumerate(mode_arr):
            n2,m2 = ii
            H[j] = k_fun(n1,n2,q1,q2,dx=dx,gamma=gammax)*k_fun(m1,m2,q1,q2,dx=dy,gamma=gammay)
        return H
    
def BH_DHT_astig(q1x,q1y,q2x,q2y,n1=0,m1=0,dx=0,dy=0,gammax=0,gammay=0,maxtem=10):
    # Bayer-Helms DHT (mostly for debugging Bayer-Helms)
    N = maxtem+1
    H = np.zeros([N,N],dtype=np.complex128)   

    for ii in range(N):
        for jj in range(N):
            H[jj,ii] = k_fun(n1,jj,q1x,q2x,dx=dx,gamma=gammax)*k_fun(m1,ii,q1y,q2y,dx=dy,gamma=gammay)
            
    return H  

def BH_knm(q1x,q1y,q2x,q2y,dx=0,dy=0,gammax=0,gammay=0,maxtem=10):
    m_list = mode_list2(maxtem)
    N = len(m_list)
    H = np.zeros([N,N],dtype=np.complex128)

    for i,nm1 in enumerate(m_list):
        for j,nm2 in enumerate(m_list):
            n1,m1 = nm1
            n2,m2 = nm2
            H[i,j] = k_fun(n1,n2,q1x,q2x,dx=dx,gamma=gammax)*k_fun(m1,m2,q1y,q2y,dx=dy,gamma=gammay)
    return H         

def iDHT(H, x, y, z, w0, lam=1064e-9):
    N = len(H)
    F = 0
    for m in range(N):
        for n in range(N):
            F = F + H[m,n]*u_nm(x,y,z,w0,n,m,lam)
    return F

def iDHT_q(H, x, y, q, lam=1064e-9, mode_list=None):
    F = 0
    
    # legacy
    if mode_list is None:
        N = len(H)
        for m in range(N):
            for n in range(N):
                F = F + H[m,n]*u_nm_q(x,y,q,n,m,lam)
    else:
        for h,jj in zip(H,mode_list):
            n,m = jj
            F = F + h*u_nm_q(x,y,q,n,m,lam)
    return F

def parse_DHT(H,power=False,sortby='magnitude'):
    out_list = []
    for n in range(len(H)):
        for m in range(len(H)):
            out_list.append([abs(H[n,m]),np.angle(H[n,m],deg=True),n,m])
    df = pd.DataFrame(out_list)
    df.columns = ['abs','deg','n','m']
    if sortby == 'magnitude':
        df = df.sort_values('abs',ascending=False).reset_index(drop=True)
    elif sortby == 'n,m':
        df = df.sort_values(by=['m','n'],ascending=True).reset_index(drop=True)
    if power:
        df['abs'] = df['abs']**2
        df.columns = ['power','deg','n','m']
    return df
    
def maxtem_scan(F, x, y, z, w0, lam=1064e-9, maxtem=None):
    iDHT_s = lambda H: iDHT(H, x, y, z, w0, lam)
    DHT_s = lambda F, maxtem: DHT(F,x,y,z,w0,lam,maxtem)
    dx = np.abs(x[0] - x[1])
    dy = np.abs(y[0] - y[1])
    maxtems = np.arange(0,maxorder+1,1)
    out_list = []
    for maxtem in maxtems:
        residual = iDHT_s(DHT_s(F,maxtem)) - F
        res_int = np.sum(np.abs(residual)**2)*dx*dy
        out_list.append([maxtem,res_int])
    return out_list
    
def radial_zernike(n,m,r):
    
    r = np.asarray(r)
    scalar_input = False
    if r.ndim == 0:
        r = r[None]  # Makes x 1D
        scalar_input = True

    # The magic happens here
    
#     r[r>1] = np.nan
    
    fac = np.math.factorial
    t = 0
    if np.mod(n-m,2) == 0:
        for k in range((n-m)//2+1):
            num = ((-1)**k * fac(n-k))
            denom = (fac(k)*fac((n+m)/2-k)*fac((n-m)/2-k))
            t += num/denom * r**(n-2*k)  
        return t
        if scalar_input:
            return np.squeeze(t)
        else: return t
    else:
        return np.zeros(np.shape(r))
    
def zernike(n,m,r,phi):

    if m >= 0:
#         return np.einsum('i,j->ij',radial_zernike(n,m,r),np.cos(m*phi))
        return np.multiply.outer(radial_zernike(n,m,r),np.cos(m*phi))
    else:
#         return np.einsum('i,j->ij',radial_zernike(n,-m,r),np.sin(m*phi))
        return np.multiply.outer(radial_zernike(n,-m,r),np.sin(m*phi))

def finesse(R1, R2=None):
    if R2 is None:
        R2 = R1
    F = np.pi/2 * (np.arcsin((1 - np.sqrt(R1*R2))/(2*np.sqrt(np.sqrt(R1*R2)))))**-1
    return F

def fsr(L):
    c = 299792458
    return c/(2*L)

def fwhm(L, R1, R2=None):
    return fsr(L)/finesse(R1, R2)
        
def DZT(F, r, phi, maxorder):
    # Discrete Zernike Transform
    # n -> [0:N]
    # m -> [-N:N]
    N = maxorder+1
    H = np.zeros([N,N*2-1])
    dr = abs(r[1] - r[0])
    dphi = abs(phi[1] - phi[0])

    for n in range(N):
        for m in range(-n,n+1):
#             print(ii,jj,np.shape(r),np.shape(phi))
            if m == 0:
                epsilon_m = 2
            else:
                epsilon_m = 1            
            norm_fac = (2*n+2)/(np.pi*epsilon_m)
#             norm_fac = 1
            
            test_fun = zernike(n,m,r,phi)
            H[n,m] = intgr.simps(intgr.simps(F*np.conj(test_fun)*r[:,None]))*dr*dphi * norm_fac
            
    return H

def iDZT(Z,r,phi):
    N = np.shape(Z)[0]
    F = 0
    for n in range(N):
        for m in range(-n,n+1):
            F = F + Z[n,m]*zernike(n,m,r,phi)            
    return F
    
def parse_DZT(Z):
    N = np.shape(Z)[0]
    out_list = []
    for n in range(N):
        for m in range(-n,n+1):
            out_list.append([abs(Z[n,m]),n,m])
    df = pd.DataFrame(out_list)
    df.columns = ['abs','n','m']
    return df.sort_values('abs',ascending=False)
    
# numeric version of coupling coeffs
def k_fun(n1,n2,q1,q2,gamma=0,dx=0,lam=1064e-9):
    
    import numpy as np
    fac = np.math.factorial
    sqrt = np.lib.scimath.sqrt
    
    # rip out subcomponents of q
    z1 = np.real(q1)
    z2 = np.real(q2)
    zR1 = np.imag(q1)
    zR2 = np.imag(q2)
    w01 = sqrt(lam*zR1/np.pi)
    w02 = sqrt(lam*zR2/np.pi)
    
    # Bayer-Helms sub terms
    K2 = (z1-z2)/zR2
    K0 = (zR1 - zR2)/zR2  
    K = 1j*np.conj(q1-q2)/(2*np.imag(q2))
    K = (K0 + 1j*K2)/2
    #X_bar = (1j*zR2 - z2)*np.sin(gamma)/(sqrt(1+np.conj(K))*w0)
    #X = (1j*zR2 + z2)*np.sin(gamma)/(sqrt(1+np.conj(K))*w0)
    X_bar = (dx/w02-(z2/zR2 - 1j)*np.sin(gamma)*zR2/w02)/(sqrt(1+np.conj(K)))
    X = (dx/w02-(z2/zR2 + 1j*(1+2*np.conj(K)))*np.sin(gamma)*zR2/w02)/(sqrt(1+np.conj(K)))
    F_bar = K/(2*(1+K0))
    F = np.conj(K)/2
    
    E_x = np.exp(-(X*X_bar)/2 - 1j*dx/w02 * np.sin(gamma)*zR2/w02)
    # note that there is a typo in BH paper for the E_x term where he uses w02 (ie bar(w0))
    # in the 1j*dx/w0 term instead of w01 (ie w0 without a bar)
    # upon closer inspection it looks like BH agrees with Riemann if the last index matches 
    # that of the term multiplying it 
    
    # i.e. 
    # 1j*dx/w02 * np.sin(gamma)*zR2/w02 (choose this one arbitrarily)
    # or 
    # 1j*dx/w01 * np.sin(gamma)*zR1/w01
    # give identical results equivelant with Riemann 
    
    # whereas
    # 1j*dx/w02 * np.sin(gamma)*zR1/w01 (orginal BH)
    # or
    # 1j*dx/w01 * np.sin(gamma)*zR2/w02
    # do not
    
    def S_g(n1,n2):
        s1 = 0
        for mu1 in range(0, (n1//2 if n1 % 2 == 0 else (n1-1)//2) + 1):
            for mu2 in range(0, (n2//2 if n2 % 2 == 0 else (n2-1)//2) + 1):
                t1 = ((-1)**mu1*X_bar**(n1-2*mu1)*X**(n2-2*mu2))/(fac(n1-2*mu1)*fac(n2-2*mu2))
                s2 = 0
                for sigma in range(0,min(mu1,mu2)+1):
                    s2 += ((-1)**sigma * F_bar**(mu1-sigma)*F**(mu2-sigma))/(fac(2*sigma)*fac(mu1-sigma)*fac(mu2-sigma))
    #                 print(mu1,mu2,sigma)
                s1 += t1 * s2
        return s1

    def S_u(n1,n2):
        s1 = 0
        for mu1 in range(0, ((n1-1)//2 if (n1-1) % 2 == 0 else ((n1-1)-1)//2) + 1):
            for mu2 in range(0, ((n2-1)//2 if (n2-1) % 2 == 0 else ((n2-1)-1)//2) + 1):
                t1 = ((-1)**mu1*X_bar**((n1-1)-2*mu1)*X**((n2-1)-2*mu2))/(fac((n1-1)-2*mu1)*fac((n2-1)-2*mu2))
                s2 = 0
                for sigma in range(0,min(mu1,mu2)+1):
                    s2 += ((-1)**sigma*F_bar**(mu1-sigma)*F**(mu2-sigma))/(fac(2*sigma+1)*fac(mu1-sigma)*fac(mu2-sigma))
                    # print(mu1,mu2,sigma)
                s1 += t1 * s2
        return s1
        
        
    # print('S_g = ', S_g(n1,n2))
    # print('S_u = ', S_u(n1,n2))
    
    expr = (-1)**n2 * E_x * sqrt(float(fac(n1)*fac(n2))) \
    * (1 + K0)**(n1/2 + 1/4) * (1+np.conj(K))**(-(n1+n2+1)/2) \
    * (S_g(n1,n2) - S_u(n1,n2))
    
    return expr

def k_nmnm(n1,m1,n2,m2,q1,q2,gammax=0,gammay=0,dx=0,dy=0):
    return k_fun(n1,n2,q1,q2,gamma=gammax,dx=dx) * k_fun(m1,m2,q1,q2,gamma=gammay,dx=dy)
    
def k_00(q1,q2,gamma=0,dx=0,lam=1064e-9):
    import numpy as np
    sqrt = np.lib.scimath.sqrt
    
    z1 = np.real(q1)
    z2 = np.real(q2)
    zR1 = np.imag(q1)
    zR2 = np.imag(q2)
    w01 = sqrt(lam*zR1/np.pi)
    w02 = sqrt(lam*zR2/np.pi)
    
    # Bayer-Helms sub terms
    K2 = (z1-z2)/zR2
    K0 = (zR1 - zR2)/zR2  
    K = 1j*np.conj(q1-q2)/(2*np.imag(q2))
    K = (K0 + 1j*K2)/2
    #X_bar = (1j*zR2 - z2)*np.sin(gamma)/(sqrt(1+np.conj(K))*w0)
    #X = (1j*zR2 + z2)*np.sin(gamma)/(sqrt(1+np.conj(K))*w0)
    X_bar = (dx/w02-(z2/zR2 - 1j)*np.sin(gamma)*zR2/w02)/(sqrt(1+np.conj(K)))
    X = (dx/w02-(z2/zR2 + 1j*(1+2*np.conj(K)))*np.sin(gamma)*zR2/w02)/(sqrt(1+np.conj(K)))
    F_bar = K/(2*(1+K0))
    F = np.conj(K)/2
    
    E_x = np.exp(-(X*X_bar)/2 - 1j*dx/w02 * np.sin(gamma)*zR2/w02)
    
    return E_x*(1+K0)**(1/4.0)*(1+np.conj(K))**(-1/2.0)
    
def k_01(q1,q2,gamma=0,dx=0,lam=1064e-9):
    import numpy as np
    sqrt = np.lib.scimath.sqrt
    
    z1 = np.real(q1)
    z2 = np.real(q2)
    zR1 = np.imag(q1)
    zR2 = np.imag(q2)
    w01 = sqrt(lam*zR1/np.pi)
    w02 = sqrt(lam*zR2/np.pi)
    
    # Bayer-Helms sub terms
    K2 = (z1-z2)/zR2
    K0 = (zR1 - zR2)/zR2  
    K = 1j*np.conj(q1-q2)/(2*np.imag(q2))
    K = (K0 + 1j*K2)/2
    #X_bar = (1j*zR2 - z2)*np.sin(gamma)/(sqrt(1+np.conj(K))*w0)
    #X = (1j*zR2 + z2)*np.sin(gamma)/(sqrt(1+np.conj(K))*w0)
    X_bar = (dx/w02-(z2/zR2 - 1j)*np.sin(gamma)*zR2/w02)/(sqrt(1+np.conj(K)))
    X = (dx/w02-(z2/zR2 + 1j*(1+2*np.conj(K)))*np.sin(gamma)*zR2/w02)/(sqrt(1+np.conj(K)))
    F_bar = K/(2*(1+K0))
    F = np.conj(K)/2
    
    E_x = np.exp(-(X*X_bar)/2 - 1j*dx/w02 * np.sin(gamma)*zR2/w02)
    
    return -E_x * (1+K0)**(1/4.0) * (1+np.conj(K))**(-1) * (X)
    
def k_02(q1,q2,gamma=0,dx=0,lam=1064e-9):
    import numpy as np
    sqrt = np.lib.scimath.sqrt
    
    z1 = np.real(q1)
    z2 = np.real(q2)
    zR1 = np.imag(q1)
    zR2 = np.imag(q2)
    w01 = sqrt(lam*zR1/np.pi)
    w02 = sqrt(lam*zR2/np.pi)
    
    # Bayer-Helms sub terms
    K2 = (z1-z2)/zR2
    K0 = (zR1 - zR2)/zR2  
    K = 1j*np.conj(q1-q2)/(2*np.imag(q2))
    K = (K0 + 1j*K2)/2
    #X_bar = (1j*zR2 - z2)*np.sin(gamma)/(sqrt(1+np.conj(K))*w0)
    #X = (1j*zR2 + z2)*np.sin(gamma)/(sqrt(1+np.conj(K))*w0)
    X_bar = (dx/w02-(z2/zR2 - 1j)*np.sin(gamma)*zR2/w02)/(sqrt(1+np.conj(K)))
    X = (dx/w02-(z2/zR2 + 1j*(1+2*np.conj(K)))*np.sin(gamma)*zR2/w02)/(sqrt(1+np.conj(K)))
    F_bar = K/(2*(1+K0))
    F = np.conj(K)/2
    
    E_x = np.exp(-(X*X_bar)/2 - 1j*dx/w02 * np.sin(gamma)*zR2/w02)
    
    return E_x * 2**(-1/2.0) * (1+K0)**(1/4.0)*(1+np.conj(K))**(-3/2.0) * (X**2 + 2*F_bar)
    
def mode_mismatch(q1,q2):
    return np.abs((q2-q1)/(q2-np.conj(q1)))**2
    
def multigauss_overlap(q1,q2):
    return 1/2 + 1/4*k_nmnm(0,0,0,0,q1,q2) + 1/4*k_nmnm(0,0,0,0,q2,q1)
    
def multigauss_mismatch(q1,q2):
    # fractional power loss from adding two gaussian beams together
    return 1/2 - 1/4*k_nmnm(0,0,0,0,q1,q2) - 1/4*k_nmnm(0,0,0,0,q2,q1)
    
def mode_list(maxtem,even_only=False):
    modes_list = []
    for n in range(0,maxtem+1):
        for m in range(0,maxtem+1):
            if n+m <= maxtem:
                if even_only:
                    if np.mod(n,2)==0 and np.mod(m,2)==0 :
                        modes_list.append([n,m])
                else:
                    modes_list.append([n,m])
    return modes_list
    
def mode_list2(maxtem):
    '''This one counts up by mode order'''
    modes_list = []
    for i in range(0, maxtem+1):
         for j in range(0, i+1):
                mode = (i-j, j)
                modes_list.append(mode)
    return modes_list
    
def mode_list_maxorder(maxorder,linearise=False):
    '''
    maxorder is easier to plot but doesnt really mean anything
    '''
    N = maxorder + 1
    m_list = []
    for ii in range(N):
        temp = []
        for jj in range(N):
            if linearise:
                m_list.append([ii,jj])
            else:
                temp.append([ii,jj])
        if not linearise:
            m_list.append(temp)
    return m_list
    
def lin_mode_index(mode_list):
    lin_mode_list = []
    for n,m in mode_list:
        lin_mode_list.append((n+m) + m/(n+m+1))
    return lin_mode_list
    
def lin_mode_index2(mode_list):
    '''excludes mapping to boundaries'''
    lin_mode_list = []
    for n,m in mode_list:
        lin_mode_list.append((n+m) + (m+1)/(n+m+2))
    return lin_mode_list
    
def zr2m(zr1,M):
    return zr1*(2*np.sqrt(M) - M - 1)/(M - 1)

def zr2p(zr1,M):
    return -zr1*(2*np.sqrt(M) + M + 1)/(M - 1)

def z2m(zr1,z1,M):
    return (M*z1 + 2.0*zr1*np.sqrt(-M*(M - 1.0)) - z1)/(M - 1.0)

def z2p(zr1,z1,M):
    return (M*z1 - 2.0*zr1*np.sqrt(-M*(M - 1.0)) - z1)/(M - 1.0)

def mismatch_circle(q1, M=None, t=None, m=None):
    '''
    The natural parameterization. Equidistant in complex mismatch m-space.

    Alternative forms:
    q2 = np.conj(q1) + 1j*zr1 * 2/(1 + 1j*m*np.exp(1j*t))
    q2 = (q1 + 1j*m*np.conj(q1))/(1+1j*m)
    '''
    z1, zr1 = q1.real, q1.imag
    if t is None:
        t = np.random.random()*2*np.pi
    if m is None:
        m = np.sqrt(M)*np.exp(1j*t) # same m as in complex_mismatch(q1, q2)
    q2 = np.conj(q1) + 1j*zr1 * 2/(1 + 1j*m)
    return q2

def mismatch_circle2(q1, M, t):
    '''
    The q-space equidistant parametrization. A comparison between the different parametrizations can be given by

    import optics_funs as of
    import numpy as np
    import matplotlib.pyplot as plt

    q1 = 1j
    M = 0.4
    t = np.linspace(0, 2*np.pi, 30)[:-1]

    qc1 = of.mismatch_circle(q1,M,t)
    qc2 = of.mismatch_circle2(q1,M,t)
    m1 = of.complex_mismatch(q1, qc)
    m2 = of.complex_mismatch(q1, q2)

    plt.figure()
    plt.plot(qc1.real,qc1.imag,'x')
    plt.plot(qc2.real,qc2.imag,'.')
    plt.axis('image')

    plt.figure()
    plt.plot(m1.real,m1.imag,'x')
    plt.plot(m2.real,m2.imag,'.')
    plt.axis('image')
    '''
    z1, zr1 = q1.real, q1.imag
    m = np.sqrt(M)
    scale = -(1+m*(m-2j*np.exp(1j*t)))/(m**2-1)
    q2 = z1 + 1j*zr1 * scale
    return q2

def complex_mismatch(q1, q2):
    '''
    The inverse of mismatch_circle
    '''
    m = 1j*(q2-q1)/(q2-np.conj(q1))
    return m
    
def gauss_intens(x, y, wx=1, wy=1, x0=0, y0=0, a=1, floor=0, theta=0):
    "the integral of this is normalized to 1"
    norm = 2/(np.pi*wx*wy)
    Ix = np.exp(-2*x**2/wx**2)
    Iy = np.exp(-2*y**2/wy**2)
    Im = norm * np.outer(Iy,Ix)
    return Im

def gauss_spot_propag(z, z0, w0, M_2=1, lam=1064e-9):
    return np.sqrt(w0**2 + M_2**2 * (lam/(np.pi*w0))**2 * (z-z0)**2)
    
def b_nm(n,m,a,b):

    def gee(n,m,x):
        '''a part of an iterative analytical solution to an indefinite integral 
        of a product of two hermite polynomials and a gaussian'''
        return -herm(n,x)*herm(m,x)*np.exp(-x**2)

    # make sure n >= m always
    if n < m:
        n,m = m,n
    
    s = 0
    for i in range(0,m+1):
        s += 2**i * np.math.factorial(m)/np.math.factorial(m-i) * (gee(n-1-i,m-i,b)-gee(n-1-i,m-i,a))
    return s
    
def gauss_norm(n, q, lam=CONSTANTS['lambda'], include_gouy=True):
    '''
    The normalization factor for a 1D HG electric field distribution to ensure
    that the overlap integral equates to 1.

    Traditionally the normalization includes a Gouy phase factor for free space 
    propagation but that can be turned off by setting include_gouy=False
    '''
    lam = complex(lam)

    zr = np.imag(q)
    w0 = q2w0(q, lam=lam)
    w  = q2w(q, lam=lam)
    
    t1 = np.sqrt(np.sqrt(2/np.pi))
    t2 = np.sqrt(1.0/(2.0**n*scipy.special.factorial(n)*w0))
    if include_gouy:
        t3 = np.sqrt(1j*zr/q)
        t4 = (-np.conj(q)/q)**(n/2)
    else:
        t3,t4 = np.sqrt(w0/w),1
    
    return t1*t2*t3*t4
    
def c_nm(n,m,q,a,b,lam=1064e-9):
    w = q2w(q,lam=lam)
    
    # we integrate with respect to dx_bar = sqrt(2)/w * dx
    # so change the limits of integration to be in barred units
    
    a_bar = a*np.sqrt(2)/w
    b_bar = b*np.sqrt(2)/w
    
    # the factor of w/np.sqrt(2) comes from the fact we changed a dx into a dx_bar
    return gauss_norm(n,q,lam) * np.conj(gauss_norm(m,q,lam)) * b_nm(n,m,a_bar,b_bar) * w/np.sqrt(2)
    
def g_rt(rs,phi):
    return np.product(rs)*np.exp(-1j*phi)

def E_trans(r1,r2,phi):
    t1 = np.sqrt(1-r1**2)
    t2 = np.sqrt(1-r2**2)
    return -t1*t2/np.sqrt(r1*r2) * np.sqrt(g_rt([r1,r2],phi))/(1-g_rt([r1,r2],phi))

def epsilon_q(q1,q2):
    zR = (np.imag(q1) + np.imag(q2))/2
    return (q2-q1)/zR

def epsilon_z(q1,q2):
    return np.real(epsilon_q(q1,q2))

def epsilon_zR(q1,q2):
    return np.imag(epsilon_q(q1,q2))

def gamma2eps_gamma(gamma, q, lam=1064e-9):
    w = q2w(q, lam=lam)
    eps_gamma = gamma * w * np.pi / lam
    return eps_gamma

def eps_gamma2gamma(eps_gamma, q, lam=1064e-9):
    w = q2w(q, lam=lam)
    gamma = eps_gamma / w / np.pi * lam
    return gamma

def eps_delta2delta(eps_delta, q, lam=1064e-9):
    w0 = q2w0(q, lam=lam)
    delta = eps_delta * w0
    return delta

def delta2eps_delta(delta, q, lam=1064e-9):
    w0 = q2w0(q, lam=lam)
    eps_delta = delta/w0
    return eps_delta

def random_q(q0=1j, N=1, alpha=3, beta=10):
    M = scipy.stats.beta(alpha, beta).rvs(N)
    t = np.random.rand(N)*2*np.pi
    q1 = mismatch_circle(q0, M=M, t=t)
    return np.squeeze(q1)