import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import colorspacious
except ModuleNotFoundError as e:
    print(f'Warning: {e}, skipping')
    
try:
    import colour
except ModuleNotFoundError as e:
    print(f'Warning: {e}, skipping')

def rgb_color_wheel(N=301, ns=None, offset=0, reverse=False):
    '''
    Generate the rgb values for an HSV colorwheel.
    '''
    if ns is None:
        ns = np.linspace(0,1,N)
    h = (ns + offset) % 1
    if reverse:
        h = np.flipud(h)
    s = np.ones(N)
    v = np.ones(N)
    hsv = np.vstack([h,s,v]).T
    rgb = mpl.colors.hsv_to_rgb(hsv)
    return rgb

def bgr_colormap(N=301):
    '''
    Generate the rgb values for a bgr (blue-green-red) colormap from the HSV colorwheel.
    '''
    ns = np.linspace(0,2/3,N)[::-1]
    bgr_rgb = rgb_color_wheel(ns=ns)
    return bgr_rgb

def cull_invlaid_rgb(rgb_arr, cull_val=np.nan):
    '''
    Sets pixels with invalid rgb values to the cull value (default nan).
    Valid rgb values are between 0 or 1. This function is useful when doing colorspace
    conversions to eliminate colors as going from some colorspaces (e.g. XYZ, LAB) back
    to RGB can create invalid colors as the sRGB colorspace is quite small.
    
    Checks if any of the r,g,b channels contains an invalid value and if it does 
    kills the entire pixel.
    '''
    rgb_arr = rgb_arr.copy()
    r_nan = np.logical_or(rgb_arr[..., 0]<0, rgb_arr[..., 0]>1)
    g_nan = np.logical_or(rgb_arr[..., 1]<0, rgb_arr[..., 1]>1)
    b_nan = np.logical_or(rgb_arr[..., 2]<0, rgb_arr[..., 2]>1)
    rgb_nan = np.logical_or(np.logical_or(r_nan, g_nan), b_nan)
    rgb_arr[rgb_nan,:] = np.nan
    return rgb_arr

def apply_colormap(data_arr, colormap=mpl.cm.viridis, vmin=None, vmax=None, alpha=True):
    '''
    Get raw rgba values from a colormap.
    Useful if you want to set your own transparency.
    
    code adapted from https://stackoverflow.com/a/28147716/13161738
    '''
    norm = plt.Normalize(vmin, vmax)
    rgba_arr = colormap(norm(data_arr))
    if alpha:
        out = rgba_arr
    else:
        rgb_arr = rgba_arr[...,0:3]
        out = rgb_arr
    return out

def cmap_lab(cmap=plt.cm.viridis, N=301, ns=None):
    '''
    Compute the coordinates of a colormap in the CIECAM02 colorspace coordinates.
    The coordinates are given in terms of a 3-tuple of (L,a,b), where
    L: perceptual lightness
    a: green-red axis
    b: blue-yellow axis
    '''
    if ns is None:
        ns = np.linspace(0,1,N)
    rgb = apply_colormap(ns, colormap=cmap, alpha=False)
    lab = colorspacious.cspace_convert(rgb, "sRGB1", "CAM02-UCS")
    return lab

def complex_to_Lab(c_arr, C=0.2, L=0.808, L_expnt=1/3):
    '''Converts complex array into Lab colourspace
    
    c_arr.shape :: (..., 3)
    C : chroma scaling
    L : lightness scaling
    L_expnt : lightness exponent
    '''
    ampl = np.abs(c_arr)
    ampl_norm = np.max(ampl)

    l = L*(ampl/np.max(ampl))**L_expnt
    a = C*np.real(c_arr)/ampl_norm
    b = C*np.imag(c_arr)/ampl_norm
    
    lab_mat = np.stack([l,a,b], axis=-1)
    return lab_mat

def complex_to_rgb(c_arr, C=0.2, L=0.808, L_expnt=1/3, clip_rgb=True):
    '''Converts complex array into RGB using the CAM16-UCS model.
    
    c_arr.shape :: (..., 3)
    C : chroma scaling
    L : lightness scaling
    L_expnt : lightness exponent
    
    Example
    -----------
    import numpy as np
    import matplotlib.pyplot as plt
    import colour
    
    N1, N2 = (301, 300)
    A = np.linspace(0, 1, N1)
    t = np.linspace(0, 2*np.pi, N2)
    p = np.exp(1j*t)
    B = np.outer(A,p)

    rgb_mat = complex_to_rgb(B)
    plt.imshow(rgb_mat, extent=[0,360,0,1], aspect=100)
    plt.grid(False)
    '''
    import warnings
    
    lab_mat = complex_to_Lab(c_arr, C=C, L=L, L_expnt=L_expnt)
    rgb_mat = colour.convert(lab_mat, 'cam16ucs', 'rgb')
    
    if clip_rgb:
        rgb_threshold = 1/255
        rgb_min = np.min(rgb_mat)
        rgb_max = np.max(rgb_mat)
        if rgb_min < -rgb_threshold:
            warnings.warn(f'{rgb_min=} underflow clipped to 0')
        if rgb_max > 1+rgb_threshold:
            warnings.warn(f'{rgb_max=} overflow clipped to 1')
        rgb_mat = np.clip(rgb_mat, 0, 1)
        
    return rgb_mat