import matplotlib as mpl
import matplotlib.figure
import matplotlib.colors
import matplotlib.style

import matplotlib.pyplot as plt

import general_funs as gf
import numerical_funs as nf
import numpy as np

from .colors import apply_colormap
from .colormaps import cm
import PIL

import functools

import os
from pathlib import Path

mpl_rc_validators = mpl.rcsetup._validators

cwd = Path(os.path.realpath(__file__)).parent

def generate_mpl_default_style():
    '''https://stackoverflow.com/a/49495180/13161738
    '''
    path_to_rc = mpl.matplotlib_fname()
    with open(path_to_rc, "r") as f:
        rclines = f.readlines()
    newlines = []
    for line in rclines:
        if line[0] == "#":
            newline = line[1:]
        else:
            newline = line
        if "$TEMPLATE_BACKEND" in newline:
            newline = "backend : "+mpl.rcParams["backend"]
        if "datapath" in newline:
            newline = ""
        newlines.append(newline)

    with open("mynewstyle.mplstyle", "w") as f:
        f.writelines(newlines)

def style_use(stylename):
    full_path = cwd / ('styles/'+stylename+'.mplstyle')
    # print(full_path)
    return mpl.style.use(full_path)

style_use('default')

def style_context(stylename):
    if not isinstance(stylename, (list, tuple)):
        stylename = list(stylename)
    full_path = [cwd / ('styles/'+ x +'.mplstyle') for x in stylename]
    rcs = [mpl.rc_params_from_file(x, use_default_template=False) for x in full_path]
    rc_out = rcs[0]
    for i,rc in enumerate(rcs):
        rc_out.update(rc)
    # print(full_path)
    # return mpl.style.context(full_path)
    return mpl.rc_context(rc=rc_out) # slightly more flexible than mpl.style.context

nice_ticks = [0.1, 0.125, 0.2, 0.25, 0.5]
# nice_ticks = np.arange(0.01,1,0.01)

function_dictionary = {'real': np.real, 're': np.real, 'imag': np.imag, 'im': np.imag, 'abs': np.abs, 'pow': lambda x: np.abs(x)**2, 'abs2': lambda x: np.abs(x)**2, 'deg': lambda x: np.angle(x, deg=True), 'rad': np.angle}

label_dictionary = {'real': 'Real part', 're': 'Real part', 'imag': 'Imaginary part', 'im': 'Imaginary part', 'abs': 'Absolute value', 'pow': 'Power', 'abs2': 'Power', 'deg': 'Phase [deg]', 'rad': 'Phase [rad]'}

class MPFFigure:
    '''
    Doesn't work yet. Need to figure out how ipython/jupyter display() works.
    '''
    def __init__(self, *args):
        fig, ax, ln = args
        self.fig = fig
        self.ax = ax
        self.ln = ln

    def __repr__(self):
        return self.fig.__repr__()

    def _repr_html_(self):
        return self.fig._repr_html_()

def figure(*args, **kwargs):
    fig = mpl.figure.Figure(*args, **kwargs)
    return fig

def PIL_resize(arr, factor=2, method='nearest'):
    im = im = PIL.Image.fromarray(arr)
    if method == 'nearest':
        im2 = im.resize(np.flipud(arr.shape)*factor, resample=PIL.Image.NEAREST)
    elif method == 'bilinear':
        im2 = im.resize(np.flipud(arr.shape)*factor, resample=PIL.Image.BILINEAR)
    elif method == 'bicubic':
        im2 = im.resize(np.flipud(arr.shape)*factor, resample=PIL.Image.BICUBIC)
    elif method == 'lanczos':
        im2 = im.resize(np.flipud(arr.shape)*factor, resample=PIL.Image.LANCZOS)
    return np.asarray(im2)

def PIL_imsave_rgb(fname, rgb_arr):
    if rgb_arr.dtype == float:
        rgb_arr = np.uint8(rgb_arr*255)
    im = PIL.Image.fromarray(rgb_arr, 'RGBA')
    im.save(fname)
    return im

def PIL_imsave(fname, arr, scaling=1, scaling_method='nearest', cmap=mpl.cm.viridis, vmin=None, vmax=None):
    arr2 = PIL_resize(arr, scaling, scaling_method)
    rgb_arr = apply_colormap(arr2, cmap, vmin, vmax)
    im = PIL_imsave_rgb(fname, rgb_arr)
    return im

def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None, origin=None, scaling=1, dpi=100, *, metadata=None, pil_kwargs=None):
    '''
    Clone of mpl.image.imsave but applies some extra savefig parameters that the mpl version doesn't
    to get rid of remaining whitespace.
    '''
    if isinstance(fname, os.PathLike):
        fname = os.fspath(fname)
    if format is None:
        format = (Path(fname).suffix[1:] if isinstance(fname, str)
                  else mpl.rcParams["savefig.format"]).lower()
    fig = figure(dpi=dpi, frameon=False, figsize=[1,1])
    print(fig)
    arr = PIL_resize(arr, scaling, 'nearest')
    im = fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, resize=True)
    fig.savefig(fname, dpi=dpi, format=format, transparent=True, metadata=metadata, bbox_inches='tight', pad_inches=0)
    return fig, im

def plot_complex(x=None, y=None, axs=None, use_plt=True, label='', title='', mode1='re:im', mode2='plot:plot', **kwargs):
    if axs is None:
        if use_plt:
            fig = plt.gcf()
            axs = fig.axes
            if axs == []:
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)
                fig.set_size_inches(10*1.618, 5)
                axs = [ax1, ax2]
        else:
            fig, axs = subplots(1, 2, figsize=(10*1.618, 5))

    val1, val2 = mode1.split(':')
    plot_type1, plot_type2 = mode2.split(':')

    if y is None:
        getattr(axs[0], plot_type1)(function_dictionary[val1](x), label=label, **kwargs)
        getattr(axs[1], plot_type2)(function_dictionary[val2](x), label=label, **kwargs)
    else:
        getattr(axs[0], plot_type1)(x, function_dictionary[val1](y), label=label, **kwargs)
        getattr(axs[1], plot_type2)(x, function_dictionary[val2](y), label=label, **kwargs)

    axs[0].set_ylabel(label_dictionary[val1])
    axs[1].set_ylabel(label_dictionary[val2])
    axs[0].set_title(title)
    axs[1].set_title(title)
    return axs

def complex_scatter(z, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None, **kwargs):
    zr = np.real(z)
    zi = np.imag(z)
    plt.scatter(zr, zi, s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, edgecolors=edgecolors, plotnonfinite=plotnonfinite, data=data, **kwargs)
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')

def add_colorbar(ax, im, label=''):
    fig = ax.get_figure()
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(label)
    return cb

def imshow_complex(im, axs=None, use_plt=True, label='', title='', mode='re:im', size=4, sep=2.8, aspect='auto', interpolation='nearest'):
    if axs is None:
        if use_plt:
            fig, axs = plt.subplots(1, 2, figsize=(size*sep, size))
        else:
            fig, axs = subplots(1, 2, figsize=(size*sep, size))
    fig = axs[0].figure
    if mode == 're:im':
        im0 = axs[0].imshow(np.real(im), label=label, aspect=aspect, interpolation=interpolation)
        im1 = axs[1].imshow(np.imag(im), label=label, aspect=aspect, interpolation=interpolation)
        cb0 = add_colorbar(axs[0], im0)
        cb1 = add_colorbar(axs[1], im1)
        cb0.set_label('Real part')
        cb1.set_label('Imaginary part')
    elif mode == 'abs:deg':
        im0 = axs[0].imshow(np.abs(im), label=label, aspect=aspect, interpolation=interpolation)
        im1 = axs[1].imshow(np.angle(im, deg=True), label=label, aspect=aspect, interpolation='nearest', cmap='twilight')
        cb0 = add_colorbar(axs[0], im0)
        cb1 = add_colorbar(axs[1], im1)
        cb0.set_label('Absolute value')
        cb1.set_label('Phase [deg]')
    fig.suptitle(title)
    return axs

def matshow(M, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.imshow(np.rot90(M))

def matshow_complex(M, axs=None, use_plt=True, label='', title='', size=4, **kwargs):
    return imshow_complex(np.rot90(M), axs=axs, use_plt=use_plt, label=label, title=title, size=size, **kwargs)

@functools.wraps(plot_complex)
def complex_plot(*args, **kwargs):
    return plot_complex(*args, **kwargs)

@functools.wraps(imshow_complex)
def complex_imshow(*args, **kwargs):
    return imshow_complex(*args, **kwargs)

@functools.wraps(matshow_complex)
def complex_matshow(*args, **kwargs):
    return matshow_complex(*args, **kwargs)


def bare_figure_axes(*args, **kwargs):
    '''
    Stolen from https://stackoverflow.com/a/8218887/13161738

    Creates a figure and axes pair where the figure only contains the data (ie. no axis, ticks, labels, etc.)
    
    Example:
    data = np.random.randn(101,100)
    fig, ax = mpf.bare_figure_axes()
    im = ax.imshow(data, interpolation='lanczos')
    ax.axis('square')
    display(fig)
    fig.savefig('foo.png', dpi=300, bbox_inches='tight', pad_inches=0)
    '''
    fig = figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig, ax

def subplots(nrows=1, ncols=1, figsize=None, figscale=[8.1, 5], sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, use_plt=True, **fig_kw):
    '''
    Clone of plt.subplots for when you don't want to use plt because you suspect it's buggy and slow.
    
    It correctly selects the default size.
    '''
    if figsize is None:
        figsize = np.array([ncols, nrows]) * np.array(figscale)
    if use_plt:
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey, squeeze=squeeze, subplot_kw=subplot_kw, gridspec_kw=gridspec_kw)
    else:
        fig = mpl.figure.Figure(figsize=figsize, **fig_kw)
        ax = fig.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, squeeze=squeeze, subplot_kw=subplot_kw, gridspec_kw=gridspec_kw)
    return fig, np.atleast_1d(ax)

def subplots_square(*args, **kwargs):
    '''
    Makes subplots for square image.
    '''
    return subplots(*args, figscale=[5.5, 4], **kwargs)

def _axes_dispatcher(plot_str, *args, fig_kw={}, figsize=(5*1.618, 5), fast=True, debug=False, **kwargs):
    fig, ax = subplots(figsize=figsize, **fig_kw)
    if debug:
        print('args', args)
        print('kwargs', kwargs)
    ln = getattr(ax[0], plot_str)(*args, **kwargs)
    if fast:
        return fig
    else:
        return fig, ax, ln

def plot(*args, **kwargs):
    return _axes_dispatcher('plot', *args, **kwargs)

# def plot(*args, **kwargs):
#     return MPFFigure(*_axes_dispatcher('plot', *args, **kwargs))

def semilogx(*args, **kwargs):
    return _axes_dispatcher('semilogx', *args, **kwargs)

def semilogy(*args, **kwargs):
    return _axes_dispatcher('semilogy', *args, **kwargs)

def loglog(*args, **kwargs):
    return _axes_dispatcher('loglog', *args, **kwargs)

def pcolormesh(*args, **kwargs):
    return _axes_dispatcher('pcolormesh', *args, **kwargs)

def contour(*args, **kwargs):
    return _axes_dispatcher('contour', *args, **kwargs)

def contourf(*args, **kwargs):
    return _axes_dispatcher('contourf', *args, **kwargs)

def thesis_figsize(fig=None, scale=1):
    '''
    https://gist.github.com/bougui505/b844feb1bdfba3072ca92a6d6961f71b
    '''
    if fig is not None:
        ax = fig.axes
        ax0 = np.ravel(ax)[0]
        gs = ax0.get_gridspec()
        nrows, ncols = gs.get_geometry()
    else:
        nrows, ncols = 1,1

    fig_width_pt = 441.01773                        # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = (fig_width*ncols*scale, fig_height*nrows*scale)

    if fig is not None:
        fig.set_size_inches(*fig_size)
    return fig_size

def savefig(fig, figpath, /, *args, **kwargs):
    '''
    This one will make a new directory if it doesn't exist
    '''
    dirname, fname = os.path.split(figpath)
    if dirname == '':
        dirname = '.'
    if os.path.exists(dirname):
        pass
    else:
        os.makedirs(dirname)
    fig.savefig(figpath, *args, **kwargs)

def savefig_timestamped(fig, *args, timestamp=None, **kwargs):
    fname = args[0]
    if timestamp is None:
        timestamp = gf.get_timestamp()
    im_name = timestamp+'_'+fname
    # check if file with same name and timestamp exists
    # append copy to filename
    if os.path.exists(im_name):
        fname += '_copy'
        savefig_timestamped(fig, fname, *args[1:], timestamp=timestamp, **kwargs)
    else:
        fig.savefig(im_name, *args[1:], **kwargs)

def thesis_savefig(fig, figpath, /, scale=1.0, *args, **kwargs):
    thesis_figsize(fig, scale)
    figpath = Path(figpath).absolute()
    for ext in ['.png', '.pdf']:
        full_path = figpath.with_suffix(ext)
        print(full_path)
        savefig(fig, full_path, *args, **kwargs)
    return 0

def plot_colourmapped_lines(func, x, y, z, xlabel=None, ylabel=None, zlabel=None, cmap=plt.cm.Spectral_r, fig=None, **kwargs):
    pass
    if fig is None:
        fig, ax = plt.subplots(1,1)
    else:
        ax = fig.axes[0]
    norm = mpl.colors.Normalize(vmin=min(z), vmax=max(z))
    s_m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    s_m.set_array([])
    func = getattr(ax, func)
    for i, _y in enumerate(y.T):
        ln = func(x, _y, c=s_m.to_rgba(z[i]), **kwargs)
    cb = fig.colorbar(s_m, ax=ax)
    cb.set_label(zlabel)
    # cb.solids.set_rasterized(True)
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
    return ln

def get_symmetric_ticks():
    pass

def set_symmetric_yticks(ax, nsigfig=1, nticks=9, symm_point=0):
    lb, ub = ax.get_ybound()
    ticks = gf.symmetric_linspace(gf.floor_nsigfig(lb,nsigfig), gf.ceil_nsigfig(ub,nsigfig), nticks, symm_point=symm_point)
    out = ax.set_yticks(ticks)
    return ticks, out

def set_yticks(ax, nsigfig=1, nticks=9):
    ticks = np.linspace(gf.floor_nsigfig(ax.get_ybound()[0],nsigfig), gf.ceil_nsigfig(ax.get_ybound()[1],nsigfig), nticks)
    out = ax.set_yticks(ticks)
    return out

def forceAspect(ax=None, aspect=1):
    '''
    AFAIK matplotlib has no single tidy method/function for making a plot
    with square axes. This is designed to fill that hole.
    
    Works with any axes object as long as the xaxis and yaxis are defined, 
    which they are in most cases when there is data in the plot.
    
    So call this after the data is already in the axes.
    
    Confirmed to work on imshow and contourf. Likely works on all types
    of plots.
    '''
    if ax is None:
        ax = plt.gca()
        
    xl,xu = ax.get_xaxis().get_data_interval()
    yl,yu = ax.get_yaxis().get_data_interval()
    
    ax.set_aspect(abs((xu-xl)/(yu-yl))/aspect)
    
def imshow(x=None, y=None, z=None, ax=None, colorbar=True, clabel='', aspect='auto', *args, **kwargs):
    '''
    Just like imshow except it allows for the user to set the x and y axes
    unlike regular imshow which implicitly defines x and y as array indices.  
    '''
    if ax is None:
        fig = plt.gcf()
        ax = plt.gca()
    if y is None and z is None:
        # imitate behaviour of regular imshow
        z = x
        z_shape = np.shape(z)
        x = np.arange(z_shape[1])
        y = np.arange(z_shape[0])
    elif y is not None and z is None:
        raise Exception('not enough arguments')
    if z.dtype == np.complex128:
        print('WARNING: complex data not supported, plotting real part only')
        z = np.real(z)
    if np.ndim(z) == 3:
        rgb_imshow = True
    else:
        rgb_imshow = False
    xl, xu = x[0], x[-1]
    yl, yu = y[0], y[-1]
    im = ax.imshow(z, extent=[xl,xu,yl,yu], aspect=aspect, **kwargs)
    if colorbar and not rgb_imshow:
        cb = add_colorbar(ax, im, label=clabel)
        return im, cb
    else:
        return im

def matshow_with_axes(x=None, y=None, z=None, *args, **kwargs):
    if y is None and z is None:
        z = np.rot90(x)
        Ny, Nx = np.shape(z)
        x = np.arange(Nx)
        y = np.arange(Ny)[::-1]
    elif y is not None and z is None:
        raise Exception('not enough arguments')
    else:
        z = np.rot90(z)
        y = np.flipud(y)
    return imshow(x=x, y=y, z=z, *args, **kwargs)

def axvlines(xs, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    for x in xs:
        ax.axvline(x, **kwargs)

def axrline(ax, theta=0, r=1, **kwargs):
    '''
    Draw radial line of length r and at angle theta in plot data units
    '''
    c = r*np.exp(1j*theta)
    xmax = np.real(c)
    ymax = np.imag(c)
    l = plt.Line2D([0, xmax], [0, ymax], **kwargs)
    ax.add_line(l)

def get_axes_from_fig(fig):
    '''
    Like fig.axes but strips off the wrapping list or tuple if 
    it only has one element.
    '''
    ax = fig.axes
    try:
        if len(ax) == 1:
            return ax[0]
        else:
            return ax
    except Exception:
        return ax

def get_xy_from_fig(fig):
    '''
    output dimensions : [x/y, point]
    '''
    ax = fig.axes[0]
    line = ax.get_lines()[0]
    return line.get_xydata().T
    
def get_xy_from_fig_deep(fig):
    '''
    output dimensions : [subplot, line, x/y, point]
    '''
    out = []
    axs = fig.axes
    for ax in axs:
        lines = ax.get_lines()
        temp = []
        for line in lines:
            temp.append(line.get_xydata().T)
        out.append(temp)    
    return out

def complex_to_hsv(X, Hg=1, H0=0):
    '''
    For use in mpl.colors.hsv_to_rgb
    
    Hg controls the gain on the hue from complex phase.
    H0 controls the offset, or the 0 point of the hue
    '''
    out_shape = list(np.shape(X)) + [3]
    out = np.zeros(out_shape)
    X_abs = np.abs(X)
    X_arg = np.angle(X)
    
    H = (X_arg - np.min(X_arg))/np.ptp(X_arg)
    H = (H*Hg - H0)%1
    S = np.ones_like(X, dtype=float)
    V = X_abs/np.max(X_abs)
    
    out[...,0] = H
    out[...,1] = S
    out[...,2] = V
    return out

def complex_imshow_rgb(X, Hg=1, H0=0):
    rgb = mpl.colors.hsv_to_rgb(complex_to_hsv(X, Hg=Hg, H0=H0))
    im = plt.imshow(rgb)
    return im

def set_minor_xtick_step(step=1, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_minor_locator(plt.MultipleLocator(step))
    return step

def set_minor_ytick_step(step=1, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.yaxis.set_minor_locator(plt.MultipleLocator(step))
    return step

def set_major_xtick_step(step=1, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(step))
    return step

def set_major_ytick_step(step=1, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.yaxis.set_major_locator(plt.MultipleLocator(step))
    return step

def set_major_ytick_resolution(ax, res=9):
    '''
    Tries to adjust tick spacing so that the requested number of ticks are shown.
    Tries to use "nice" tick step sizes.
    Doesn't always get the exact number of ticks because there might not be a "nice"
    step size that divides it that way.

    In that case set_ylim_by_ytick_padding() can be used to add extra ticks.
    '''
    lns = ax.lines
    min = np.inf
    max = -np.inf
    for ln in lns:
        mn, mx = nf.minmax(ln.get_ydata())
        if mn < min:
            min = mn
        if mx > max:
            max = mx
    ptp = max - min

    step = ptp/res
    step_mag = gf.float_mag(step) + 1
    step_digits = step / 10**step_mag
    # print(ptp)
    # print(step, step_digits, 10**step_mag)
    closest_nice_step_digits = nice_ticks[nf.find_nearest_ind(nice_ticks, step_digits)]
    nice_step = closest_nice_step_digits * 10**step_mag
    # print(closest_nice_step_digits, nice_step)
    ax.yaxis.set_major_locator(plt.MultipleLocator(nice_step))
    return step

def set_ylim_by_ytick_padding(ax, lp=0.6, up=0.6):
    '''
    Useful as final step to adjust yticks for twinx plots.
    Setting either lp=2 and up=1.0 adds 2 extra ticks below and 1 extra tick above.
    '''
    ylim_min, ylim_max = ax.get_ylim()
    yticks = ax.get_yticks()
    yticks = yticks[yticks < ylim_max]
    yticks = yticks[yticks > ylim_min]
    ytick_step = yticks[1] - yticks[0]
    ytick_max = yticks[-1]
    ytick_min = yticks[0]
    # print(ytick_min, ytick_max, ytick_step)
    new_ylim_max = ytick_max + up*ytick_step*np.sign(ytick_step)
    new_ylim_min = ytick_min - lp*ytick_step*np.sign(ytick_step)
    ax.set_ylim(new_ylim_min, new_ylim_max)
    return new_ylim_min, new_ylim_max

def qplot(y, *args, **kwargs):
    '''Quantile plot
    '''
    if len(y) == 1:
        y = np.repeat(y, 2)
    x = np.linspace(0,1,len(y))
    ln = plt.plot(x, y, *args, **kwargs)
    return ln

def qsemilogy(y, *args, **kwargs):
    '''Quantile semilogy plot
    '''
    if len(y) == 1:
        y = np.repeat(y, 2)
    x = np.linspace(0,1,len(y))
    ln = plt.semilogy(x, y, *args, **kwargs)
    return ln

def outer_legend(*args, **kwargs):
    leg = plt.legend(*args, loc='upper left', bbox_to_anchor=(1, 1), **kwargs)
    return leg