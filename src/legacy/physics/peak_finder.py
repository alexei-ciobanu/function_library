import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import NaN, Inf, arange, isscalar, asarray, array

__author__ = "Eli Billauer"
__version__ = "3.4.05"
__license__ = "public domain"

def peakdet(v, delta, x = None, return_min = False, show=False):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    
    mxposs = []
    mnposs = []
    mxs = []
    mns = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta < 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                mxposs.append(mxpos)
                mxs.append(mx)
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mnposs.append(mnpos)
                mns.append(mn)
                mx = this
                mxpos = x[i]
                lookformax = True

    mxind = array(mxposs,dtype=int)
    mnind = array(mnposs,dtype=int)
    
    if show:
        plt.figure(figsize=(8, 4))
        plt.semilogy(v,'b',lw=1)
        plt.semilogy(mxind,v[mxind],'x',mec='r',mew=2, ms=8)
        plt.semilogy(mnind,v[mnind],'x',mec='g',mew=2, ms=8)
        plt.show()
    
    if return_min:
        return mxind, mnind
    else:
        return mxind
        
        
def my_peak_det(x,delta_new=None,delta_old=None,num_peaks=1):
    '''finds the single biggest peak'''
    
    # initialisation
    if delta_new is None:
        delta_new = 2*np.max(np.abs(np.diff(x)))
    
    # feedback to prevent solution from oscillating
    if delta_old is not None:
        delta_new = (delta_new + delta_old)/2
               
    arr = peakdet(x,delta_new)
    s = 1 + num_peaks
    
    # recursion baby
    if len(arr) > num_peaks:
        print('too many peaks',len(arr))
        return my_peak_det(x,delta_new=delta_new*2,delta_old=delta_new,num_peaks=num_peaks)
    elif len(arr) < num_peaks:
        print('too few peaks',len(arr))
        return my_peak_det(x,delta_new=delta_new/2,delta_old=delta_new,num_peaks=num_peaks)
    
    return delta_new, arr
    
def my_peak_det_bs(x,delta_lower=None,delta_upper=None,num_peaks=1,show=None,verbose=False):
    '''Finds the single biggest peak. This one uses binary search.'''
    
    # initialisation
    if delta_upper is None:
        delta_upper = np.ptp(x)
    
    if delta_lower is None:
        delta_lower = 0
        
    if verbose:
        print(delta_upper-delta_lower, delta_lower, delta_upper)
           
    delta_m = (delta_upper + delta_lower)/2
            
    arr = peakdet(x,delta_m)
    s = 1 + num_peaks
    
    if np.isclose(delta_lower, delta_upper):
        return delta_m, arr
    
    # recursion baby
    if len(arr) > num_peaks:
        if verbose:
            print('too many peaks',len(arr), delta_m)
        return my_peak_det_bs(x,delta_lower=delta_m, delta_upper=delta_upper, num_peaks=num_peaks, show=show, verbose=verbose)
    elif len(arr) < num_peaks:
        if verbose:
            print('too few peaks',len(arr), delta_m)
        return my_peak_det_bs(x,delta_lower=delta_lower, delta_upper=delta_m, num_peaks=num_peaks, show=show, verbose=verbose)
    
    #print(show)
    
    if show == 'semilogy':
        plt.figure(figsize=[8*1.8,8])
        plt.semilogy(x)
        plt.semilogy(np.arange(len(x))[arr],x[arr],'x')
        plt.show()
    
    return delta_m, arr
    
def win_peak_find(v,x0,w=None,downsample=None,delta_lower=None,delta_upper=None,num_peaks=1):
    '''Find peaks in a small window of a dataset. The index it returs'''
    
#     if downsample is None:
    if w is None:
        w = np.size(v)//10
        
    xl = np.round(x0-w/2).astype(int)
    xl = xl * (xl >= 0) # don't go into negative indices
    
    xu = np.round(x0+w/2).astype(int)
    
    #print(xl,xu)
    
    _delta,ind = my_peak_det_bs(v[xl:xu],delta_lower=None,delta_upper=None,num_peaks=1)
    return ind + xl

def peakdet_rewrite(v, delta, x = None, return_min = False, show=False):

    mxposs = []
    mnposs = []
    mxs = []
    mns = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta < 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    decreasing = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if decreasing:
            if this < mx-delta:
                mxposs.append(mxpos)
                mxs.append(mx)
                mn = this
                mnpos = x[i]
                decreasing = False
                increasing = True
        elif increasing:
            if this > mn+delta:
                mnposs.append(mnpos)
                mns.append(mn)
                mx = this
                mxpos = x[i]
                decreasing = True
                increasing = False
        else:
            sys.exit('something wrong here')

    mxind = array(mxposs,dtype=int)
    mnind = array(mnposs,dtype=int)
    
    if show:
        plt.figure(figsize=(8, 4))
        plt.semilogy(v,'b',lw=1)
        plt.semilogy(mxind,v[mxind],'x',mec='r',mew=2, ms=8)
        plt.semilogy(mnind,v[mnind],'x',mec='g',mew=2, ms=8)
        plt.show()
    
    if return_min:
        return mxind, mnind
    else:
        return mxind
        
def win_max_find(v,x0,w=None):
    if w is None:
        w = np.size(v)//50
        
    xl = np.round(x0-w/2).astype(int)
    xl = xl * (xl >= 0) # don't go into negative indices
    
    xu = np.round(x0+w/2).astype(int)
    
#     print(xl,xu)
    
    ind = np.argmax(v[xl:xu])
    return ind + xl

def win_max_find_2(v,x0,w=None):
    '''
    recursively move the box if the argmax not in the middle of the box
    '''
    if w is None:
        w = np.size(v)//50
        
    xl = np.round(x0-w/2).astype(int)
    xl = xl * (xl >= 0) # don't go into negative indices
    
    xu = np.round(x0+w/2).astype(int)
    
    ind = np.argmax(v[xl:xu])
    print(ind/w)
    
    if ind/w < 0.5 - 2/w or ind/w > 0.5 + 2/w :
        return win_max_find_2(v,ind+xl,w)
    else:
        return (ind+xl)

def interactive_peak_find(ydata,w=None,plot='linear',peaks=12):
    '''
    If clicking doesn't work run %matplotlib notebook
    '''

    import ipywidgets as wdg 
    import matplotlib.pyplot as plt 
    
########################################################################################### 

    # Create and display textarea widget
    txt = wdg.Textarea(
        value='',
        placeholder='',
        description='event:',
        disabled=False
    )

    plot_button = wdg.Button(
        description='plot lines',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='plot lines',
    )

    clear_button = wdg.Button(
        description='clear all',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='clear all',
    )
    
########################################################################################### 

    hbox = wdg.HBox([plot_button,clear_button])

    def make_child(name):
        return wdg.HBox([wdg.FloatText(description=name)])

    tab_contents = ['P0', 'P1', 'P2', 'P3', 'P4','P5','P6','P7','P8','P9','P10','P11','P12']
    children = [make_child(name) for name in tab_contents]
    tab = wdg.Tab()
    tab.children = children
    tab.dic = {}
    for i in range(len(children)):
        tab.set_title(i, str(i)+'')

    # displaying widgets "runs" them
    display(tab)
    display(hbox)

###########################################################################################    
    
    fig, ax1 = plt.subplots(figsize=[9.5,5])
    if plot == 'linear':
        ax1.plot(ydata)
    elif plot == 'semilogx':
        ax1.semilogx(ydata)
    elif plot == 'semilogy':
        ax1.semilogy(ydata)
    elif plot == 'loglog':
        ax1.loglog(ydata)
    plt.grid(True)
    plt.xlabel('sample number')
    plt.ylabel('')
    plt.tight_layout()
    
    
###########################################################################################

    def plot_lines(x):
        for child in tab.children:
            try:
                child.vline.set_xdata([child.children[0].value,child.children[0].value])
                if child.children[0].value:
                    child.vline.set_visible(True)
            except:
                child.vline = ax1.axvline(child.children[0].value,0,1,c='r',visible=False)
                if child.children[0].value:
                    child.vline.set_visible(True)
        return

    def clear_lines(x):
        tab.dic.clear()
        for child in tab.children:
            child.children[0].value = 0
            try:
                child.vline.set_visible(False)
            except:
                pass
        return

    # register clicky functions
    clear_button.on_click(clear_lines)
    plot_button.on_click(plot_lines)
    
########################################################################################### 

    # Define a callback function that will update the textarea
    def onclick(event):
        indx = int(event.xdata)
        indp = win_max_find_2(ydata,indx,w=w)
        ind_tab = tab.selected_index
        child = tab.children[ind_tab]
        child.children[0].value = indp # Dynamically update the text box above
        plot_button.click()
        tab.dic[ind_tab] = indp
    #     ax1.axvline(indp,0,1,c='r')

    # Create an hard reference to the callback not to be cleared by the garbage collector
    ka = fig.canvas.mpl_connect('button_press_event', onclick)
    
    # return a reference to a dictionary that the gui can modify
    return tab.dic
