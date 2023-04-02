import peak_finder
import numpy as np  
from numpy import fft
import matplotlib.pyplot as plt
import scipy.signal
from peak_finder import peakdet
import sys
import csv

## STAGE 1 (parsing and cleaning)
    
def parse_cds_output(filepath, columns=['time','data'], verbose=True):
    dc = {}
    data =[]
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ')
        data = list(csvreader)
        
    data = np.array(data).astype(float)
    dc[columns[0]] = data[:,0]
    dc[columns[1]] = data[:,1]
    
    return dc
    
def parse_OMC_scan_data(filepath_DCPD, filepath_PZT=None, columns=['time','omc','pzt'], verbose=True):

    if filepath_PZT is None:
        # filepath_DCPD contains DCPD and PZT data
        dc = {}
        data =[]
        with open(filepath_DCPD, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
            data = list(csvreader)
            
        data = np.array(data).astype(float)
        dc[columns[0]] = data[:,0]
        dc[columns[1]] = data[:,1]
        dc[columns[2]] = data[:,2]

    else:
        dc_DCPD = parse_cds_output(filepath=filepath_DCPD,columns=[columns[i] for i in [0,1]],verbose=verbose)
        dc_PZT = parse_cds_output(filepath=filepath_PZT,columns=[columns[i] for i in [0,2]],verbose=verbose)
        
        dc = dc_DCPD   
        # append the last column of pzt data onto the dcpd data and return
        dc[columns[2]] = dc_PZT[columns[2]]
    
    return dc
    
def crop_data(dc,keep_range=[0,1]):
    min_pzt = np.min(dc['pzt'])
    max_pzt = np.max(dc['pzt'])
    range_pzt = max_pzt - min_pzt
    
    # crop the pzt data around the top and bottom by the percentages in keep_range if the user asks for it
    # keep_range=[0,1] will keep all of the data
    mask = (dc['pzt'] > min_pzt+range_pzt*keep_range[0]) & (dc['pzt'] < min_pzt+range_pzt*keep_range[1])
    
    dc2 = {}
    for key in dc:
        dc2[key] = dc[key][mask]
    
    return dc2

def filter_by_gradient(dc,gradient_tolerance=0.01,verbose=True):
    dc['pzt_gradient'] = np.gradient(dc['pzt'])
    # use median instead of mean because spikes from railing can be huge
    median = np.median((dc['pzt_gradient']))

    dc['pzt_gradient_median_diff'] = np.abs(np.abs(dc['pzt_gradient'])-np.abs(np.median(dc['pzt_gradient'])))

    # if the magnitude of the gradient of the pzt differs too much from the median then reject it
    mask = dc['pzt_gradient_median_diff'] < np.abs(np.median(dc['pzt_gradient'])*gradient_tolerance)
    
    dc2 = {}
    for key in dc:
        dc2[key] = dc[key][mask]
    
    #print(df)
    
    if verbose:
        fig = plt.figure()
        plt.plot(dc2['time'],dc2['pzt_gradient'],'.',ms=1)
        plt.xlabel('time')
        plt.ylabel('pzt gradient')
        plt.ylim([median*(1-gradient_tolerance*2),median*(1+gradient_tolerance*2)])
        plt.show()
        #plt.ylim([median*(1-gradient_tolerance*2),median*(1+gradient_tolerance*2)])
        
    return dc2
    
def offset_negative_DCPD(dc,floor = 1e-6):
    min_dcpd = np.min(dc['omc'])
    if min_dcpd < 0:
        dc['omc'] = dc['omc'] - min_dcpd + floor
    else:
        # no negative data
        pass
    return dc
    
def triangle_hysterisis_compensation(dc,dir=1):
    dc['pzt_gradient'] = np.gradient(dc['pzt'])
    
    mask = np.sign(dc['pzt_gradient']) == dir
    dc2 = {}
    for key in dc:
        dc2[key] = dc[key][mask]
        
    return dc2
    
    
def initial_clean(dc,pzt_waveform='sawtooth',floor=1e-6,gradient_tolerance=0.01,dir=1,keep_range=[0,1],remove_negative=True,verbose=True):
    # this does rudimentary cropping and cleaning based on pzt gradient to try remove weird jumps in pzt

    # sawtooth suffers from artifacts due to sudden change
    # triangle suffers from hysterisis (upward path isn't the same as the downward path)
    if pzt_waveform in ['sawtooth']:
        # perform a check to remove railing in pzt data
        dc = filter_by_gradient(dc,gradient_tolerance=gradient_tolerance,verbose=verbose)
    elif pzt_waveform in ['triangle']:
        dc = triangle_hysterisis_compensation(dc,dir=dir)
       
    # crop the pzt data around the top and bottom by the percentages in keep_range if the user asks for it
    # keep_range=[0,1] will keep all of the data
    dc = crop_data(dc,keep_range=keep_range)
    
    if remove_negative:
        dc = offset_negative_DCPD(dc,floor=floor)
#     #remove negative PD readings
#     df = df[df['omc']>0]
    
    if verbose:
        fig = plt.figure(figsize=[12,10])

        plt.subplot(2, 1, 1)
        plt.plot(dc['pzt'])
        plt.xlabel('samples')
        plt.ylabel('pzt')
        plt.subplot(2, 1, 2)
        plt.semilogy(dc['pzt'],dc['omc'],'.',markersize=2)
        plt.xlabel('pzt')
        plt.ylabel('omc')
        plt.grid()
        plt.show()
    
    return dc
        
def digitize_output(dc,target='pzt',N_bins=40000,verbose=True):
    
    lbnd = np.min(dc[target])
    ubnd = np.max(dc[target])*(1+np.spacing(1.0)) # this is necessary to get the right number of bins
    data = dc[target]
    
    def binary_search(search_limits,test_fun,verbose=False):
        # test_fun has to return [True,...,True,False,...,False]
        # on the search range, where the binary search will find the
        # last interger in the search range for which test_fun returns
        # true, if it's all True then it will return the upper search 
        # limit and lower limit if it's all False
        L,R = search_limits
        while True:
            m = np.floor((L+R)/2).astype(int)
            if verbose:
                print(L,R,m)
            if L > R:
                return m
            else:
                if test_fun(m):
                    L = m + 1
                else:
                    R = m - 1
    
    def test_fun(lbnd,ubnd,data,nbins):
    # test function intended for use in binary search
    # returns True if every digitization bin contains a sample in it, otherwise returns False
        bin_boundaries = np.linspace(lbnd, ubnd, nbins+1)        
        # remember there are 1 fewer bins than there are boudnaries
        bin_ind = np.digitize(data, bin_boundaries)
#         print(len(np.unique(bin_ind)),nbins)
        return len(np.unique(bin_ind)) == (nbins)

    f_test_fun = lambda m: test_fun(lbnd,ubnd,data,m)
         
    bins = np.linspace(lbnd, ubnd, N_bins+1)
    bin_ind = np.digitize(data, bins)
    
    # user might have asked for too many bins, check if thats the case and correct it
    if len(np.unique(bin_ind)) != (N_bins):
        if verbose:
            print('Warning: Empty digitization bins found. Iteratively decreasing bin size until every bin contains at least one sample.')
        L = 0
        R = N_bins
        # run binary search
        m = binary_search([L,R],f_test_fun)  
        
        # update bin sizes
        bins = np.linspace(lbnd, ubnd, m+1)
        bin_ind = np.digitize(data, bins)

    digitized_target = bins[bin_ind]
    
    # groupby digitized_target
    target_groups = {}
    target_bins = digitized_target
    omc_vals = dc['omc']
    for target_bin, omc_val in zip(target_bins, omc_vals):
        if target_bin not in target_groups:
            # new bin: create group
            target_groups[target_bin] = []
        # add omc_val to the target_bin
        target_groups[target_bin].append(omc_val)
    
    # take mean across omc_vals in the same target_bins
    group_mean = []
    for target_bin in target_groups:
        group_mean.append([target_bin,np.mean(target_groups[target_bin])])
    # print(result)

    # sort array by target val (so that it prints nicely)
    dn = np.array(group_mean)
    sort_ind = np.argsort(dn[:,0])
    dc2 = {}
    dc2[target] = dn[sort_ind,0]
    dc2['omc'] = dn[sort_ind,1]
    
    return dc2
    
def split_runs(dc,target='pzt',grad_cutoff = 1e-3):
    A = dc['pzt']
    mask = np.where(np.abs(A[:] - np.median(A)) > grad_cutoff)[0]
    dc2 = {}
    i1 = 0
    for i2 in mask:
        dc2[i2] = {}
        for key in dc:
            dc2[i2][key] = dc[key][i1:i2]
        i1 = i2
        
def apply_lowpass_omc(dc,b=35,M=70):
    dc2 = dc.copy()
    ydata = dc['omc']
    dc2['omc'] = np.real(brick_lowpass(ydata,b=35,M=70,wn_fun=hamming_wn))
    return dc2
    
def stage1(filepath_DCPD,filepath_PZT=None,out_filename='stage_1_out',columns=['time','omc','pzt'],floor=1e-6,pzt_waveform='triangle'
,remove_negative=True,digitize_pzt=True,lowpass_omc=True,N_bins=40000,q=6,keep_range=[0,1],dir=1
,gradient_tolerance=0.01,plot_output=True,linear_yscale=True,verbose=False,export_csv=True,export_plot=True):
    '''
    TODO: add docstring
    '''
    
    dc = parse_OMC_scan_data(filepath_DCPD,filepath_PZT,columns=columns,verbose=verbose)
    dc2 = initial_clean(dc,floor=floor,pzt_waveform=pzt_waveform,gradient_tolerance=gradient_tolerance,keep_range=keep_range,dir=dir,verbose=verbose)
    if digitize_pzt:
        dc3 = digitize_output(dc2,verbose=verbose,N_bins=N_bins,target='pzt')
    else:
        dc3 = dc2.copy()
        sort_ind = np.argsort(dc3['pzt'])
        dc3['pzt'] = dc3['pzt'][sort_ind]
        dc3['omc'] = dc3['omc'][sort_ind]
        
        #y2 = scipy.signal.decimate(df3['omc'],q)
        #x2 = scipy.signal.decimate(df3['pzt'],q)
        #return x2,y2
        
    if lowpass_omc:
        dc4 = dc3.copy()
        dc4 = apply_lowpass_omc(dc3)
        
    dc_out = dc4
    
    fig = plt.figure(figsize=[12,6])
    if linear_yscale:
        plt.plot(dc_out['pzt'],dc_out['omc'],ms=1,lw=1)
    else:
        plt.semilogy(dc_out['pzt'],dc_out['omc'],ms=1,lw=1)
    plt.xlabel('pzt counts (uncalibrated)')
    plt.ylabel('OMC PD counts')
    plt.grid(True)
    
    if export_plot:
        fig.savefig(out_filename+'.pdf')
    
    if plot_output:
        plt.show()
        
    if export_csv:
        with open(out_filename+'.csv', 'w') as csvfile:
            wr = csv.writer(csvfile, delimiter=' ')
            wr.writerow(list(dc_out.keys()))
            for row in zip(dc_out['pzt'],dc3['omc']):
                wr.writerow(row)
        
    return dc_out,fig
        
## STAGE 2 (pzt counts to frequency conversion, pzt calibration)

def init_bootstrap(dc,FSR=2.649748e+08,show=False,delta=None):
    ydata = dc['omc']
    if delta is None:
        delta = max(ydata)/1.5
    zeroth_peaks = peakdet(ydata,delta=delta,show=show)
    
    fit0= {}
    fit0['omc'] = dc['omc'][zeroth_peaks]
    fit0['pzt'] = dc['pzt'][zeroth_peaks]
    
    if len(fit0['pzt']) < 2:
        sys.exit('Not enough zeroth order peaks detected. Try manually lowering delta, or use interactive mode (if it\'s ready).')
    elif len(fit0['pzt']) == 2:
        order = 1
    if len(fit0['pzt']) > 2:
        order = 2
        
    n_zeros = len(fit0['pzt'])
        
    fit0['freq'] = np.arange(0,n_zeros)*FSR

    p0 = np.polyfit(fit0['pzt'],fit0['freq'],order)

    f1 = dc.copy()
    f1['freq'] = np.polyval(p0,f1['pzt'])

    return fit0,f1
    
def lorentzian(fwhm,x,x0):
    return 0.5/np.pi*fwhm/((x-x0)**2+(0.5*fwhm)**2)
    
def lorentzian_filt(fwhm,N):
    # if used as a conv filter then the fft spectrum is 
    # exp(-fwhm/(1*N/pi) * range(N))
    x = np.arange(N)
    x0 = N/2
    return lorentzian(fwhm,x,x0)
    
def sinc_filt(b,N):
    # if used as a conv filter then the fft spectrum is 
    # rect(N/b) ie b is the fraction of the frequencies kept
    x = np.arange(N)-N/2
    return 1/b * np.sinc(x/b)

def myconv(u,v,mode='same'):
    # numpy.convolve is time-domain for some stupid reason
	u_fft = np.fft.fft(u)
	v_fft = np.fft.fft(v)
	return np.fft.fftshift(np.fft.ifft(u_fft*v_fft))
    
def hamming_wn(N):
    alpha = 25/46
    beta = 1 - alpha
    n = np.arange(N)
    return alpha - beta*np.cos(2*np.pi*n/(N-1))
    
def sinc_filt_fiir(b,N,M=100,wn_fun=hamming_wn):
    # if used as a conv filter then the fft spectrum is 
    # rect(N/b) ie b is the fraction of the frequencies kept
    full_filt = np.zeros(N)
    wn = wn_fun(M)
    filt = sinc_filt(b,M)*wn
    filt = filt / np.sum(filt) # renormalize filter for unity gain
    full_filt[0:M] = filt
    return np.roll(full_filt,np.ceil(N/2-M/2).astype(int))

def brick_lowpass(signal,b=30,M=100,wn_fun=hamming_wn,plot_response=False):
    # b is the fraction of the low frequencies kept
    # 1 keeps all freqs, 1/2, keeps half of freqs, 1/10 keeps a tenth ...
    # M is the length of the filter as M approaches N the filter approaches an ideal brick_lowpass
    # in practice M should be in range of [10,100]
    N = len(signal)
    full_filt = sinc_filt_fiir(b,N,M,wn_fun)
    if plot_response:
        plt.figure()
        plt.semilogy(np.abs(np.fft.fft(full_filt))[0:N//2])
        plt.show()
    
    return myconv(signal,full_filt)

def matched_filt_omc(xdata,ydata,fwhm_hz=643942.754981):  
    xrange = max(xdata)- min(xdata)
    fwhm_samples = fwhm_hz * len(ydata)/xrange
    lorentz_filt = lorentzian_filt(fwhm_samples,len(ydata))
    # since lorentzian is symmetric convolution=cross-correlation
    y_conv = np.real(myconv(lorentz_filt,ydata,mode='same'))
    return y_conv

def iter_bootstrap(dc1,old_fit_dc,f0,peaks_loc,RF=0,FSR=2.649748e+08,show=False):
    
    locs_dc = {}
    for key in dc1:
        locs_dc[key] = dc1[key][peaks_loc]
    
    locs_dc['diff'] = np.abs(np.abs(locs_dc['freq']-f0) - RF)
    sort_ind = np.argsort(locs_dc['diff'])
    for key in locs_dc:
        locs_dc[key] = locs_dc[key][sort_ind]

    if RF == 0:
        # we're locking to a carrier
        fit_dc ={}
        for key in locs_dc:
            # since the dict is sorted by 'diff' the first element of locs_dc 
            # correspobds to the peak closest in frequency to what we asked
            fit_dc[key] = locs_dc[key][0]
        # specify the freq of the peak (user gives FSR + HOM in f0)
        fit_dc['freq'] = np.array([0])+f0
    else:   
        # we're locking to RF sidebands
        fit_dc ={}
        for key in locs_dc: 
            # since the dict is sorted by 'diff' the first two elements of locs_dc 
            # correspobds to the peaks closest in frequency to what we asked
            fit_dc[key] = locs_dc[key][[0,1]]
        
        # needs to be sorted by pzt value to get consistent labelling of +/- sidebands
        sort_ind = np.argsort(fit_dc['pzt'])
        for key in fit_dc:
            fit_dc[key] = fit_dc[key][sort_ind]
        # specify the freq of the peak (user gives FSR + HOM in f0)
        fit_dc['freq'] = np.array([-RF,RF])+f0
    
    # append new fit dict to old fit dict
    new_fit_dc = {}
    for key in old_fit_dc:
        new_fit_dc[key] = np.hstack([old_fit_dc[key],fit_dc[key]])
    
    return new_fit_dc

def update_freq_model(dc_old,comp,poly_order=4,FSR=2.649748e+08,plot=True):
    p = np.polyfit(comp['pzt'],comp['freq'],poly_order)
    xnew = np.linspace(np.min(comp['pzt']),np.max(comp['pzt']),100)
    fnew = np.polyval(p,xnew)
    resid = (np.polyval(p,comp['pzt']) - comp['freq'])/FSR #in units of FSR
    
    if plot:
        plt.figure(figsize=[12,4])
        plt.subplot(1,2,1)
        plt.plot(comp['pzt'],comp['freq'],'x')
        plt.plot(xnew,fnew,'-',lw=1,mew=0.1)
        
        plt.subplot(1,2,2)
        plt.plot(comp['pzt'],resid,'x')
        plt.show()
        
    dc_new = dc_old.copy()
    dc_new['freq'] = np.polyval(p,dc_old['pzt'])
    
    return dc_new,p

def stage2(dc,deltaf=5.813185e+07,FSR=2.649748e+08,delta_upper=None,delta_lower=None,use_matched_filt=False,verbose=False,RFs=None,show=False,plot=False
    ,return_full=False):
    '''
    By default this will try to bootstrap a frequency model by the following procedure
        * Find all peaks that have a delta of at least y_max/1.5 and label them a zeroth order, spaced by FSR.
            The resulting model will be either linear if two peaks are found, or quadratic if more than two are found.
            The algorithm will halt and throw an error if only one zeroth peak is found.
        * Find all peaks with a delta of at least y_min*3 and find the first order ones by frequency distance from the previous model.
            Update the model (always quadtratic).
        * Find all peaks with a delta of at least y_min*3 and find the second order ones by frequency distance from the previous model.
            Update the model (one order higher than preivous model)
        
        Return the dictinary with the latest frequency model
        
        User can add RF frequencies to lock the model to, which will happen prior to the HOM phase (because RFs are closer to carrier than HOMs).
            User is responsible for confirming that every carrier has the RF sidebands around it visible. 
            The algorithm will likely give an inaccurate model if it can't find the sidebands (which is why I don't do this by default).
            
        delta_upper: used for finding zeroth order peaks and should be slightly lower than the height of the zeroth peak
            (default is y_max/1.5)
        
        delta_lower: used for finding ALL peaks and should be slightly higher than the noise amplitude in the data
            (default is y_min*3) NOTE: susceptible to large negative spikes (removing negatives in stage 1 needs to work correctly)
            
        use_matched_filt: if data is too noisy for peak finder to work (shouldn't be necessary anymore due to lowpass in stage1)
    '''

    fit0,f0_init = init_bootstrap(dc,delta=delta_upper,show=show)
    # make a copy since we need the original to make a comparison plot
    f0 = f0_init.copy()
    n_zeros = len(fit0['freq'])
    if n_zeros >= 2:
        poly_order = 2

    xdata = f0['freq']
    ydata = f0['omc']
    # before we start iter_bootstrap we will compute the peak locations beforehand
    # because the matched filter it uses is expensive (too expensive for realtime)
    if use_matched_filt:
        yconv = matched_filt_omc(xdata,ydata)
    else:
        yconv = ydata
    
    if delta_lower is None:
        delta_lower = min(yconv)*3
    
    peaks_loc = peakdet(yconv,delta=delta_lower)
        
    # first check if we have RFs (this might not always work)
    # user needs to make sure they are visible around every carrier
    if RFs is not None:
        for RF in RFs:
            for i in np.arange(0,n_zeros,1):
                fit0 = iter_bootstrap(f0,fit0,FSR*i,RF=RF,peaks_loc=peaks_loc,show=show)
                f0,p = update_freq_model(f0,fit0,poly_order=poly_order,plot=plot)
    
        # upgrade poly_order (since we have more points)
        poly_order += 1
    
    # lock to second order carriers
    if abs(f0['freq'][0] - fit0['freq'][0]) < deltaf*3:
        # don't look for zeroth order HOMs of zero FSR
        start = 1 # start at first FSR
    else:
        start = 0
    for i in np.arange(start,n_zeros,1):
        hom_freq_2 = FSR*i-2*deltaf
        hom_freq_1 = FSR*i-1*deltaf
        fit0 = iter_bootstrap(f0,fit0,hom_freq_1,peaks_loc=peaks_loc,show=show)
        # update frequency model from first order peak before trying to find second
        f0,p = update_freq_model(f0,fit0,poly_order=poly_order,plot=plot)
        fit0 = iter_bootstrap(f0,fit0,hom_freq_2,peaks_loc=peaks_loc,show=show)
    
    # create final freq model
    f1,p = update_freq_model(f0,fit0,poly_order=poly_order+1,plot=plot)
    
    if verbose:
        plt.figure(figsize=(8, 4))
        plt.semilogy(f0_init['freq'],f0_init['omc'],lw=1)
        plt.semilogy(f1['freq'],f1['omc'],lw=1)
        plt.semilogy(fit0['freq'],fit0['omc'],'x',mec='r',mew=2, ms=8)
        plt.show()
    
    if return_full:
        return f1,p,fit0
    else:
        return f1
    
## STAGE 3 (peak labelling and mismatch calculating)

# the flow order should go as 
#df1 = gen_freq_table(xdata,RFs,FSR,deltaf,8)
#df2 = get_peak_labels(df1,xdata,ydata)
#df3 = get_even_peak_heights(df2,confidence_cut)

def gen_HOM_locs(freq_data,FSR=2.649748e+08,deltaf=-5.813185e+07,maxtem=8):
    min_f = np.min(freq_data)
    max_f = np.max(freq_data)
    
    maxHOM = deltaf*maxtem
    
    if deltaf > 0:
        # we want to see lower FSRs that will ring up their HOMs
        minFSR = np.floor(min_f/FSR)
        maxFSR = np.floor(max_f/FSR)
    elif deltaf < 0:
        # we want to see higher FSRs that will ring down their HOMs
        minFSR = np.ceil(min_f/FSR)
        maxFSR = np.ceil(max_f/FSR)

    f_locs = {}
    f_locs['HOM_order'] = np.arange(0,maxtem+1)
    for FSR_order in np.arange(minFSR,maxFSR+1).astype(int):
        f_locs[FSR_order] = np.zeros([maxtem+1])
        f_locs[FSR_order][:] = np.nan
        HOM_order = 0
        f = HOM_order*deltaf
        for HOM_order in range(maxtem+1):
            f = FSR*(FSR_order) + HOM_order*deltaf
            f_locs[FSR_order][HOM_order] = f
            
    return f_locs

def add_RF_peak_locs(dc,RFs):
    cols = list(dc.keys())
    for FSR_order in cols[1:]:
        for i,RF in enumerate(RFs):
            label_p = str(FSR_order)+'_RF'+str(i)+'p'
            label_m = str(FSR_order)+'_RF'+str(i)+'m'
            dc[label_p] = dc[FSR_order] + RF
            dc[label_m] = dc[FSR_order] - RF
    return dc

def gen_freq_table(freq_data,RFs=[],FSR=2.649748e+08,deltaf=-5.813185e+07,maxtem=8):
    dc1 = gen_HOM_locs(freq_data,FSR,deltaf,maxtem)
    dc2 = add_RF_peak_locs(dc1,RFs)
    return dc2

def linearise_freq_table(dc):
    cols = list(dc.keys())

    dc2 = {}
    dc2['HOM_order'] = np.array([],dtype=int)
    dc2['label'] = np.array([],dtype=object)
    dc2['freq'] = np.array([],dtype=float)
    maxtem = len(dc['HOM_order']) - 1
    for col in cols[1:]:
        # print(col)
        dc2['HOM_order'] = np.hstack([dc2['HOM_order'],dc['HOM_order']])
        dc2['label'] = np.hstack([dc2['label'],np.repeat(col,maxtem+1)])
        dc2['freq'] = np.hstack([dc2['freq'],dc[col]])
    return dc2

def find_most_likely_peak(peak_f,dc,FSR=2.649748e+08):
    # df needs to have been linearized
    dc_foo = dc.copy()
    
    fun = lambda x: x/FSR # normalizing the difference metric
    vec_fun = np.vectorize(fun)
    dc_foo['metric'] = vec_fun(np.abs(dc_foo['freq'] - peak_f))   
    # minimum of metric is the closest peak to what we asked for
    best_candidate_id = np.argmin(dc_foo['metric'])
    
    best_candidate = {}
    for key in dc_foo:
        best_candidate[key] = dc_foo[key][best_candidate_id]
        
    return best_candidate

def find_peak_ind(xdata,ydata,delta=None,fwhm_hz=643942.754981,verbose=True):
    # could use a different backend here  

    if verbose:
        show = True
    else:
        show = False
        
    y_conv = matched_filt_omc(xdata,ydata)
    
    if delta is None:
        delta = min(y_conv)*2
        
    peak_ind = peak_finder.peakdet(y_conv,delta=delta,show=False)
    if show:
        plt.figure(figsize=(8, 4))
        plt.semilogy(ydata,'b',lw=1)
        plt.semilogy(peak_ind,ydata[peak_ind],'x',mec='r',mew=2, ms=8)
        plt.show()
        
    return peak_ind

def get_peak_labels(dc,freq_data,height_data,delta=None,verbose=True,debug=True):
    peak_ind = find_peak_ind(freq_data,height_data,delta=delta,verbose=verbose)
    
    peak_freqs = freq_data[peak_ind]  
    #freq_spacing = np.abs(freq_data[0] - freq_data[1])
    freqs_list = linearise_freq_table(dc)
    
    cs = []
    for peak_f in peak_freqs:
        cs.append(find_most_likely_peak(peak_f,freqs_list))       
    # could return cs in debugging since it's easier to read
    if debug:
        for dict in cs:
            print(dict)
    
    # bet you wish you had pandas now
    # cs is a list of dictionaries, turn it into a dictionary of lists
    
    # initialise dictionary
    dc_cs = {}
    for key in cs[0]:
        dc_cs[key] = []
    
    # transfer list of dictionaries into dictionary of lists
    for row in cs:
        for key in row:
            dc_cs[key].append(row[key])
    
    # turn lists into arrays
    for key in row:
        if key == 'label':
            dc_cs[key] = np.array(dc_cs[key],dtype=object)
        else:
            dc_cs[key] = np.array(dc_cs[key])
    
    dc_cs['peak_index'] = peak_ind
    dc_cs['peak_freq'] = peak_freqs
    dc_cs['peak_height'] = np.abs(height_data[dc_cs['peak_index']])
    
    return dc_cs

def get_even_peak_heights(dc_cs,max_confidence=1e-3):
    # 1e-3 is good confidence in 1/FSR metric

    dc_even = dc_cs.copy()
    
    def test_int(x):
        try:
            int(x)
        except:
            return False
        return True
    
    # select all carriers (test if label is an integer)
    vec_test_int = np.vectorize(test_int)
    carrier_mask = vec_test_int(dc_even['label'])
    for key in dc_cs:
        dc_even[key] = dc_even[key][carrier_mask]
    
    # only keep entries less than max_confidence
    confidence_mask = dc_even['metric'] < max_confidence
    for key in dc_even:
        dc_even[key] = dc_even[key][confidence_mask]
    
    # groupby HOM_order
    HOM_groups = {}
    HOM_bins = dc_even['HOM_order']
    height_vals = dc_even['peak_height']
    for HOM_bin, height_val in zip(HOM_bins, height_vals):
        if HOM_bin not in HOM_groups:
            # new bin: create group
            HOM_groups[HOM_bin] = []
        # add omc_val to the pzt_bin
        HOM_groups[HOM_bin].append(height_val)
    
    # take mean across height_vals in the same HOM_bins
    group_mean_std = HOM_groups.copy()
    for HOM_bin in group_mean_std:
        group_mean_std[HOM_bin] = {}
        group_mean_std[HOM_bin]['mean'] = np.mean(HOM_groups[HOM_bin])
        group_mean_std[HOM_bin]['std'] = np.std(HOM_groups[HOM_bin])
    # print(result)
    
    # dc_out = {}
    # dc_out['HOM_order'] = []
    # dc_out['peak_height'] = []
    # dc_out['peak_std'] = []
    # for HOM_bin in group_mean_std:
        # dc_out['HOM_order'].append(HOM_bin)
        # dc_out['peak_height'].append(group_mean_std[HOM_bin]['mean'])
        # dc_out['peak_std'].append(group_mean_std[HOM_bin]['std'])
    dc_out = {}
    for key in group_mean_std:
        dc_out[key] = {}
        dc_out[key]['peak_height'] = group_mean_std[key]['mean']
        dc_out[key]['peak_std'] = group_mean_std[key]['std']
    
    return dc_out
 
def stage3(dc,FSR=2.649748e+08,deltaf=-5.813185e+07,RFs=[],maxtem=8,delta=None,confidence_cutoff=1e-3\
,return_full=False,verbose=True):
    '''
    default parameters computed from the FINESSE model of the OMC
    
    deltaf: mode separation frequency. Default value is chosen as the one computed by FINESSE OMC model
        along the x-axis (agrees better with measurements than the y-axis mode separation freq.)

    min_peak_ratio: is roughly the minimum peak height the peak finder will look for. Effectively equal 
       to the minimum mismatch that can be detected. Setting it too low will make it find false positive peaks 
       second order peaks around the second order frequency which will wildly throw off the mismatch value
    
    confidence_cutoff: peak label fitting goodness cutoff (chosen for best results)
        my labelling algorithm only considers frequency distance
    '''
    xdata = dc['freq']
    ydata = dc['omc']
    
    freq_labels = gen_freq_table(xdata,RFs,FSR,deltaf,maxtem)
    dc2 = get_peak_labels(freq_labels,xdata,ydata,delta=delta,verbose=verbose)
    dc3 = get_even_peak_heights(dc2,confidence_cutoff)
    mismatch = dc3[2]['peak_height']/dc3[0]['peak_height']
    mismatch_std = mismatch*np.sqrt((dc3[2]['peak_std']/dc3[2]['peak_height'])**2 + (dc3[0]['peak_std']/dc3[0]['peak_height'])**2)
    if verbose:
        print('mismatch is %3.5g%% +/- %3.1g%%' % (float(mismatch*100),mismatch_std*100))
    
    if return_full:
        return {'mismatch': mismatch,'peak labels': dc2, 'carrier heights': dc3, 'freq_labels' : freq_labels}
    else:
        return mismatch
