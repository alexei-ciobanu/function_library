import peak_finder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from peak_finder2 import detect_peaks
import sys

## STAGE 1 (parsing and cleaning)

def parse_cds_output(filepath, columns=['time','data'], verbose=True):
    df = pd.read_csv(filepath, delim_whitespace=True)
    df.columns = columns
    
    if verbose:
        fig = plt.figure(figsize=[7,4])

        plt.plot(df[columns[0]],df[columns[1]])
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
   
        plt.show()       
    return df
    
def parse_OMC_scan_data(filepath_DCPD, filepath_PZT=None, columns=['time','omc','pzt'], verbose=True):
    
    if filepath_PZT is None:
        # filepath_DCPD contains DCPD and PZT data
        df = pd.read_csv(filepath_DCPD, delim_whitespace=True)
        df.columns = columns

    else:
        df_DCPD = parse_cds_output(filepath=filepath_DCPD,columns=[columns[i] for i in [0,1]],verbose=verbose)
        df_PZT = parse_cds_output(filepath=filepath_PZT,columns=[columns[i] for i in [0,2]],verbose=verbose)
        
        df = df_DCPD.copy()    
        # append the last column of pzt data onto the dcpd data and return
        df[columns[2]] = df_PZT[columns[2]]
    
    return df
    
def crop_data(df,keep_range=[0,1]):
    min_pzt = df['pzt'].min()
    max_pzt = np.max(df['pzt'])
    range_pzt = max_pzt - min_pzt
    
    # crop the pzt data around the top and bottom by the percentages in keep_range if the user asks for it
    # keep_range=[0,1] will keep all of the data
    df = df[(df['pzt'] > min_pzt+range_pzt*keep_range[0]) & (df['pzt'] < min_pzt+range_pzt*keep_range[1])]
    
    df = df.reset_index(drop=True) # need to reset index after filtering
    
    return df

def filter_by_gradient(df,gradient_tolerance=0.01,verbose=True):
    df['pzt_gradient'] = np.gradient(df['pzt'])
    # use median instead of mean because spikes from railing can be huge
    median = np.median((df['pzt_gradient']))

    df['pzt_gradient_median_diff'] = np.abs(np.abs(df['pzt_gradient'])-np.abs(np.median(df['pzt_gradient'])))

    # if the magnitude of the gradient of the pzt differs too much from the median then reject it
    df = df[df['pzt_gradient_median_diff'] < np.abs(np.median(df['pzt_gradient'])*gradient_tolerance)]

    df = df.reset_index(drop=True) # need to reset index after filtering
    
    #print(df)
    
    if verbose:
        fig = plt.figure()
        plt.plot(df['time'],df['pzt_gradient'],'.',ms=1)
        plt.xlabel('time')
        plt.ylabel('pzt gradient')
        plt.ylim([median*(1-gradient_tolerance*2),median*(1+gradient_tolerance*2)])
        plt.show()
        #plt.ylim([median*(1-gradient_tolerance*2),median*(1+gradient_tolerance*2)])
        
    return df
    
def offset_negative_DCPD(df,floor = 1e-6):
    df = df.copy()
    min_dcpd = df['omc'].min()
    if min_dcpd < 0:
        df['omc'] = df['omc'] - df['omc'].min() + floor
    else:
        # no negative data
        pass
    return df
    
def triangle_hysterisis_compensation(df,dir=1):
    df = df.copy()
    df['pzt_gradient'] = np.gradient(df['pzt'])
    df = df[np.sign(df['pzt_gradient']) == dir]
    return df
    
    
def initial_clean(df,pzt_waveform='sawtooth',floor=1e-6,gradient_tolerance=0.01,dir=1,keep_range=[0,1],remove_negative=True,verbose=True):
    # this does rudimentary cropping and cleaning based on pzt gradient to try remove weird jumps in pzt
    
    df = df.copy()   

    # sawtooth suffers from artifacts due to sudden change
    # triangle suffers from hysterisis (upward path isn't the same as the downward path)
    if pzt_waveform in ['sawtooth']:
        # perform a check to remove railing in pzt data
        df = filter_by_gradient(df,gradient_tolerance=gradient_tolerance,verbose=verbose)
    elif pzt_waveform in ['triangle']:
        df = triangle_hysterisis_compensation(df,dir=dir)
       
    # crop the pzt data around the top and bottom by the percentages in keep_range if the user asks for it
    # keep_range=[0,1] will keep all of the data
    df = crop_data(df,keep_range=keep_range)
    
    if remove_negative:
        df = offset_negative_DCPD(df,floor=floor)
#     #remove negative PD readings
#     df = df[df['omc']>0]
    
    df = df.reset_index(drop=True) # need to reset index after filtering
    
    if verbose:
        fig = plt.figure(figsize=[12,10])

        plt.subplot(2, 1, 1)
        plt.plot(df['pzt'])
        plt.xlabel('samples')
        plt.ylabel('pzt')
        plt.subplot(2, 1, 2)
        plt.semilogy(df['pzt'],df['omc'],'.',markersize=2)
        plt.xlabel('pzt')
        plt.ylabel('omc')
        plt.grid()
        plt.show()
    
    return df
        
def digitize_output(df2,N_bins=40000,verbose=True):
    
    lbnd = min(df2['pzt'])
    ubnd = max(df2['pzt'])*(1+np.spacing(1.0))
    data = df2['pzt'].values
    
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

    digitized_pzt = bins[bin_ind]
    
    # stitch back the digitized dataframe
    df2_digitized = pd.concat([pd.DataFrame(digitized_pzt),df2['omc']],axis=1)
    df2_digitized.columns = ['pzt','omc']
    
    # average omc pd values in the same pzt bins
    df2_digitized = df2_digitized.groupby('pzt',as_index=True).mean()
    # reset and sort
    df2_digitized = df2_digitized.reset_index().sort_values(by='pzt')
    
    return df2_digitized
    
def just_do_it(filepath_DCPD,filepath_PZT=None,columns=['time','omc','pzt'],floor=1e-6,pzt_waveform='triangle',remove_negative=True,digitize_pzt=True,N_bins=40000,q=6,\
keep_range=[0,1],dir=1,gradient_tolerance=0.01,plot_output=True,linear_yscale=True,verbose=False,export_csv=True,export_plot=True):
    
    df = parse_OMC_scan_data(filepath_DCPD,filepath_PZT,columns=columns,verbose=verbose)
    df2 = initial_clean(df,floor=floor,pzt_waveform=pzt_waveform,gradient_tolerance=gradient_tolerance,keep_range=keep_range,dir=dir,verbose=verbose)
    if digitize_pzt:
        df3 = digitize_output(df2,verbose=verbose,N_bins=N_bins)
    else:
        df3 = df2.copy()
        df3 = df3.sort_values(by='pzt').reset_index(drop=True)
        #y2 = scipy.signal.decimate(df3['omc'],q)
        #x2 = scipy.signal.decimate(df3['pzt'],q)
        #return x2,y2
    
    fig = plt.figure(figsize=[12,6])
    if linear_yscale:
        plt.plot(df3['pzt'],df3['omc'],'.',ms=1,lw=1)
    else:
        plt.semilogy(df3['pzt'],df3['omc'],'.',ms=1,lw=1)
    plt.xlabel('pzt counts (uncalibrated)')
    plt.ylabel('OMC PD counts')
    plt.grid(True)
    
    if export_plot:
        fig.savefig('stage_1_out.pdf')
    
    if plot_output:
        plt.show()
        
    if export_csv:
        with open('stage_1_out.csv','w') as fopen:
            fopen.write(df3.to_csv())
        
    return df3,fig
        
## STAGE 2 (pzt counts to frequency conversion, pzt calibration)

def init_bootstrap(df,FSR=2.649748e+08,show=False,level=1.2,n_zeros=None,order=1):
    ydata = df['omc']
    zeroth_peaks = detect_peaks(ydata,mph=max(ydata/level),mpd=len(ydata)/100,show=show)
    
    fit0= df.loc[zeroth_peaks].reset_index(drop=True)
    if len(fit0) != n_zeros:
        print(len(fit0),n_zeros)
        sys.exit('wrong number of peaks specified//detected')
    fit0['freq'] = np.arange(0,n_zeros)*FSR
    
    p0 = np.polyfit(fit0['pzt'],fit0['freq'],order)
    
    f1 = df.copy()
    f1['freq'] = f1['pzt'].apply(lambda x: np.polyval(p0,x))

    return fit0,f1

def iter_bootstrap(df1,old_fit_df,f0,RF=0,show=False,mpd=1,min_peak_ratio=1/1000,threshold=1e-5):    
    ydata = df1['omc']
    
    peaks_loc = detect_peaks(ydata,mph=max(ydata)*min_peak_ratio,mpd=mpd,threshold=threshold,show=show)
    locs_df = df1.loc[peaks_loc]
    
    locs_df['diff'] = np.abs(np.abs(locs_df['freq']-f0) - RF)
    locs_df = locs_df.sort_values(by='diff').reset_index(drop=True)

    if RF == 0:
        fit_df = locs_df.loc[0]
        fit_df['freq'] = np.array([0])+f0
        # fit_df is a series, coerce it to a DataFrame (transposed to match the other fits)
        fit_df = pd.DataFrame(fit_df).T
    else:   
        fit_df = locs_df.loc[[0,1]]
        # needs to be sorted by pzt value to get consistent labelling of +/- sidebands
        fit_df = fit_df.sort_values(by='pzt').reset_index(drop=True)
        fit_df['freq'] = np.array([-RF,RF])+f0
    
    return pd.concat([old_fit_df,fit_df])

def update_freq_model(df_old,comp,poly_order=4,FSR=2.649748e+08,plot=True):
    p = np.polyfit(comp['pzt'],comp['freq'],poly_order)
    xnew = np.linspace(comp['pzt'].min(),comp['pzt'].max(),100)
    fnew = np.polyval(p,xnew)
    resid = (np.polyval(p,comp['pzt']) - comp['freq'])/FSR #in units of FSR
    
    if plot:
        plt.figure()
        plt.plot(comp['pzt'],comp['freq'],'x')
        plt.plot(xnew,fnew,'-',lw=1,mew=0.1)
        plt.show()
        
        plt.figure()
        plt.plot(comp['pzt'],resid,'x')
        plt.show()
        
    df_new = df_old.copy()
    df_new['freq'] = df_old['pzt'].apply(lambda x: np.polyval(p,x))
    
    return df_new,p

## STAGE 3 (peak labelling and mismatch calculating)

# the flow order should go as 
#df1 = gen_freq_table(xdata,RFs,FSR,deltaf,8)
#df2 = get_peak_labels(df1,xdata,ydata)
#df3 = get_even_peak_heights(df2,confidence_cut)

def gen_HOM_locs(freq_data,FSR,deltaf,maxtem):
    min_f = np.min(freq_data)
    max_f = np.max(freq_data)
    
    maxHOM = deltaf*maxtem
    
    if deltaf > 0:
        # we want to see lower FSRs that will ring up their HOMs
        minFSR = np.ceil((min_f-maxHOM)/FSR)
        maxFSR = np.floor(max_f/FSR)
    elif deltaf < 0:
        # we want to see higher FSRs that will ring down their HOMs
        minFSR = np.ceil(min_f/FSR)
        maxFSR = np.floor((max_f - maxHOM)/FSR)

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
            
    return pd.DataFrame(f_locs)

def add_RF_peak_locs(df,RFs):
    cols = list(df.columns)
    for FSR_order in cols[1:]:
        for i,RF in enumerate(RFs):
            label_p = str(FSR_order)+'_RF'+str(i)+'p'
            label_m = str(FSR_order)+'_RF'+str(i)+'m'
            df[label_p] = df[FSR_order] + RF
            df[label_m] = df[FSR_order] - RF
    return df

def gen_freq_table(freq_data,RFs,FSR,deltaf,maxtem):
    df1 = gen_HOM_locs(freq_data,FSR,deltaf,maxtem)
    df2 = add_RF_peak_locs(df1,RFs)
    return df2

def linearise_freq_table(df):
    cols = list(df.columns)
    df_foo = df.melt(id_vars=cols[0],value_vars=cols[1:])
    return df_foo

def find_most_likely_peak(peak_f,df,f_spacing):
    # df needs to have been linearized
    df_foo = df.copy()
    df_foo['value'] = np.abs(df_foo['value'] - peak_f).apply(lambda x: abs(f_spacing/x))
    best_candidate = df_foo.loc[df_foo['value'].idxmax()]
    return best_candidate

def find_peak_ind(df,ydata,mph=0,mpd=3,threshold=0,verbose=True):
    # could use a different backend here
    if verbose:
        show = True
    else:
        show = False
    peak_ind = detect_peaks(ydata,mph=mph,mpd=mpd,threshold=threshold,edge=None,show=show)
    return peak_ind

def get_peak_labels(df,freq_data,height_data,mph=0,mpd=3,threshold=0,verbose=True):
    peak_ind = find_peak_ind(df,height_data,mph=mph,mpd=mpd,threshold=threshold,verbose=verbose)
    
    peak_freqs = freq_data[peak_ind]  
    freq_spacing = np.abs(freq_data[0] - freq_data[1])
    freqs_list = linearise_freq_table(df)
    
    cs = []
    for peak_f in peak_freqs:
        cs.append(find_most_likely_peak(peak_f,freqs_list,freq_spacing))
    
    df_cs = pd.DataFrame(cs).reset_index(drop=True)
    df_cs['peak_index'] = peak_ind
    df_cs['peak_freq'] = peak_freqs
    df_cs['peak_height'] = np.abs(height_data[df_cs['peak_index']])
    
    return df_cs

def get_even_peak_heights(df_cs,min_confidence):

    df_even = df_cs.copy()
    
    df_even = df_even[np.mod(df_cs['HOM_order'],1)==0][df_cs['variable'].isin(list(range(-10,11)))]
    df_even = df_even[df_even['value'] > min_confidence]
    
    df_out = df_even[['HOM_order','peak_height']].groupby(['HOM_order'], as_index=False).mean()
    df_out['peak_std'] = df_even[['HOM_order','peak_height']].groupby(['HOM_order'], as_index=False).std()['peak_height']
    return df_out
 
def make_it_happen(xdata,ydata,FSR=2.649748e+08,deltaf=-5.813185e+07,RFs=[],maxtem=8,min_peak_ratio=1/1000,threshold=1e-3,mpd=1,confidence_cutoff=1e-4\
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
    
    freq_labels = gen_freq_table(xdata,RFs,FSR,deltaf,maxtem)
    df2 = get_peak_labels(freq_labels,xdata,ydata,threshold=threshold,mph=max(ydata)*min_peak_ratio,mpd=mpd,verbose=verbose)
    df3 = get_even_peak_heights(df2,confidence_cutoff)
    mismatch = df3['peak_height'][2]/df3['peak_height'][0]
    mismatch_std = mismatch*np.sqrt((df3['peak_std'][2]/df3['peak_height'][2])**2 + (df3['peak_std'][0]/df3['peak_height'][0])**2)
    if verbose:
        print('mismatch is %3.5g%% +/- %3.1g%%' % (float(mismatch*100),mismatch_std*100))
    
    if return_full:
        return {'mismatch': mismatch,'peak labels': df2, 'carrier heights': df3, 'freq_labels' : freq_labels}
    else:
        return mismatch