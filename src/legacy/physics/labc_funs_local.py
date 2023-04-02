'''
Example:

measurement = ['this can be any pickleable python object']
labc_funs.save_measurement(measurement, 'rp6_in1_dark')
labc_funs.load_latest_measurement('rp6_in1_dark')

'''

__version__ = "0.0.1"

import numpy as np
import shutil
import zipfile
import io
import os
import pickle as pl
import itertools
import time
import matplotlib.pyplot as plt

import requests
import bs4

try:
    import pykat
    pykat.init_pykat_plotting()
except ModuleNotFoundError:
    print("warning: module \"pykat\" not found. Using default plotting style.")

import general_funs as gf

home_dir = os.path.expanduser("~")
labc_dir = os.path.join(home_dir, 'labc')

os.makedirs(labc_dir, exist_ok=True)

labc_rps = ['rp6', 'rp7', 'rp8'] 
rp_channels = ['in1', 'in2', 'iq0', 'iq1', 'iq2', 'pid0', 'pid1', 'pid2', 'out1', 'out2']
locked1_states = ['locked1_TEM00_carrier', 'locked1_TEM00_sideband', 'locked1_TEM02_carrier', 'locked1_TEM02_sideband']
locked2_states = ['locked2_TEM00_carrier']
phase_loop_states = ['locked_phase_loop']
all_locked_states = locked2_states + locked1_states + [y[0]+'_'+y[1] for y in itertools.product((locked1_states + [x[0]+'_'+x[1] for x in itertools.product(locked1_states, locked2_states)]), phase_loop_states)]
all_rp_codes = [x[0]+'_'+x[1] for x in itertools.product(labc_rps, rp_channels)]
all_measurement_codes = [x[0]+'_'+x[1] for x in itertools.product(all_rp_codes, ['dark'] + ['LO'] + all_locked_states)]

def assert_measurement_code(code):
    if code not in all_measurement_codes:
        raise Exception(f"Code \"{code}\" doesn't match any known measurement codes. Fix name or pass force=True.")

def save_measurement(obj, file_name, force=False, debug=False):
    if not force:
        if file_name not in all_measurement_codes:
            raise Exception(f"File name \"{file_name}\" doesn't match any known measurement codes. Fix name or pass force=True.")
    unixtime = time.time()
    timestamp = gf.get_timestamp(unixtime)
    stamped_name = file_name + "_" + timestamp
    abs_path = os.path.join(labc_dir, stamped_name)
    file_dir = abs_path
    os.makedirs(file_dir, exist_ok=True)
    pickle_path = os.path.join(file_dir, stamped_name)

    meta_dict = {'name': file_name, 'labc_ver': __version__, 'unixtime': unixtime, 'timestamp': timestamp, 'data': obj}
    if debug:
        print(pickle_path)
    with open(pickle_path + ".pl", 'wb') as f:
        pl.dump(meta_dict, f)

    shutil.make_archive(file_dir, 'zip', file_dir)
    shutil.rmtree(file_dir)
    print('written: ', file_dir+'.zip')

def load_measurement(filename):
    filepath = os.path.join(labc_dir, filename)
    print("loaded: ", filepath)
    return gf.extract_pickles_from_zip(filepath)

def load_latest_measurement(file_name, debug=False):
    labc_ls = os.listdir(labc_dir)
    if debug:
        print(labc_ls)
        print([x[0:len(file_name)] for x in labc_ls])
    filtered_ls = [x for x in labc_ls if x[0:len(file_name)] == file_name]
    if debug:
        print(filtered_ls)
    return load_measurement(sorted(filtered_ls, reverse=True)[0])

def extract_spectrum(load_dict):
    return load_dict['data']['spectrum']

def load_latest_spectrum(file_name):
    return extract_spectrum(load_latest_measurement(file_name))

def plot_latest_spectra(codes):
    for code in codes:
        assert_measurement_code(code)

    fig, ax = plt.subplots(1,1,figsize=[12,6])
    for code in codes:
        load_dict = load_latest_measurement(code)
        ax.loglog(*load_dict['data']['spectrum'], label=load_dict['name']+' '+load_dict['timestamp'])
    ax.legend()
    ax.set_title(gf.get_timestamp())
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"PSD [V$^2$ / Hz]")
    return fig, ax
