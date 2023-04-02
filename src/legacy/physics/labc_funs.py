'''
This version works by running off of a local http server (running on port 8000), 
which serves home directory. A server can be started with

~$ python -m http.server

For a remote connection you need to ssh into the labc computer and forward port 
8000 to client's localhost

ssh cao@129.127.102.124 -L 8000:localhost:8000

Example:

measurement = ['this can be any pickleable python object']
labc_funs.save_measurement(measurement, 'rp6_in1_dark')
labc_funs.load_latest_measurement('rp6_in1_dark')

TODO add index. Don't want to load the entire just to check its description.

'''

# only update this if the database storage methods change
__version__ = "0.0.4"

import numpy as np
import shutil
import zipfile
import io
import os
import pickle as pl
import itertools
import time
import matplotlib.pyplot as plt
import pprint
import re

import collections

import requests
import bs4

import matplotlib_funs as mpf
mpf.style_use('default')

import general_funs as gf
import rp_funs as rpf
import ld_funs as ldf

home_dir = os.path.expanduser("~")
labc_dir = os.path.join(home_dir, 'labc')
labc_url = "http://localhost:8000/labc/"

os.makedirs(labc_dir, exist_ok=True)

labc_rps = ['rp6', 'rp7', 'rp8'] 
rp_channels = ['in1', 'in2', 'iq0', 'iq1', 'iq2', 'pid0', 'pid1', 'pid2', 'out1', 'out2']
locked1_states = ['locked1_TEM00_carrier', 'locked1_TEM00_sideband', 'locked1_TEM02_carrier', 'locked1_TEM02_sideband']
locked2_states = ['locked2_TEM00_carrier']
phase_loop_states = ['locked_phase_loop']
all_locked_states = locked2_states + locked1_states + [y[0]+'_'+y[1] for y in itertools.product((locked1_states + [x[0]+'_'+x[1] for x in itertools.product(locked1_states, locked2_states)]), phase_loop_states)]
all_rp_codes = [x[0]+'_'+x[1] for x in itertools.product(labc_rps, rp_channels)]
all_measurement_codes = ['dark', 'electronic_dark', 'optical_dark', 'LO', 'sensing', 'science_mode'] + all_locked_states
all_full_codes = [x[0]+'_'+x[1] for x in itertools.product(all_rp_codes, all_measurement_codes)]
all_full_codes.extend([None, 'science_mode'])

ANY_PREVIOUS = gf.SentinelValue("ANY_PREVIOUS")
SAME_PREVIOUS = gf.SentinelValue("SAME_PREVIOUS")

# CHECK THIS CAREFULLY
LD_cav = ldf.LD1
LD_lo = ldf.LD0

LDcav = LD_cav
LDlo = LD_lo

class labc:

    dc_offset_12_TEM00_carrier = 0.03
    dc_offset_12_TEM02_sideband = -0.29 # only valid for gain_12, eom_12 = 150, 0.6
    offset_12 = 21.900989217683673

    i_gain_12_TEM00_carrier = -9
    i_gain_12_TEM02_sideband = -15
    iq_gain_12_TEM00_carrier = 40 # for TEM00 carrier
    iq_gain_12_TEM02_sideband = 150 # for TEM02 sideband
    gain_mm_12 = 300 # mode matching errsig gain
    eom_12 = 0.6

    dc_offset_17 = 0.0
    offset_17 = -111.48896955884993
    gain_17 = 600
    i_gain_17_TEM00_carrier = -2
    eom_17 = 0.1

    @classmethod
    def setup_rp6(cls, show_gui=False):
        # p6 = rpf.connect(hostname='pitaya6') # will not work if the DNS doesn't work
        p6 = rpf.connect(config='temp1', source='source_pitaya6', hostname='129.127.102.184', show_gui=show_gui)
        rp6 = p6.rp
        cls.p6 = p6
        cls.rp6 = rp6
        rpf.scope_default_setup(rp6)

        rp6.scope.input1 = 'iq0'

        rp6.asg0.free()
        rp6.asg0.frequency = 12e6
        rp6.asg0.amplitude = cls.eom_12
        rp6.asg0.trigger_source = 'immediately'
        rp6.asg0.output_direct = 'out1'

        rp6.asg1.free()
        rp6.asg1.frequency = 0.0
        rp6.asg1.amplitude = 0.0
        rp6.asg1.trigger_source = 'immediately'
        rp6.asg1.output_direct = 'out2'

        rp6.pid0.free()
        rp6.pid0.input = 'iq0'
        rp6.pid0.p = 0
        rp6.pid0.i = 0
        rp6.pid0.ival = 0
        rp6.pid0.inputfilter = [0, 0, 0, 0]
        rp6.pid0.output_direct = 'out2'

        rp6.iq0.free()
        rp6.iq0.input = 'in1'
        rp6.iq0.acbandwidth = 0
        rp6.iq0.frequency = rp6.asg0.frequency
        rp6.iq0.bandwidth = [1.5e5, 0]
        rp6.iq0.quadrature_factor = cls.iq_gain_12_TEM00_carrier
        rp6.iq0.output_direct = 'off'

        rp6.iq1.free()
        rp6.iq1.input = 'in2'
        rp6.iq1.acbandwidth = 0
        rp6.iq1.frequency = rp6.asg0.frequency
        rp6.iq1.bandwidth = [1.5e5, 0]
        rp6.iq1.quadrature_factor = cls.gain_mm_12
        rp6.iq1.output_direct = 'off'

        rp6.iq2.free()
        rp6.iq2.input = 'in2'
        rp6.iq2.acbandwidth = 0
        rp6.iq2.frequency = rp6.asg0.frequency
        rp6.iq2.bandwidth = [1.5e5, 0]
        rp6.iq2.quadrature_factor = cls.gain_mm_12
        rp6.iq2.output_direct = 'off'

        rpf.zero_demod_phase(rp6)

        cls.rp6_iq0_zero_phase = rp6.iq0.phase
        rp6.iq0.phase += cls.offset_12

        rp6.iq1.phase += 0
        rp6.iq2.phase += 90

        return p6

    @classmethod
    def setup_rp7(cls, show_gui=False):
        # p7 = rpf.connect(hostname='pitaya7') # will not work if the DNS doesn't work
        p7 = rpf.connect(config='temp2', source='source_pitaya7', hostname='129.127.102.185', show_gui=show_gui)
        rp7 = p7.rp
        cls.p7 = p7
        cls.rp7 = rp7
        rpf.scope_default_setup(rp7)

        rp7.asg0.free()
        rp7.asg0.frequency = 17e6
        rp7.asg0.amplitude = cls.eom_17
        rp7.asg0.trigger_source = 'immediately'
        rp7.asg0.output_direct = 'out1'

        rp7.asg1.free()
        rp7.asg1.frequency = 0.0
        rp7.asg1.amplitude = 0.0
        rp7.asg1.trigger_source = 'immediately'
        rp7.asg1.output_direct = 'out2'

        rp7.pid0.free()
        rp7.pid0.input = 'iq0'
        rp7.pid0.p = 0
        rp7.pid0.i = 0
        rp7.pid0.ival = 0
        rp7.pid0.inputfilter = [0, 0, 0, 0]
        rp7.pid0.output_direct = 'out2'

        rp7.pid1.free()
        rp7.pid1.input = 'in2'
        rp7.pid1.p = 1
        rp7.pid1.i = 0
        rp7.pid1.setpoint = -0.011
        rp7.pid1.ival = 0
        rp7.pid1.inputfilter = [0, 0, 0, 0]
        rp7.pid1.output_direct = 'off'

        rp7.iq0.free()
        rp7.iq0.input = 'in1'
        rp7.iq0.acbandwidth = 0
        rp7.iq0.frequency = rp7.asg0.frequency
        rp7.iq0.bandwidth = [1.5e5, 0]
        rp7.iq0.quadrature_factor = cls.gain_17
        rp7.iq0.output_direct = 'off'

        rp7.iq1.free()
        rp7.iq1.input = 'in2'
        rp7.iq1.acbandwidth = 0
        rp7.iq1.frequency = rp7.asg0.frequency
        rp7.iq1.bandwidth = [1.5e5, 0]
        rp7.iq1.quadrature_factor = 20
        rp7.iq1.output_direct = 'off'

        rp7.iq2.free()
        rp7.iq2.input = 'in2'
        rp7.iq2.acbandwidth = 0
        rp7.iq2.frequency = rp7.asg0.frequency
        rp7.iq2.bandwidth = [1.5e5, 0]
        rp7.iq2.quadrature_factor = 20
        rp7.iq2.output_direct = 'off'

        rpf.zero_demod_phase(rp7)

        cls.rp7_iq0_zero_phase = rp7.iq0.phase
        rp7.iq0.phase += cls.offset_17

        return p7

    @classmethod
    def setup_rp8(cls, show_gui=False):
        # p8 = rpf.connect(hostname='pitaya8') # will not work if the DNS doesn't work
        p8 = rpf.connect(config='temp3', source='source_pitaya8', hostname='129.127.102.186', show_gui=show_gui)
        rp8 = p8.rp
        cls.p8 = p8
        cls.rp8 = rp8
        rpf.scope_default_setup(rp8)

        rp8.asg0.free()
        rp8.asg0.frequency = 1e6
        rp8.asg0.amplitude = 0.0
        rp8.asg0.trigger_source = 'immediately'
        rp8.asg0.output_direct = 'out1'

        rp8.asg1.free()
        rp8.asg1.frequency = 0.0
        rp8.asg1.amplitude = 0.0
        rp8.asg1.trigger_source = 'immediately'
        rp8.asg1.output_direct = 'out2'

        rp8.pid0.free()
        rp8.pid0.input = 'iq0'
        rp8.pid0.p = 0
        rp8.pid0.i = 0
        rp8.pid0.ival = 0
        rp8.pid0.inputfilter = [0, 0, 0, 0]
        rp8.pid0.output_direct = 'off'

        rp8.iq0.free()
        rp8.iq0.input = 'in1'
        rp8.iq0.acbandwidth = 0
        rp8.iq0.frequency = rp8.asg0.frequency
        rp8.iq0.bandwidth = [1e5, 0]
        rp8.iq0.quadrature_factor = 20
        rp8.iq0.output_direct = 'off'

        rp8.iq1.free()
        rp8.iq1.input = 'in2'
        rp8.iq1.acbandwidth = 0
        rp8.iq1.frequency = rp8.asg0.frequency
        rp8.iq1.bandwidth = [1e5, 0]
        rp8.iq1.quadrature_factor = 20
        rp8.iq1.output_direct = 'off'

        rp8.iq2.free()
        rp8.iq2.input = 'in2'
        rp8.iq2.acbandwidth = 0
        rp8.iq2.frequency = rp8.asg0.frequency
        rp8.iq2.bandwidth = [1e5, 0]
        rp8.iq2.quadrature_factor = 20
        rp8.iq2.output_direct = 'off'

        rpf.zero_demod_phase(rp8)

        return p8

def assert_measurement_code(code):
    if code not in all_full_codes:
        raise Exception(f"Code \"{code}\" doesn't match any known measurement codes. Fix or pass force=True.")

def print_measurement(load_dict):
    print('name:', load_dict['name'])
    print('timestamp:', load_dict['timestamp'])
    print('description:', gf.default_key(load_dict, 'description', default=''))
    print('related:', gf.default_key(load_dict, 'related', default=''))

def extract_timestamp_from_filename(filename):
    '''
    Timestamp should be at the end of the filename.
    Ignore the .zip on the end.
    '''
    return filename[-23:-4]

def save_measurement(obj, meas_code, description='', related=None, force=False, debug=False):
    if not force:
        assert_measurement_code(meas_code)
    unixtime = time.time()
    timestamp = gf.get_timestamp(unixtime)
    stamped_name = meas_code + "_" + timestamp
    abs_path = os.path.join(labc_dir, stamped_name)
    file_dir = abs_path
    os.makedirs(file_dir, exist_ok=True)
    pickle_path = os.path.join(file_dir, stamped_name)

    meta_dict = {'name': meas_code, 'labc_ver': __version__, 'unixtime': unixtime, 
    'timestamp': timestamp, 'data': obj, 'description': description, 'related': related}
    if debug:
        print(pickle_path)
    with open(pickle_path + ".pl", 'wb') as f:
        pl.dump(meta_dict, f)

    shutil.make_archive(file_dir, 'zip', file_dir)
    shutil.rmtree(file_dir)
    print('written: ', file_dir+'.zip')

def load_measurement(filename, silent=False):
    filepath = labc_url + filename
    r = requests.get(filepath)
    if not silent:
        print("loaded: ", filepath)
    return gf.extract_pickles_from_zip(file=io.BytesIO(r.content))

def load_measurements(filenames, silent=False):
    return [load_measurement(x, silent) for x in filenames]

def labc_url_ls(debug=False):
    r = requests.get(labc_url)
    soup = bs4.BeautifulSoup(r.text, 'html.parser')
    return [x.get('href') for x in soup.find_all('a')]

def load_latest_measurement(meas_code=None, debug=False, offset=0):
    labc_ls = labc_url_ls()
    offset = np.abs(offset)
    if debug:
        print('meas_code = ', meas_code)
        print('labc_ls')
        pprint.pprint(labc_ls[0:10])
    if meas_code is not None:
        filtered_codes = [x[0:len(meas_code)] for x in labc_ls]
        filtered_ls = [x for x in labc_ls if x[0:len(meas_code)] == meas_code]
        if debug:
            print('filtered_codes = ', filtered_codes[0:10])
            print('filtered_ls = ', filtered_ls[0:10])
        sorted_ls = sorted(filtered_ls, reverse=True)
    else:
        filtered_ls = labc_ls
        sorted_ls = sorted(filtered_ls, key=extract_timestamp_from_filename, reverse=True)
    if debug:
        print('sorted_ls')
        pprint.pprint(sorted_ls[0:10])
    return load_measurement(sorted_ls[offset])

def load_latest_measurements(codes, offset=0, debug=False, force=False):
    if not force:
        for code in codes:
            assert_measurement_code(code)

    offsets = np.array(compute_code_offsets(codes))
    offsets += offset

    return [load_latest_measurement(meas_code=x, offset=y, debug=debug) for x,y in zip(codes, offsets)]

def extract_spectrum(load_dict):
    return load_dict['data']['spectrum']

def load_latest_spectrum(meas_code=None, debug=False, **kwargs):
    return extract_spectrum(load_latest_measurement(meas_code=meas_code, debug=debug, **kwargs))

def compute_code_offsets(codes):
    '''
    If multiple identical codes are given just generate offsets to allow loading multiple
    measurements of the same type from load_latest type functions.
    '''
    seen = collections.defaultdict(lambda: -1)
    offsets = []
    for code in codes:
        seen[code] += 1
        offsets.append(seen[code])
    return offsets           

def plot_latest_spectra(codes, offset=0, debug=False):
    for code in codes:
        assert_measurement_code(code)

    offsets = np.array(compute_code_offsets(codes))
    offsets += offset

    if debug:
        print(offsets)

    fig, ax = plt.subplots(1,1,figsize=[12,6])
    for code, offset in zip(codes, offsets):
        if debug:
            print(code, offset)
        load_dict = load_latest_measurement(code, offset=offset)
        ax.loglog(*load_dict['data']['spectrum'], label=load_dict['name']+' '+load_dict['timestamp'])
    ax.legend()
    ax.set_title(gf.get_timestamp())
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"PSD [V$^2$ / Hz]")
    return fig, ax

def parse_measurement_filename(filename):
    parse_dict = {}
    re_expr = '(.*)[_](.{10})[_](.{8})[.](.*)$'
    re_match = re.search(re_expr, filename)
    parse_dict['filename'] = re_match[0]
    parse_dict['meas_code'] = re_match[1]
    parse_dict['date'] = re_match[2]
    parse_dict['time'] = re_match[3]
    parse_dict['ext'] = re_match[4]
    return parse_dict

def filter_measurement_filenames(filenames, meas_code=None, date=None, time=None, ext=None):
    parse_dicts = list(map(lambda x: parse_measurement_filename(x), filenames))
    filtered_filenames = []
    for x in parse_dicts:
        filt_cond = gf.default_eq(x['meas_code'], meas_code) and gf.default_eq(x['date'], date) and gf.default_eq(x['time'], time) and gf.default_eq(x['ext'], ext)
        if filt_cond:
            filtered_filenames.append(x['filename'])
    return filtered_filenames

def take_liquid_lens_scan(currents, LD_name, rp, T=0.067108864):

    rp.scope.average = True
    rp.scope.input1 = 'iq1'
    rp.scope.input2 = 'iq2'
    rp.scope.duration = T
    rp.scope.trigger_source = 'immediately'

    LD_addr = eval(LD_name)

    arrs = []

    with ldf.LiquidLens(LD_addr) as ld:
        for current in currents:
            gf.inplace_print(current)
            ldcav_set_curr = ldf.set_current(ld, current)
            ldcav_read_curr = ldf.read_current(ld)
            ldcav_temp = ldf.read_temperature(ld)

            fut = rp.scope.curve_async()

            gf.polled_sleep(T*1 + 0.05)
            # Retrieve data from all rps immediately. They should be ready.
            arr = fut.await_result(timeout=0.001)
            arrs.append(arr)
    return arrs


def take_mode_matching_scan(current, rp6, rp7, rp8, T=0.067108864, TEM00_scan=False, debug=False):
    out_dict = {}

    rp6.scope.average = True
    rp6.scope.duration = T
    rp6.scope.input1 = 'iq1'
    rp6.scope.input2 = 'iq2'
    rp6.scope.trigger_source = 'immediately'

    rp7.scope.average = True
    rp7.scope.duration = T
    rp7.scope.input1 = 'iq0'
    rp7.scope.input2 = 'in2'
    rp7.scope.trigger_source = 'immediately'

    rp8.scope.average = True
    rp8.scope.duration = T
    rp8.scope.input1 = 'in1'
    rp8.scope.input2 = 'in2'
    rp8.scope.trigger_source = 'immediately'

    if TEM00_scan:
        rp6.iq0.quadrature_factor = labc.iq_gain_12_TEM00_carrier
        rp6_lock_TEM00_carrier(rp6=rp6)
    else:
        rp6.iq0.quadrature_factor = labc.iq_gain_12_TEM02_sideband
        rp6_lock_TEM02_sideband(rp6=rp6)

    # use liquid lens I/O to provide a time delay
    # for rp6 to finalize lock
    with ldf.LiquidLens(LD_cav) as ld:
        ldcav_set_curr = ldf.set_current(ld, current)
        ldcav_read_curr = ldf.read_current(ld)
        ldcav_temp = ldf.read_temperature(ld)
        
    with ldf.LiquidLens(LD_lo) as ld:
        ldlo_read_curr = ldf.read_current(ld)
        ldlo_temp = ldf.read_temperature(ld)

    # rp7 and rp8 only depend on rp6 being stable
    rp7_lock_TEM00_carrier(rp7=rp7)
    rp8_lock_phase_loop(rp8=rp8)

    if debug:
        print(rp6.pid0.i, rp7.pid0.i, rp8.pid0.i)

    # give time for locks to settle before grabbing data
    gf.polled_sleep(0.03)

    # Ask all rps to grab data asynchronously.
    fut6 = rp6.scope.curve_async()
    fut7 = rp7.scope.curve_async()
    fut8 = rp8.scope.curve_async()
    # Sleep the required amount of time.
    gf.polled_sleep(T*1.1 + 0.05)
    # Retrieve data from all rps immediately. They should be ready.
    arr6 = fut6.await_result(timeout=0.001)
    arr7 = fut7.await_result(timeout=0.001)
    arr8 = fut8.await_result(timeout=0.001)

    # Package output
    out_dict['hom_i'] = arr6[0]
    out_dict['hom_q'] = arr6[1]
    out_dict['PDH17'] = arr7[0]
    out_dict['trans'] = arr7[1]
    out_dict['phase_errsig'] = arr8[1]
    out_dict['phase_DC'] = arr8[0]
    out_dict['ld_cav_current'] = ldcav_read_curr
    out_dict['ld_cav_temp'] = ldcav_temp
    out_dict['ld_lo_current'] = ldlo_read_curr
    out_dict['ld_lo_temp'] = ldlo_temp

    rpf.scope_default_setup(rp6)
    rpf.scope_default_setup(rp7)
    rpf.scope_default_setup(rp8)

    rp8_lock_low_gain(rp8)
    rp8.pid0.ival = 0

    return out_dict

def rp6_lock_TEM00_carrier(rp6):
    rpf.adjust_pid_i_gain(rp6, 'pid0', labc.i_gain_12_TEM00_carrier)

def rp6_lock_TEM02_sideband(rp6):
    rpf.adjust_pid_i_gain(rp6, 'pid0', labc.i_gain_12_TEM02_sideband)

def rp7_lock_TEM00_carrier(rp7):
    rpf.adjust_pid_i_gain(rp7, 'pid0', labc.i_gain_17_TEM00_carrier)

def rp8_lock_phase_loop(rp8):
    rpf.adjust_pid_i_gain(rp8, 'pid0', 10)

def rp6_lock_low_gain(rp6):
    rpf.adjust_pid_i_gain(rp6, 'pid0', 0.1)

def rp7_lock_low_gain(rp7):
    rpf.adjust_pid_i_gain(rp7, 'pid0', 0.01)

def rp8_lock_low_gain(rp8):
    rpf.adjust_pid_i_gain(rp8, 'pid0', 0.1)
