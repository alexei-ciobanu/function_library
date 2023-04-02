import numpy as np
import time
import labc_funs as labc
import general_funs as gf
import pyqtgraph_funs as pyqtgf
import os

import pyrpl
from pyrpl.async_utils import sleep

def connect(config='', source='', hostname='pitaya6', show_gui=False):
    gf.try_remove(os.path.join(gf.home_dir, 'pyrpl_user_dir', 'config', config+'.yml'))
    gf.try_remove(os.path.join(gf.home_dir, 'pyrpl_user_dir', 'config', config+'.yml.bak'), silent=True)
    p = pyrpl.Pyrpl(config=config, source=source, hostname=hostname)
    if not show_gui:
        p.hide_gui()
    return p

def setup(rp, module, **kwds):
    for key in kwds:
        setattr(getattr(rp,module), key, kwds[key])

def scope_default_setup(rp):
    rp.scope.average = True
    rp.scope.trigger_source = 'immediately'
    rp.scope.running_state = 'running_continuous'
    rp.scope.duration = 8
    return rp.scope.setup_attributes

def wait_for_done(fut):
    while not fut.done():
        sleep(0.001)

def _take_spectrum(rp, durations, drop=1, debug=False):
    metas = []
    out = []
    for i,d in enumerate(durations):
        rp.scope.duration = d
        print(i,d)
        fut = rp.scope.curve_async()
        metas.append(get_rp_meta(rp))
        # try:
        #     gf.polled_sleep(d*1.1)
        # except KeyboardInterrupt:
        #     raise
        wait_for_done(fut)
        arr = fut.await_result(timeout=1e-9)
        if debug:
            print('arr.shape : ', arr.shape)
        out.append(arr[:,drop:])
    return out, metas

def _take_spectrum2(rp, durations, debug=False):
    metas = []
    out = []
    for i,d in enumerate(durations):
        print(i,d)
        # print((i != 0) and (d != rp.scope.duration_options[-1]))
        if (i != 0) and (d != rp.scope.duration_options[-1]):
            # print('sleeping')
            rp.scope.duration = d
            fut = rp.scope.curve_async()
            # try:
            #     gf.polled_sleep(d*1.1)
            # except KeyboardInterrupt:
            #     raise
            wait_for_done(fut)
            arr = fut.await_result(timeout=1e-9)
        else:
            arr = rp.scope.data_avg
            # data avg returns a couple of the first elements as none
            # remove them and preserve structure
            arr = np.vstack([arr[0][~np.isnan(arr[0])], arr[1][~np.isnan(arr[1])]])
        metas.append(get_rp_meta(rp))
        out.append(arr)
    return out, metas

def take_spectrum(rp, meas_code=None, input1=None, input2=None, skip_first_sleep=True, 
durations=None, description='', save=False, safe=False, debug=False):

    if meas_code is None and save is True:
        raise Exception('Must provide a measurement code in order to save. \
            Saving can be disabled by passing save=False')

    if meas_code is not None and save is False:
        raise Exception('You probably want to pass save=True.')

    rp_name = 'rp'+rp.parent.name[-1]

    if durations is None:
        durations = rp.scope.durations
        durations = sorted(durations, reverse=True)

    fs_sample = 1/np.array(durations)*2**14
    fs_min = 1/np.array(durations)

    if input1 is not None:
        rp.scope.input1 = input1
    if input2 is not None:
        rp.scope.input2 = input2
    # rp.scope.average = True
    # rp.scope.trigger_source = "immediately"

    if save:
        # assert valid measurement code before wasting time on taking measurements
        code0 = rp_name+'_'+rp.scope.input1+'_'+meas_code
        code1 = rp_name+'_'+rp.scope.input2+'_'+meas_code

        labc.assert_measurement_code(code0)
        labc.assert_measurement_code(code1)

    if safe:
        out, metas = _take_spectrum(rp, durations, debug=debug)
    else:
        out, metas = _take_spectrum2(rp, durations, debug=debug)

    # reset scope to default state
    scope_default_setup(rp)
        
    print('making spectrum')

    fnew = np.logspace(np.log10(np.min(fs_min)), np.log10(np.max(fs_sample)/2), 1000)
    Ys = gf.welch_spectra_stitch(out, fs_sample, fnew, nperseg=2**14/8, debug=debug)

    if debug:
        print('Ys.shape', Ys.shape)

    save_dict0 = {'spectrum': (fnew, Ys[0]), 'data': [x[0] for x in out], 'meta': metas}
    save_dict1 = {'spectrum': (fnew, Ys[1]), 'data': [x[1] for x in out], 'meta': metas}

    if save:
        labc.save_measurement(save_dict0, code0, description=description)
        labc.save_measurement(save_dict1, code1, description=description)

    print('done')

    return save_dict0, save_dict1


def get_rp_meta(rp):
    meta_dict = {}
    meta_dict['scope'] = rp.scope.setup_attributes
    meta_dict['asg0'] = rp.asg0.setup_attributes
    meta_dict['asg1'] = rp.asg1.setup_attributes
    meta_dict['iq0'] = rp.iq0.setup_attributes
    meta_dict['iq1'] = rp.iq1.setup_attributes
    meta_dict['iq2'] = rp.iq2.setup_attributes
    meta_dict['pid0'] = rp.pid0.setup_attributes
    meta_dict['pid1'] = rp.pid1.setup_attributes
    meta_dict['pid2'] = rp.pid2.setup_attributes
    return meta_dict

def grab_scope_data(rp, in1=None, in2=None, duration=0.001, trigger='immediately', timeout=None):
    old_in1 = rp.scope.input1
    old_in2 = rp.scope.input2
    old_duration = rp.scope.duration
    old_trigger_source = rp.scope.trigger_source
    
    if in1 is not None:
        rp.scope.input1 = in1
        
    if in2 is not None:
        rp.scope.input2 = in2
        
    rp.scope.duration = duration
    rp.scope.average = True
    rp.scope.trigger_source = trigger
    actual_duration = rp.scope.duration
    fut = rp.scope.curve_async()
    
    if timeout is None:
        timeout = actual_duration*3
    
    arr = fut.await_result(timeout=timeout)
    
    rp.scope.input1 = old_in1
    rp.scope.input2 = old_in2
    rp.scope.duration = old_duration
    rp.scope.trigger_source = old_trigger_source
    
    return arr

def adjust_pid_i_gain(rp, pid='pid0', new_i_gain=None):
    rp_pid_sign = np.sign(getattr(rp, pid).i)
    setattr(getattr(rp, pid), 'i', rp_pid_sign*abs(new_i_gain))
    return getattr(rp, pid).i
        
def zero_demod_phase(rp, channels=['iq0','iq1','iq2'], set_new_demod=True, debug=False):

    demod_freq = rp.asg0.frequency

    if demod_freq == 0:
        raise Exception('asg0 frequency cannot be zero. Change it before calling zero_demod_phase')
    
    def find_demod_sign(chan, p):
        '''
        find if we are on the positive or negative slope of the sine wave
        '''
        getattr(rp,chan).phase += 5
        p1 = rp.scope.voltage_in1

        getattr(rp,chan).phase -=5
        if p1 < p:
            return 1
        else:
            return -1
       
    new_demods = []
    for chan in channels:

        old_scope_setup = rp.scope.setup_attributes
        old_asg0_setup = rp.asg0.setup_attributes
        old_iq_setup = getattr(rp, chan).setup_attributes

        getattr(rp,chan).free()
        
        rp.scope.input1 = chan
        rp.asg0.output_direct = 'off'
        rp.asg0.frequency = demod_freq
        rp.asg0.amplitude = 1
        getattr(rp,chan).quadrature_factor = 1
        getattr(rp,chan).bandwidth = [demod_freq/1e2, demod_freq/1e2]
        getattr(rp,chan).frequency = rp.asg0.frequency
        getattr(rp,chan).input = 'asg0'
    
        p1 = rp.scope.voltage_in1
        sign = find_demod_sign(chan, p1)
        d1 = getattr(rp,chan).phase + sign*(np.arcsin(2*p1)/np.pi*180)
#         d2 = getattr(rp,chan).phase + np.arccos(2*p2)/np.pi*180 - 90

        v1 = rp.scope.voltage_in1
        zero_sign = find_demod_sign(chan, v1)
        if debug:
            print('before', chan, v1, zero_sign)
        if zero_sign == -1:
            if debug:
                print('adding 180 to ', chan)
            d1 += 180     
    
        new_demods.append(d1)

        setup(rp, chan, **old_iq_setup)
        rp.scope.input1 = old_scope_setup['input1']
        # have to reset asg carefully (setting trigger_source resets clock)
        setup(rp, 'asg0', **gf.filter_dict(old_asg0_setup, 'trigger_source'))

        if set_new_demod:
            getattr(rp,chan).phase = d1

            if debug:
                time.sleep(0.1)
                v1 = rp.scope.voltage_in1
                print('after', chan, v1, find_demod_sign(chan, v1))
    
    return new_demods


def rp6_cavity_scan(rp6, timeout=1):
    ## release the PID lock
    rp6.pid0.p = 0
    rp6.pid0.i = 0
    rp6.pid0.ival = 0
    rp6.pid0.output_direct = 'off'
    
    rp6.asg1.free()
    rp6.asg1.frequency = 14
    rp6.asg1.offset = 0
    rp6.asg1.amplitude = 0.5
    rp6.asg1.output_direct = 'out2'
    
    data = grab_scope_data(rp6, in1='in1', in2='in2', trigger='asg1', timeout=timeout)
    rp6.asg1.amplitude = 0
    
    return data

def rp_cavity_scan(rp, duration=None, timeout=1):
    ## release the PID lock
    rp.pid0.p = 0
    rp.pid0.i = 0
#     rp.pid0.ival = 0
    rp.pid0.output_direct = 'off'
    
    rp.asg1.free()
    rp.asg1.output_direct = 'off'
    rp.asg1.offset = 0
    rp.asg1.amplitude = 0.0
    rp.asg1.output_direct = 'out2'
    
    data = grab_scope_data(rp, in1='in1', in2='asg1', trigger='asg1', duration=duration, timeout=timeout)
    
    return data

def rp_cavity_scan_oneoff(rp, frequency=15, duration=3/14, amplitude=0.5, offset=0, timeout=1):
    ## release the PID lock
    rp.pid0.p = 0
    rp.pid0.i = 0
#     rp.pid0.ival = 0
    rp.pid0.output_direct = 'off'
    
    rp.asg1.free()
    rp.asg1.output_direct = 'off'
    rp.asg1.frequency = frequency
    rp.asg1.offset = 0
    rp.asg1.amplitude = 0.0
    rp.asg1.output_direct = 'out2'
    ramp_amplitude(rp.asg1, amplitude)
    
    data = grab_scope_data(rp, in1='in1', in2='asg1', trigger='asg1', duration=duration, timeout=timeout)
    ramp_amplitude(rp.asg1, 0)
    
    return data

def ramp_amplitude(asg, target, debug=False):
    if debug:
        print('beginning ramp')
    current = asg.amplitude
    step = (target-current)/20
    for i in range(20):
        time.sleep(0.04)
        asg.amplitude += step
        if debug:
            print(asg.amplitude)
    time.sleep(0.04)
    asg.amplitude = target

def setup_ramp_scan(rp, T=None, asg='asg1', amplitude=0.1, out=None):
    if T is None:
        T = rp.scope.duration
        
    rp_asg = getattr(rp, asg)
        
    rp_asg.amplitude = amplitude
    rp_asg.frequency = (10/9) / (2*T)
    rp_asg.offset = 0
    rp_asg.waveform = 'ramp'
    rp_asg.trigger_source = 'immediately'
    
    if out is not None:
        rp_asg.output_direct = out
    
    rp.scope.input1 = asg
    rp.scope.average = True
    rp.scope.trigger_source = asg
    rp.scope.trigger_delay = (9/10) * (T/2)

def setup_pyqtgf_scope(rp, T=0.05):
    # setup futures and xaxis

    if T is not None:
        rp.scope.duration = T
    T = rp.scope.duration

    state_dict = {}
    fut = rp.scope.curve_async()
    xaxis = np.linspace(0, rp.scope.duration, 2**14)
    wait_for_done(fut)
    arr = fut.await_result(1e-9)
    
    state_dict['arr'] = arr
    state_dict['xaxis'] = xaxis
    state_dict['T'] = rp.scope.duration

    state_dict['rp'] = rp
    state_dict['fut'] = rp.scope.curve_async()
    state_dict['max_refresh_rate'] = 30
    # state_dict['sleep_factor'] = 2
    state_dict['running'] = True
    
    # def prepare_update(state_dict):  
    #     def update():
    #             c1, c2 = state_dict['plot_dict']['plot'].curves
    #             try:
    #                 arr = state_dict['fut'].await_result(0.001)
    #                 pyqtgf.set_curve_ydata(c1, arr[0,:])
    #                 pyqtgf.set_curve_ydata(c2, arr[1,:])
    #                 state_dict['fut'] = state_dict['rp'].scope.curve_async()
    #             except Exception as e:
    #                 state_dict['exception'] = e
    #                 state_dict['stop']()

    #             if state_dict['running']:
    #                 T = rp.scope.duration
    #                 state_dict['timer_sleep'] = get_timer_sleep(T, state_dict)
    #                 state_dict['start']()
    #     return update

    def prepare_update(state_dict):  
        def update():
            if state_dict['running'] and state_dict['fut'].done():
                c1, c2 = state_dict['plot_dict']['plot'].curves
                try:
                    arr = state_dict['fut'].await_result(1e-9)
                    state_dict['arr'] = arr
                    state_dict['fut'] = state_dict['rp'].scope.curve_async()

                    T = state_dict['rp'].scope.duration
                    if T != state_dict['T']:
                        state_dict['T'] = T
                        state_dict['xaxis'] = np.linspace(0, T, 2**14)
                        pyqtgf.set_curve_data(c1, x=state_dict['xaxis'], y=arr[0,:])
                        pyqtgf.set_curve_data(c2, x=state_dict['xaxis'], y=arr[1,:])
                    else:
                        pyqtgf.set_curve_ydata(c1, arr[0,:])
                        pyqtgf.set_curve_ydata(c2, arr[1,:])

                except Exception as e:
                    print(e)
                    state_dict['exception'] = e
                    state_dict['stop']()

            state_dict['start']()
        return update

    # def get_timer_sleep(T, state_dict):
    #     trigger_asg = state_dict['rp'].scope.trigger_source
    #     if trigger_asg in ['asg0', 'asg1']:
    #         trigger_timer = 1/getattr(state_dict['rp'], trigger_asg).frequency
    #     else:
    #         trigger_timer = 0
    #     wait_timers = [
    #         1/state_dict['max_refresh_rate'], 
    #         T*state_dict['sleep_factor'], 
    #         trigger_timer*state_dict['sleep_factor']]
    #     wait_source_idx = np.argmax(wait_timers)
    #     wait_timer_sources = ['max_refresh_rate', 'scope_duration', 'trigger']
    #     state_dict['wait_timer_source'] = wait_timer_sources[wait_source_idx]
    #     return int(wait_timers[wait_source_idx]*1e3)

    plot_dict = pyqtgf.plot(xaxis, arr[0])
    state_dict['plot_dict'] = plot_dict
    pyqtgf.add_curve(plot_dict['plot'], xaxis, arr[1]) 
    plot_dict['window'].setWindowTitle(rp.parent.name)

    timer = pyqtgf.QtCore.QTimer()
    # timer_sleep = get_timer_sleep(T, state_dict)
    state_dict['timer_sleep'] = 1/state_dict['max_refresh_rate']
    state_dict['timer'] = timer
    
    state_dict['update'] = prepare_update(state_dict)

    def start(continuous=False):
        state_dict['log'] = f"running timer with {state_dict['timer_sleep']}"
        if continuous:
            state_dict['running'] = True
        state_dict['timer'].singleShot(state_dict['timer_sleep'], state_dict['update'])

    def restart():
        T = state_dict['rp'].scope.duration
        # state_dict['timer_sleep'] = get_timer_sleep(T, state_dict)
        state_dict['fut'] = state_dict['rp'].scope.curve_async()
        state_dict['start'](True)

    def stop():
        state_dict['running'] = False

    state_dict['start'] = start
    state_dict['restart'] = restart
    state_dict['stop'] = stop
    start()

    return state_dict