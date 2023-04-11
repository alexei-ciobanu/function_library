import numpy as np
import optics_funs as optics_funs
import numerical_funs as nf
import debug_funs as dgf
import pykat
import pykat.exceptions as pkex

class pykat_types:
    # Dan claims that sticking all the types in one place is indicative of how 
    # I don't know how python "works"
    lock = pykat.commands.lock
    cav = pykat.commands.cavity
    func = pykat.commands.func
    constant = pykat.commands.Constant
    pd = pykat.detectors.pd # gets pd1s as well
    ad = pykat.detectors.ad
    node = pykat.node_network.Node

def katrun_to_kat(katrun, verbose=False):
    kat = pykat.finesse.kat()
    kat.verbose = verbose
    kat.parse(katrun.katScript)
    return kat

def build_linear_cavity_kat(ns):
    kat = pykat.finesse.kat()
    code = f'''
    l laser {ns.P} 0 0 n0
    s dummy0 0 n0 n1a
    bs1 mismatch_mirror 0 0 0 0 n1a n1b dump dump
    s dummy1 0 n1b n2b
    bs1 parity_mirror 0 0 0 0 n2b n2c dump dump
    s dummy 0 n2c n2

    m1 IC {ns.m1.T} 0 0 n2 n3
    s space {ns.d} n3 n4
    m1 OC {ns.m2.T} 0 0 n4 n5

    attr IC Rcx {-ns.m1.Rcx}
    attr IC Rcy {-ns.m1.Rcy}

    attr OC Rcx {ns.m2.Rcx}
    attr OC Rcy {ns.m2.Rcy}

    attr mismatch_mirror xbeta {ns.xbeta}
    attr mismatch_mirror ybeta {ns.ybeta}

    gauss* mygauss mismatch_mirror n1b {np.real(ns.qx)} {np.imag(ns.qx)} {np.real(ns.qy)} {np.imag(ns.qy)}

    cav mycav IC n3 OC n4

    pd circ n3 
    pd trans n4
    pd inc n2*
    pd refl n2

    maxtem {ns.maxtem}
    phase {ns.phase_setting}

    yaxis re:im
    xaxis IC phi lin {ns.phi.min} {ns.phi.max} {ns.phi.N - 1}
    '''
    kat.parse(code)
    return kat

def build_linear_cavity_kat(ns, debug=False):
    kat = pykat.finesse.kat()
    code = ''
    code += f'l laser {ns.P} 0 0 n0' if dgf.attr_exists(ns, 'P') else 'l laser 1 0 0 n0'
    code += f'''
s dummy0 0 n0 n1a
bs1 mismatch_mirror 0 0 0 0 n1a n1b dump dump
s dummy1 0 n1b n2b
bs1 parity_mirror 0 0 0 0 n2b n2c dump dump
s dummy 0 n2c n2

m1 IC {ns.m1.T} 0 0 n2 n3
s space {ns.d} n3 n4
m1 OC {ns.m2.T} 0 0 n4 n5

cav mycav IC n3 OC n4

pd circ n3 
pd trans n4
pd inc n2*
pd refl n2

yaxis re:im
'''
    code += f'attr IC Rcx {-ns.m1.Rcx}\n' if dgf.attr_exists(ns, 'm1.Rcx') else ''
    code += f'attr IC Rcy {-ns.m1.Rcy}\n' if dgf.attr_exists(ns, 'm1.Rcy') else ''
    
    code += f'attr OC Rcx {ns.m2.Rcx}\n' if dgf.attr_exists(ns, 'm2.Rcx') else ''
    code += f'attr OC Rcy {ns.m2.Rcy}\n' if dgf.attr_exists(ns, 'm2.Rcy') else ''
    
    code += f'attr mismatch_mirror xbeta {ns.xbeta}\n' if dgf.attr_exists(ns, 'xbeta') else ''
    code += f'attr mismatch_mirror ybeta {ns.ybeta}\n' if dgf.attr_exists(ns, 'ybeta') else ''
    
    # code += f'attr IC r_ap {ns.TM_diam/2}\n' if dgf.attr_exists(ns, 'TM_diam') else ''
    # code += f'attr OC r_ap {ns.TM_diam/2}\n' if dgf.attr_exists(ns, 'TM_diam') else ''
    
    code += f'gauss* mygauss mismatch_mirror n1b {np.real(ns.qx)} {np.imag(ns.qx)} {np.real(ns.qy)} {np.imag(ns.qy)}\n'
    
    code += f'xaxis IC phi lin {ns.phi.min} {ns.phi.max} {ns.phi.N - 1}\n' if dgf.attr_exists(ns, 'ybeta') else 'noxaxis\n'
    code += f'maxtem {ns.maxtem}\n' if dgf.attr_exists(ns, 'maxtem') else 'maxtem 0\n'
    code += f'phase {ns.phase_setting}\n' if dgf.attr_exists(ns, 'phase_setting') else 'phase 2\n'
    
    if debug:
        return code
    
    kat.parse(code)
    return kat

def build_psuedo_ring_cavity_kat(ns):
    '''
    Takes the same namespace as build_linear_cavity_kat()
    By psuedo ring cavity I mean that it's basically a linear cavity with an extra parity.
    '''
    kat = pykat.finesse.kat()
    code = f'''
    l laser {ns.P} 0 0 n0
    s dummy0 0 n0 n1a
    bs1 mismatch_mirror 0 0 0 0 n1a n1b dump dump
    s dummy1 0 n1b n2e
    bs1 parity_mirror 0 0 0 0 n2e n2c dump dump
    s dummy2 0 n2c n2

    bs1 IC {ns.m1.T} 0 0 0 n2 n2r n2t n2b
    s space {ns.d} n2t n3
    bs1 OC {ns.m2.T} 0 0 0 n3 n3r n3t n3b
    s space2a {ns.d/2} n3r n4r
    bs1 MC 0 0 0 0 n4 n4r n4t n4b
    s space2b {ns.d/2} n2b n4

    attr IC Rcx {-ns.m1.Rcx}
    attr IC Rcy {-ns.m1.Rcy}

    attr OC Rcx {ns.m2.Rcx}
    attr OC Rcy {ns.m2.Rcy}

    attr mismatch_mirror xbeta {ns.xbeta}
    attr mismatch_mirror ybeta {ns.ybeta}

    gauss* mygauss mismatch_mirror n1b {np.real(ns.qx)} {np.imag(ns.qx)} {np.real(ns.qy)} {np.imag(ns.qy)}

    cav mycav IC n2t IC n2b

    pd circ n2t
    pd trans n4
    pd inc n2*
    pd refl n2

    maxtem {ns.maxtem}
    phase {ns.phase_setting}

    yaxis re:im
    xaxis IC phi lin {ns.phi.min} {ns.phi.max} {ns.phi.N - 1}
    '''
    kat.parse(code)
    return kat

def get_component_node(kat, component, node_index=0):
    '''Pull out a global node from a component and relative node index'''
    if isinstance(component, str):
        comp = kat.components[component]
    else:
        comp = component
    return comp.nodes[node_index]

def get_q_from_kat(kat, component=None, node_index=0, node=None, both_qx_qy=False):
    '''Determine the current q parameter at a particular node. Can either specify the 
    node directly or specify a combo of component and index.'''
    # remove locks, frequencies, xaxis
    kat = kat.deepcopy()
    kat.verbose = False
    kat.noxaxis = True
    kat.yaxis = 're:im'
    for lock in kat.getAll(pykat.commands.lock):
        kat.remove(lock)
        
    # determine the node
    if node is None:
        if isinstance(component, str):
            comp = kat.components[component]
        else:
            comp = component

        comp_node = get_component_node(kat, comp, node_index)
        node = comp_node.name
    elif isinstance(node, pykat.node_network.Node):
        node = node.name
    elif isinstance(node, str):
        node = node
    else:
        raise Exception(f"Don't know what to do with {node}")
    
    kat.parse(f'bp {node}_q_x x q {node}')
    kat.parse(f'bp {node}_q_y y q {node}')
    
    out = kat.run()
    
    if both_qx_qy:
        return out[f'{node}_q_x'], out[f'{node}_q_y']
    else:
        return out[f'{node}_q_x']
    
def induce_mode_mismatch(kat, M, component='mismatch_mirror', node_index=0, node=None, theta=0):
    '''
    Returns a new kat object that has M mode mismatch relative to the previous q parameter
    at a specified node
    '''
    kat = kat.deepcopy()  
    if isinstance(component, str):
        comp = kat.components[component]
    else:
        comp = component
        
    comp_node = get_component_node(kat, component=comp, node_index=node_index)
    node = comp_node.name
    
    q1 = get_q_from_kat(kat, component=comp, node_index=node_index, node=node)
    q2 = optics_funs.mismatch_circle2(q1, M, theta)
    
    print(q1)
    
    kat.parse(f'gauss* {comp.name}_gauss {comp.name} {node} {q2.real} {q2.imag}')

    return kat

def pykat_gen_ad_block(node, maxtem=0, freq=0, mode_list=None):
    '''if user provides mode_list overwrite the asked maxtem'''
    s = '%%% FTblock ad_' + node + '\n'
    if mode_list is None:
        mode_list = optics_funs.mode_list2(maxtem)
    for pair in mode_list:
        n,m = pair
        ad_name = 'ad_' + node + '_' + str(int(n))+'_'+str(int(m))
        s += 'ad '+ad_name+' '+str(int(n))+' '+str(int(m))+' ' + str(freq) + ' '+node+' \n'
    s += '%%% FTend ad_' + node + '\n'
    return s

def pykat_gen_ad_block2(node, hom_list=[(0,0)], freq=0):
    '''if user provides mode_list overwrite the asked maxtem'''
    s = '%%% FTblock ad_' + node + '\n'
    for nm in hom_list:
        n,m = nm
        ad_name = 'ad_' + node + '_' + str(int(n))+'_'+str(int(m))
        s += 'ad '+ad_name+' '+str(int(n))+' '+str(int(m))+' ' + str(freq) + ' '+node+' \n'
    s += '%%% FTend ad_' + node + '\n'
    return s
    
def pykat_gen_TEM_block(input_name,df_H,maxtem):
    s = '%%% FTblock' + input_name + '_HOM\n'
    for i in range(len(df_H)):
        p,d,n,m = df_H.loc[i]
        if n+m <= maxtem:
            s += 'tem '+input_name+' '+str(int(n))+' '+str(int(m))+' '+str(p)+' '+str(d)+' \n'
    s += '%%% FTend HOM\n'
    return s

def clean_kat(kat, inplace=False):
    if not inplace:
        kat = kat.deepcopy()

    kat.noxaxis = True
    locks = kat.getAll(pykat.commands.lock)
    ads = kat.getAll(pykat.detectors.ad)
    pds = kat.getAll(pykat.detectors.pd)
    [kat.remove(lock) for lock in locks]
    [kat.remove(ad) for ad in ads]
    [kat.remove(pd) for pd in pds]

    while True:
        try:
            kat.removeLine('map')
        except pkex.BasePyKatException as e:
            break

    return kat

def hom_dict_norm(hom_dict):
    hom_dict = dict(hom_dict)
    
    s = 0
    for nm, a in hom_dict.items():
        n,m = nm
        p = np.abs(a)**2
        s += p
        
    for nm, a in hom_dict.items():
        hom_dict[nm] /= np.sqrt(s)
        
    return hom_dict

def u_nm_from_hom_dict(xs, ys=None, qx=1j, qy=None, hom_dict={(0,0):1}, finesse_norm=True, include_gouy=False):
    u_out = complex(0)
    if finesse_norm:
        hom_dict = hom_dict_norm(hom_dict)
        
    for nm, a in hom_dict.items():
        n,m = nm
        u_out += a * optics_funs.u_nm_q(xs, ys, qx, qy, n=n, m=m, include_gouy=include_gouy)
    return u_out

def pykat_gen_TEM_block2(laser_name, hom_dict={(0,0): 1}):

    s = '%%% FTblock' + laser_name + '_HOM\n'
    for nm, a in hom_dict.items():
        n,m = nm
        p = np.abs(a)**2
        if p == 0:
            d = 0
        else:
            d = np.angle(a, deg=True)
        s += 'tem '+laser_name+' '+str(int(n))+' '+str(int(m))+' '+str(p)+' '+str(d)+' \n'
    s += '%%% FTend HOM\n'

    return s

def pykat_gen_TEM_block3(kat, laser_name, hom_dict={(0,0): 1}):
    kat = clean_kat(kat)

    hom_max = np.max(np.sum(list(hom_dict.keys()), axis=1))
    kat.maxtem = hom_max

    s = pykat_gen_TEM_block2(laser_name, hom_dict)
    kat.parse(s)

    node = str(getattr(kat,'laser').nodes[0]) + '*'
    ad_str = pykat_gen_ad_block2(node, hom_list=hom_dict.keys())
    kat.parse(ad_str)
    kat.verbose = False

    katrun = kat.run()
    new_hom_dict = {k: katrun[f'ad_{node}_{k[0]}_{k[1]}'] for k in hom_dict.keys()}

    adj_hom_dict = {k: hom_dict[k]*np.conj(nf.phase(new_hom_dict[k])) for k in hom_dict.keys()}

    return pykat_gen_TEM_block2(laser_name, adj_hom_dict)
    
def pykat_BH_TEM(input_name, q1, q2, dx=0, dy=0, gammax=0, gammay=0, maxtem=10):
    H = optics_funs.BH_DHT(q1, q2, dx=dx, dy=dy, gammax=gammax, gammay=gammay, maxtem=maxtem)
    df_H = optics_funs.parse_DHT(H,sortby='n,m',power=True)
    df_H['n+m'] = df_H['n']+df_H['m']
    df_H = df_H.loc[df_H['n+m'] <= maxtem]
    df_H = df_H.drop('n+m', axis=1).reset_index(drop=True)
    return pykat_gen_TEM_block(input_name,df_H,maxtem=maxtem)

def pykat_recover_H(kat,out,node,maxtem):
    # only works in yaxis re:im
    # only works with the gen_ad_block
    kat=kat.deepcopy()
    det_list = kat.getAll(pykat.detectors.BaseDetector,'name')
    det_list = [det for det in det_list if 'ad_'+node in det]
    
#     maxtem = max([int(det[-1]) for det in det_list])
    
    H = np.zeros([maxtem+1,maxtem+1],dtype=np.complex128)
    
    for det in det_list:
        n,m = det[-2:]
        n = int(n)
        m = int(m)
        H[n,m] = out[det]
        
    return H
    
def compute_cav_params(kat):
    # computes all the params that you can from a cav command
    # ie eigmode, rt guoy, finesse, fsr 

    import pandas as pd
    
    kat = kat.deepcopy()
    kat.verbose = False
    kat.noxaxis = True
    kat.removeBlock('locks',False)
#     kat.maxtem = 0
    
    cavs = kat.getAll(pykat.commands.cavity)
    
    cp_cmds = ''
    for cav in cavs:
        cp_cmds += '''cp {0}_x_g {0} x stability
                    cp {0}_y_g {0} y stability
                    cp {0}_x_A {0} x A
                    cp {0}_y_A {0} y A
                    cp {0}_x_B {0} x B
                    cp {0}_y_B {0} y B
                    cp {0}_x_C {0} x C
                    cp {0}_y_C {0} y C
                    cp {0}_x_D {0} x D
                    cp {0}_y_D {0} y D
                    cp {0}_x_fsr {0} x fsr
                    cp {0}_y_fsr {0} y fsr
                    cp {0}_x_q {0} x q
                    cp {0}_y_q {0} y q
                    cp {0}_x_fwhm {0} x fwhm
                    cp {0}_y_fwhm {0} y fwhm
                    cp {0}_x_finesse {0} x finesse
                    cp {0}_y_finesse {0} y finesse
                    
                    yaxis re:im
                    '''.format(cav)
        
    kat.parse(cp_cmds)
    
    out = kat.run()
    
#     return kat.getAll(pykat.detectors.BaseDetector,'name')

    gs = []
    As = []
    Bs = []
    Cs = []
    Ds = []
    psi_rts = []
    FSRs = []
    fwhms = []
    finesses = []
    qs = []
    delta_fs = []
    cav_names = []
    append_ax = lambda name: [name + '_x', name + '_y']
    for cav in cavs:

        g_x = np.abs((out[cav.name+'_x_g']+1)/2)
        A_x = np.abs(out[cav.name+'_x_A'])
        B_x = np.abs(out[cav.name+'_x_B'])
        C_x = np.abs(out[cav.name+'_x_C'])
        D_x = np.abs(out[cav.name+'_x_D'])
        psi_rt_x = np.abs(2*np.arccos(np.sign(B_x)*np.sqrt(g_x)))
        FSR_x = np.abs(out[cav.name+'_x_fsr'])
        fwhm_x = np.abs(out[cav.name+'_x_fwhm'])
        finesse_x  = np.abs(out[cav.name+'_x_finesse'])
        q_x = out[cav.name+'_x_q']
        
        g_y = np.abs((out[cav.name+'_y_g']+1)/2)
        A_y = np.abs(out[cav.name+'_y_A'])
        B_y = np.abs(out[cav.name+'_y_B'])
        C_y = np.abs(out[cav.name+'_y_C'])
        D_y = np.abs(out[cav.name+'_y_D'])
        psi_rt_y = np.abs(2*np.arccos(np.sign(B_y)*np.sqrt(g_y)))
        FSR_y = np.abs(out[cav.name+'_y_fsr'])
        fwhm_y = np.abs(out[cav.name+'_y_fwhm'])
        finesse_y  = np.abs(out[cav.name+'_y_finesse'])
        q_y = out[cav.name+'_y_q']
        
        if np.abs(psi_rt_x/(2*np.pi)) > 1/2:
            delta_f_x = psi_rt_x/(2*np.pi) * FSR_x - FSR_x*np.sign(psi_rt_x)
        else:
            delta_f_x = psi_rt_x/(2*np.pi) * FSR_x
            
        if np.abs(psi_rt_y/(2*np.pi)) > 1/2:
            delta_f_y = psi_rt_y/(2*np.pi) * FSR_y - FSR_y*np.sign(psi_rt_y)
        else:
            delta_f_y = psi_rt_y/(2*np.pi) * FSR_y
            
        gs.extend([g_x, g_y])
        As.extend([A_x, A_y])
        Bs.extend([B_x, B_y])
        Cs.extend([C_x, C_y])
        Ds.extend([D_x, D_y])
        psi_rts.extend([psi_rt_x, psi_rt_y])
        FSRs.extend([FSR_x, FSR_y])
        delta_fs.extend([delta_f_x, delta_f_y])
        finesses.extend([finesse_x,finesse_y])
        qs.extend([q_x,q_y])
        fwhms.extend([fwhm_x,fwhm_y])
        cav_names.extend(append_ax(cav.name))
    
    # return pd.DataFrame({'g': [g_x, g_y], 'B': [B_x, B_y], \
                         # 'psi_rt': [psi_rt_x, psi_rt_y], 'FSR': np.array([FSR_x, FSR_y])\
                         # , 'delta_f': [delta_f_x, delta_f_y] , 'q': [q_x,q_y], 'fwhm': [fwhm_x,fwhm_y]})
                         
    return pd.DataFrame({'cav': cav_names, 'g': gs, 'A': As, 'B': Bs, \
                 'C': Cs, 'D': Ds, 'psi_rt': psi_rts, 'FSR': FSRs\
                 , 'delta_f': delta_fs , 'q': qs, 'fwhm': fwhms, 'finesse': finesses})
                 
                 
###############################################################################
#
# Lock Dragging stuff
#
###############################################################################

def get_arm_powers(_kat):
    '''
    just sticks two circ PDs and spits out their reading
    '''
    kat = _kat.deepcopy()
    kat.removeBlock('locks')
    kat.verbose=False
    kat.noxaxis=True
    
    kat.parse('''
    pd Y nETMY1
    pd X nETMX1
    
    yaxis abs
    noxaxis
    ''')
    
    out = kat.run()
    return out['X'],out['Y']

def run_locks(_kat,verbose = False):
    '''assumes there is a locks block in _kat'''
    kat = _kat.deepcopy()
    kat.noxaxis = True
    kat.verbose = False
    
    old_tunings = list(kat.IFO.get_tunings().values())
    
    if verbose:
        print(('Old tunings | ' +'{: 1.3e} | '*7).format(*old_tunings))
    
    out = kat.run()
    kat.IFO.apply_lock_feedback(out)
    
    new_tunings = list(kat.IFO.get_tunings().values())
    
    if verbose:
        print(('New tunings | ' +'{: 1.3e} | '*7).format(*new_tunings))
        
    return kat

def ffs(val, prefix=''):
    '''
    Finesse fix sign. 
    Used to compute algebraic sign flips on input variables
    to the finesse function parser.
    
    val : value to feed into parser as a function constant
    prefix : 
    '''
    if val < 0:
        if prefix == '':
            s = f'(-1)*{abs(val)}'
        if prefix == '+':
            s = f'- {abs(val)}'
        if prefix == '-':
            s = f'+ {abs(val)}'
    else:
        if prefix == '':
            s = f'{abs(val)}'
        if prefix == '+':
            s = f'+ {abs(val)}'
        if prefix == '-':
            s = f'- {abs(val)}'
    
    # if val was printed in exp form the e needs to
    # be capitalised for the parser to understand
    s = s.replace('e','E')
            
    return s

def optimize_demod_phase(_kat,dofs=['CARM', 'PRCL', 'MICH', 'SRCL'],verbose = True):
    '''
    returns a kat object with optimized demod phases
    
    _kat : assumed to be an kat IFO object with fully 
    implemented DOFs
    '''

    base = _kat.deepcopy()

    if verbose:
        print('DOF  | old phase | new phase')
        print('-'*30)

    for dof in dofs:
        kat = base.deepcopy()
        dof = kat.IFO.DOFs[dof]
        kat.noxaxis = True
        kat.removeBlock('locks')
        kat.parse( dof.fsig(fsig=1) )
        kat.parse( dof.transfer() )
        code = f"""
        maximize max {dof.transfer_name()} re {dof.transfer_name()} phase1 -1000 1000 1e-3
        """
        kat.parse(code)
        out = kat.run()

        dof.port.phase = out['max'] % 360 - 180
        if dof.quad == 'Q':
            dof.port.phase += 90
            
        old_phase = base.IFO.DOFs[dof.name].port.phase
        new_phase = dof.port.phase
        
        if verbose:
            print(f"{dof.name:4s} |  {old_phase:8.3f} | {new_phase:8.3f}")
        
        # apply the demod phase
        base.IFO.DOFs[dof.name].port.phase = dof.port.phase
        
#         need to regenerate errsigs
        errsigs_cmds = base.IFO.add_errsigs_block()

    return base

def set_components(_kat,compnts,attrs,final_state,ld_out):
    kat = _kat.deepcopy()
    
    kat = load_state(kat,ld_out,-1)
    
    # get initial state of relevant params and add their func blocks
    for compnt,attr,final_val in zip(compnts,attrs,final_state):      
        setattr(kat.components[compnt],attr,final_val)        
    
    return kat

def load_state(_kat,out,components,attributes,ind):
    '''
    Recovers a model state at any point during a lock drag.
    Useful for troubleshooting lock drags. Made with 
    lock_drag_2 in mind.
    
    _kat : initial _kat fed into lock drag
    out  : final run file from the lock drag
    components : same as what went into lock drag
    attributes : same as what went into lock drag
    ind : which index from the lock drag state to load
    '''
    kat = _kat.deepcopy()
    
    old_tunings = kat.IFO.get_tunings()
    old_tunings['PRM']  +=  out['PRM_lock'][ind]
    old_tunings['ITMX'] +=  out['ITMX_lock'][ind]
    old_tunings['ETMX'] +=  out['ETMX_lock'][ind]
    old_tunings['ITMY'] +=  out['ITMY_lock'][ind]
    old_tunings['ETMY'] +=  out['ETMY_lock'][ind]
    old_tunings['SRM']  +=  out['SRM_lock'][ind]
    kat.IFO.apply_tunings(old_tunings)
    
    for c,a in zip(components,attributes):
        set_val = out[f'f_{c}_{a}'][ind]
        setattr(getattr(kat,c),a,set_val)
    
    return kat

def run_lock_drag_2(_kat,compnts,attrs,final_state,n_steps=30,step_exponent=1,debug=False):
    '''
    _kat : kat object to perform the lock drag on. Initial param values 
    will be pulled from here. Assume kat object has errsigs and locks. 
    
    compnts : list of finesse components to include in lock drag. If several 
    attributes of the same component have to be varied the component name 
    has to be duplicated for each extra attribute
    
    attrs : list of finesse attributes to alter. These attributes correspond to
    the component from the compnts list
    
    n_steps : number of steps to take to go from initial params to final 
    params.
    
    step_exponent : > 1 will make the steps finer at the beginning and larger at the end
    < 1 will make steps larger at beginning and smaller at the end
    '''
    
    kat = _kat.deepcopy()
    
    code = '''
var dummy 0
xaxis dummy re lin 0 1 {} \n'''.format(n_steps)
    
    init_params = []
    
    # get initial state of relevant params and add their func blocks
    for compnt,attr,final_val in zip(compnts,attrs,final_state):
        
        init_val = getattr(getattr(kat.components[compnt],attr),'value')
        init_params.append(init_val)
        
        code += f'\rfunc f_{compnt}_{attr} = $x1^{step_exponent} * ({ffs(final_val,"")} {ffs(init_val,"-")}) {ffs(init_val,"+")} \n'     
     
    code += '\n'
    
    # add put block
    for compnt,attr in zip(compnts,attrs):
        code += '''\rput {0} {1} $f_{0}_{1} 
'''.format(compnt,attr)
        
    kat.parse(code)
    kat.verbose=True
    
    out = kat.run()
    
    if debug:
        return code,init_params,out,kat
    else:
        return out
