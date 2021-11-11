import functools
import warnings

import numpy as np
import scipy
import networkx as nx
from multimethod import multimethod

import general_funs as gef
import graph_funs as grf
import numerical_funs as nf
import optics_funs as of
import func_funs as ff
import LCT_funs as lctf
import operator_funs as opf
import new_types as nt

import finesse
import finesse.ligo
from finesse.cymath.homs import HGModes

def sym_eval(x):
    try:
        out = x.eval()
    except AttributeError:
        out = x
    return out

def map_eval(x, dtype=complex):
    f = np.vectorize(sym_eval, otypes=[dtype])
    return f(x)

def np_map(f, x):
    return np.vectorize(f)(x)

metaplectic_sign_flip_matmul = lambda M2, M1: finesse.symbols.Operation('metaplectic_sign', lambda x,y: of.metaplectic_sign_flip_matmul(map_eval(x), map_eval(y)), M2, M1)

def get_model(obj):
    if isinstance(obj, finesse.model.Model):
        model = obj
    else:
        model = obj._model
    return model

def get_node(model, node_name):
    if isinstance(node_name, str):
        node = gef.getattr(model, node_name)
    elif isinstance(node_name, finesse.components.Node):
        node = node_name
    elif node_name is None:
        node = None
    else:
        raise ValueError
    return node

def get_nodes(model, node_names):
    nodes = []
    for node_name in node_names:
        node = get_node(model, node_name)
        nodes.append(node)
    return nodes

def get_edge_ABCD(model, from_node, to_node, direction='x', symbolic=True):
    ni,no = get_nodes(model, [from_node, to_node])
    ABCD = model.ABCD(ni, no, direction=direction, symbolic=symbolic)
    return ABCD.M

def get_edge_scalar(component, from_node, to_node):
    if any([x is None for x in [component, from_node, to_node]]):
        return finesse.symbols.as_symbol(0j)
    model = get_model(component)
    ni,no = get_nodes(model, [from_node, to_node])

    scalar = finesse.symbols.as_symbol(1+0j)
    if isinstance(component, finesse.components.Space):
        scalar = finesse.symbols.as_symbol(1+0j)
    elif isinstance(component, finesse.components.Mirror):
        pi, po = ni.port, no.port
        mirror = ni.component
        if pi is po:
            # input port == ouput port :: reflection
            if pi is mirror.p1:
                # front reflection
                scalar = np.sqrt(mirror.R.ref) * np.exp(1j*mirror.phi.ref/90*np.pi)
            elif pi is mirror.p2:
                # back reflection
                scalar = np.sqrt(mirror.R.ref) * np.exp(-1j*mirror.phi.ref/90*np.pi)
        else:
            # input port != ouput port :: transmission
            scalar = 1j * np.sqrt(mirror.T.ref)
    elif isinstance(component, finesse.components.Beamsplitter):
        pi, po = ni.port, no.port
        bs = ni.component
        if (pi, po) == (bs.p1, bs.p2):
            scalar = np.sqrt(bs.R.ref) * np.exp(1j*bs.phi.ref/90*np.pi)
        elif (pi, po) == (bs.p1, bs.p3):
            scalar = 1j * np.sqrt(bs.T.ref)
        elif (pi, po) == (bs.p2, bs.p1):
            scalar = np.sqrt(bs.R.ref) * np.exp(1j*bs.phi.ref/90*np.pi)
        elif (pi, po) == (bs.p2, bs.p4):
            scalar = 1j * np.sqrt(bs.T.ref)
        elif (pi, po) == (bs.p3, bs.p4):
            scalar = np.sqrt(bs.R.ref) * np.exp(-1j*bs.phi.ref/90*np.pi)
        elif (pi, po) == (bs.p3, bs.p1):
            scalar = 1j * np.sqrt(bs.T.ref)
        elif (pi, po) == (bs.p4, bs.p3):
            scalar = np.sqrt(bs.R.ref) * np.exp(-1j*bs.phi.ref/90*np.pi)
        elif (pi, po) == (bs.p4, bs.p2):
            scalar = 1j * np.sqrt(bs.T.ref)
    elif isinstance(component, finesse.components.Lens):
        scalar = 1
    else:
        raise Exception(f'component {component} not recognized')
    return scalar

def get_source_scalar(model, source_node):
    source = model.optical_network.nodes[source_node]['owner']()
    # assume source is a laser for now
    source_scalar = finesse.symbols.as_symbol(1+0j)
    if isinstance(source, finesse.components.Laser):
        source_scalar = np.sqrt(source.P.ref) * np.exp(1j*source.phase.ref/180*np.pi)
    else:
        raise Exception(f'source {source} not recognized')
    return source_scalar

def get_edge(graph, n1, n2):
    if n1 is None or n2 is None:
        return None
    else:
        try:
            edge = grf.get_edge(graph, n1, n2)
        except Exception as e:
            print(n1,n2)
            raise
        return edge

def get_node_str(node_obj):
    if isinstance(node_obj, str) or node_obj is None:
        node_str = node_obj
    elif isinstance(node_obj, finesse.components.node.Node):
        node_str = node_obj.full_name
    else:
        raise Exception(f'Unknown node object {node_obj}')
    return node_str

def get_path_scalar(model, path):
    model = get_model(model)
    graph = model.optical_network
    # convert path to str to be able to index graph
    path = [get_node_str(n) for n in path]
    scalar = finesse.symbols.as_symbol(np.complex128(1))
    for n1, n2 in zip(path, path[1:]):
        edge = get_edge(graph, n1, n2)
        if edge is not None:
            component = edge['owner']()
            next_scalar = get_edge_scalar(component, n1, n2)
        else:
            next_scalar = finesse.symbols.as_symbol(0j)
        scalar *= next_scalar
    return scalar

def get_paths_scalar(model, paths):
    model = get_model(model)
    # convert path to str to be able to index graph
    paths = [[get_node_str(n) for n in path] for path in paths]
    scalar = finesse.symbols.as_symbol(0j)
    for path in paths:
        scalar += get_path_scalar(model, path)
    return scalar

def get_path_ABCDs(model, path, direction='x', symbolic=True):
    """
    Path is specified in component order (read left-to-right)
    Returns ABCDs in matmul order (read right-to-left)
    """
    model = get_model(model)
    graph = model.optical_network
    # convert path to str to be able to index graph
    path = [get_node_str(n) for n in path]
    ABCDs = []
    for node1, node2 in zip(path, path[1:]):
        ABCD = get_edge_ABCD(model, node1, node2, direction=direction, symbolic=symbolic)
        ABCDs.append(ABCD)
    ABCDs = ABCDs[::-1] # switch to matmul order
    return ABCDs

def get_path_ABCD(model, path, direction='x', symbolic=True):
    ABCDs = get_path_ABCDs(model, path, direction, symbolic)
    ABCD = functools.reduce(lambda x,y: x@y, ABCDs, np.eye(2))
    return ABCD

def get_path_metaplectic_signs(model, path, direction='x', symbolic=True):
    """
    Path is specified in component order (read left-to-right)
    Returns metaplectic_signs in matmul order (read right-to-left)
    """
    model = get_model(model)
    graph = model.optical_network
    path = [get_node_str(n) for n in path]
    ABCDs = get_path_ABCDs(model, path, direction=direction, symbolic=symbolic)
    ABCDs = ABCDs[::-1] # switch to component order for iteration
    # print(len(ABCDs))
    m1 = ABCDs[0]
    init = 0.0
    if symbolic:
        init = finesse.symbols.Constant(init)
    metapletic_signs = [init]
    for m2 in ABCDs[1:]:
        # print(ABCDs[1:])
        sign = metaplectic_sign_flip_matmul(m2,m1)
        metapletic_signs.append(sign)
        m1 = m2@m1
    metapletic_signs = np.array(metapletic_signs)
    metapletic_signs = metapletic_signs[::-1] # switch to matmul order
    return metapletic_signs

def get_path_data(model, path):
    data = {}
    data['scalar'] = get_path_scalar(model, path)
    data['ABCDs_x'] = get_path_ABCDs(model, path, direction='x')
    data['ABCDs_y'] = get_path_ABCDs(model, path, direction='y')
    data['metaplectic_signs_x'] = get_path_metaplectic_signs(model, path, direction='x')
    data['metaplectic_signs_y'] = get_path_metaplectic_signs(model, path, direction='y')
    data['qx'] = None
    data['qy'] = None
    return data

def eval_data(data):
    out = {}
    out['scalar'] = data['scalar'].eval()
    out['ABCDs_x'] = [map_eval(M) for M in data['ABCDs_x']]
    out['ABCDs_y'] = [map_eval(M) for M in data['ABCDs_y']]
    out['ABCD_x'] = gef.matmul_reduce(out['ABCDs_x'])
    out['ABCD_y'] = gef.matmul_reduce(out['ABCDs_y'])
    out['metaplectic_signs_x'] = [x.eval() for x in data['metaplectic_signs_x']]
    out['metaplectic_signs_y'] = [x.eval() for x in data['metaplectic_signs_y']]
    out['metaplectic_sign_x'] = functools.reduce(lambda x,y: (x+y)%2, out['metaplectic_signs_x'])
    out['metaplectic_sign_y'] = functools.reduce(lambda x,y: (x+y)%2, out['metaplectic_signs_y'])
    return out

def model2n_scalar(model, n_couple=None):
    graph = model.optical_network
    if n_couple is None:
        n_couple = grf.graph2n_couple(graph)
    s_couple = []
    for row in n_couple:
        m_row = []
        for el in row:
            m_el = finesse.symbols.as_symbol(0j)
            for path in el:
                m_el += get_path_scalar(model, path)
            m_row.append(m_el)
        s_couple.append(m_row)
    M = np.array(s_couple, dtype=object)
    return M

def get_all_laser_sources(model):
    lasers = [x for x in model.components if isinstance(x, finesse.components.Laser)]
    return lasers

def get_rhs_paths(model, n_couple=None):
    lasers = get_all_laser_sources(model)
    source_nodes = [x.p1.o.full_name for x in lasers]
    G = model.optical_network
    if n_couple is None:
        n_couple = grf.graph2n_couple(G)
    n_couple_diag = grf.get_n_couple_diag(n_couple)

    rhs_all_paths = grf.get_all_valid_paths(G, source_nodes, n_couple_diag)
    rhs_valid_paths = grf.prune_invalid_paths(rhs_all_paths, n_couple_diag)
    return rhs_valid_paths

def get_rhs_scalars(model, rhs_valid_paths=None, n_couple=None):
    if rhs_valid_paths is None:
        rhs_valid_paths = get_rhs_paths(model, n_couple=n_couple)
    rhs_scalars = model2n_scalar(model, n_couple=rhs_valid_paths)
    return rhs_scalars

def remove_empty_sources(G, inplace=False):
    if not inplace:
        G = grf.copy_graph(G)
    source_nodes = grf.get_source_nodes(G)
    for source_node in source_nodes:
        owner = G.nodes[source_node]['owner']()
        if not isinstance(owner, finesse.components.Laser):
            G.remove_node(source_node)
    return G

def get_reduced_graph(model, keep_nodes=False, residual_nodes=[]):
    G = grf.copy_graph(model.optical_network)
    grf.remove_orphans(G, inplace=True)
    remove_empty_sources(G, inplace=True)
    grf.annotate_paths_to_graph(G, inplace=True)
    grf.graph_reduction(G, inplace=True, keep_nodes=keep_nodes, residual_nodes=residual_nodes)
    return G

def get_reduced_multigraph(model, residual_nodes=[]):
    G = nx.MultiDiGraph(model.optical_network)
    grf.remove_orphans(G, inplace=True)
    remove_empty_sources(G, inplace=True)
    grf.annotate_paths_to_graph(G, inplace=True)
    grf.multigraph_reduction(G, inplace=True, residual_nodes=residual_nodes)
    return G

def get_m_couple(model, reduced_graph):
    source_nodes = grf.get_source_nodes(reduced_graph)
    reduced_loop_graph = grf.remove_nodes_from(reduced_graph, source_nodes)
    G = reduced_loop_graph
    nodes = list(G.nodes)
    N = len(nodes)
    M = np.zeros([N,N], dtype=object)
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            paths = grf.get_paths(G, nj, ni)
            # index order matters here
            M[i, j] = get_paths_scalar(model, paths)
    return M

def get_rhs_vec(model, reduced_graph):
    source_nodes = grf.get_source_nodes(reduced_graph)
    solved_nodes = [n for n in reduced_graph.nodes if n not in source_nodes]
    N = len(solved_nodes)
    rhs_vec = np.zeros(N, dtype=object)
    for i, node in enumerate(solved_nodes):
        for source_node in source_nodes:
            paths = grf.get_paths(reduced_graph, source_node, node)
            source_gain = get_source_scalar(model, source_node)
            rhs_vec[i] = source_gain * get_paths_scalar(model, paths)
    return rhs_vec

def get_edge_operator(component, from_node, to_node, LCT_params, op_type='LCT'):
    model = get_model(component)
    ni,no = get_nodes(model, [from_node, to_node])
    sx = LCT_params.sx
    sy = LCT_params.sy
    Nx = LCT_params.Nx
    Ny = LCT_params.Ny
    Mx = get_edge_ABCD(model, ni, no, direction='x', symbolic=True)
    My = get_edge_ABCD(model, ni, no, direction='y', symbolic=True)
    qxi, qyi = ni.q
    qxo, qyo = no.q
    wxi, wyi = qxi.w, qyi.w
    wxo, wyo = qxo.w, qyo.w
    x1s = np.linspace(-1,1,Nx)*wxi*sx
    x2s = np.linspace(-1,1,Nx)*wxo*sx
    y1s = np.linspace(-1,1,Ny)*wyi*sy
    y2s = np.linspace(-1,1,Ny)*wyo*sy

    edge_scalar = get_edge_scalar(component, from_node, to_node)

    if op_type == 'LCT':
        if isinstance(component, finesse.components.Space):
            Dx = lctf.LCT1D(x1s, x2s, M_abcd=component.abcd)
            Dy = lctf.LCT1D(y1s, y2s, M_abcd=component.abcd)
            def op(v):
                X = np.reshape(v, [Ny, Nx])
                X = Dy@X@Dx.T
                return np.ravel(X) * edge_scalar.eval()
            
        elif isinstance(component, finesse.components.Mirror):
            pi, po = ni.port, no.port
            mirror = ni.component
            if pi is po:
                Rcx = component.Rcx.eval()
                Rcy = component.Rcy.eval()
                # input port == ouput port :: reflection
                if pi is mirror.p1:
                    pass
                elif pi is mirror.p2:
                    # back reflection
                    Rcx = -Rcx
                    Rcy = -Rcy
                
                rx = lctf.CM_kernel(x1s, C=-2/Rcx, diag=True)
                ry = lctf.CM_kernel(y1s, C=-2/Rcy, diag=True)
                r_map = np.outer(ry,rx)
                def op(v):
                    X = np.reshape(v, [Ny, Nx])
                    X = np.fliplr(r_map*X)
                    return np.ravel(X) * edge_scalar.eval()
            else:
                # input port != ouput port :: transmission
                def op(v):
                    return v * edge_scalar.eval()

    elif op_type == 'planewave':
        def op(v):
            if isinstance(component, finesse.components.mirror.Mirror):
                adjust = -1j
            else:
                adjust = 1
            gx = of.accum_gouy_Siegman_n(qxi.q, map_eval(Mx), n=0)
            gy = of.accum_gouy_Siegman_n(qyi.q, map_eval(My), n=0)
            print(gx*gy*adjust)
            return v * gx * gy * adjust * edge_scalar.eval()

    return op

def get_path_operator(model, path, LCT_params, op_type='LCT'):
    model = get_model(model)
    graph = model.optical_network
    # convert path to str to be able to index graph
    path = [get_node_str(n) for n in path]
    path_op = []
    for n1, n2 in zip(path, path[1:]):
        edge = get_edge(graph, n1, n2)
        component = edge['owner']()
        next_op = get_edge_operator(component, n1, n2, LCT_params, op_type)
        path_op.append(next_op)
    return ff.compose(path_op)

def get_paths_operator(model, paths, LCT_params, op_type='LCT'):
    '''
    Takes a list of paths and returns an operator that is the sum of operators for each path.
    '''
    ops = [get_path_operator(model, path, LCT_params, op_type) for path in paths]
    out_op = lambda v: functools.reduce(lambda x,y: x+y(v), ops, np.zeros_like(v))
    return out_op

def get_operator_matrix(model, RG, LCT_params, op_type='LCT'):
    '''
    doc goes here
    '''
    solved_nodes = grf.get_nontrivial_nodes(RG)
    N = len(solved_nodes)
    mat = np.zeros([N,N], dtype=object)
    for i in range(N):
        for j in range(N):
            ni, nj = solved_nodes[i], solved_nodes[j]
            paths = grf.get_paths(RG, nj, ni)
            op = get_paths_operator(model, paths, LCT_params, op_type)
            mat[i, j] = op
    return mat, solved_nodes

def get_paths_matrix(RG):
    solved_nodes = grf.get_nontrivial_nodes(RG)
    N = len(solved_nodes)
    mat = np.zeros([N,N], dtype=object)
    for i in range(N):
        for j in range(N):
            mat[j,i] = grf.get_paths(RG, solved_nodes[i], solved_nodes[j])
    return mat


###############################################################################
#
#                 Another Implementation
#
###############################################################################

def get_edge_data(model, from_node, to_node, symbolic=True, LCT_params=None, sim=None):
    graph = model.optical_network

    ns = nt.Namespace()
    ns.sim = sim
    ni, no = get_nodes(model, [from_node, to_node])
    ns.ni = getattr(ni, 'full_name', ni)
    ns.no = getattr(no, 'full_name', no)
    
    graph_edge = get_edge(graph, from_node, to_node)

    if graph_edge is None:
        ns.scalar = 0
    else:
        component = graph_edge['owner']()
        if getattr(component, 'reflection_map', None) is None:
            smap = None
            composable = True
            separable = True
        else:
            composable = False
            separable = False # TODO: check for separable maps (e.g. tilts)
            smap = True

        ns.scalar = get_edge_scalar(component, ni, no)
        ns.ABCD_x = get_edge_ABCD(model, ni, no, direction='x', symbolic=symbolic)
        ns.ABCD_y = get_edge_ABCD(model, ni, no, direction='y', symbolic=symbolic)
        ns.composable = composable
        ns.separable = separable
        ns.map = smap
    
    return ns.__dict__

def get_source_operator(model, source_node, LCT_params=None, sim=None):
    source_node = get_node(model, source_node)
    source = source_node.component
    
    if isinstance(source, finesse.components.Laser):
        scalar = np.sqrt(source.P.ref) * np.exp(1j*source.phase.ref/180*np.pi)
    else:
        scalar = 0
        
    ns = nt.Namespace()
    ns.scalar = scalar
    ns.ni = source_node.full_name
    ns.no = source_node.full_name
    ns.composable = False
    ns.separable = False
    ns.map = None
    ns.sim = sim
        
    if LCT_params is None:
        scalar_only = True
    
    if not scalar_only:
        # add in the beam sizes
        LCT_params.w1x = no.qx.w
        LCT_params.w1y = no.qy.w
        LCT_params.w2x = no.qx.w
        LCT_params.w2y = no.qy.w

        ns.LCT_params = LCT_params
        
    op = opf.Operator(**ns.__dict__)
    return op

def get_path_operator(model, path, LCT_params=None, sim=None):
    if path == []:
        return opf.Operator(scalar=0, LCT_params=LCT_params, sim=sim)

    edge_list = grf.path_to_edge_list(path)
    op_descs = [get_edge_data(model, *edge, symbolic=False, LCT_params=LCT_params, sim=sim) for edge in edge_list]
    op_list = [opf.Operator(**x) for x in op_descs]
    op_chain = functools.reduce(lambda x,y: y@x, op_list)

    return op_chain

def get_paths_operator(model, paths, LCT_params=None, sim=None):
    if paths == []:
        return opf.Operator(scalar=0, LCT_params=LCT_params, sim=sim)
    
    op_chains = []
    for path in paths:
        op_chain = get_path_operator(model, path, LCT_params, sim=sim)
        op_chains.append(op_chain)
    
    op_sum = opf.OperatorSum(op_chains[0])
    for op_chain in op_chains[1:]:
        op_sum += op_chain
    
    return op_sum

def get_operator_matrix(model, RG, LCT_params=None, sim=None):
    '''
    doc goes here
    '''
    solved_nodes = grf.get_nontrivial_nodes(RG)
    N = len(solved_nodes)
    mat = np.zeros([N, N], dtype=object)
    for i in range(N):
        for j in range(N):
            ni, nj = solved_nodes[i], solved_nodes[j]
            paths = grf.get_paths(RG, nj, ni)
            op = get_paths_operator(model, paths, LCT_params, sim=sim)
            mat[i, j] = op
    return mat, solved_nodes

def scalar_rhs_vec(model, G, nodes):
    '''There are two different uses here.

    '''
    source_nodes = grf.get_source_nodes(G)
    solved_nodes = nodes
    N = len(solved_nodes)
    vec = np.zeros([N], dtype=object)
    for i in range(N):
        ni = solved_nodes[i]
        incoming_paths = grf.get_incoming_paths(G, ni)
        rhs_paths = [path for path in incoming_paths if path[0] in source_nodes]
        if ni in source_nodes:
            rhs_paths.append([ni,ni])
        rhs_ops = []
        for rhs_path in rhs_paths:
            path_op = get_path_operator(model, rhs_path)
            source_op = get_source_operator(model, rhs_path[0])
            rhs_op = path_op @ source_op
            rhs_ops.append(rhs_op)
        vec[i] = np.sum([op.scalar for op in rhs_ops])
    return vec

def scalar_interferometer_matrix(model, G, nodes):
    source_nodes = grf.get_source_nodes(G)
    solved_nodes = nodes
    N = len(solved_nodes)
    mat = np.zeros([N, N], dtype=object)
    for i in range(N):
        ni = solved_nodes[i]
        for j in range(N):
            nj = solved_nodes[j]
            mat_paths = grf.get_paths(G, nj, ni)
            mat_op = get_paths_operator(model, mat_paths)
            mat[i, j] = mat_op.scalar
    return mat

def get_scalar_matrix_vec(model, G, nodes):
    '''
    doc goes here
    '''
    source_nodes = grf.get_source_nodes(G)
    solved_nodes = nodes
    N = len(solved_nodes)
    mat = np.zeros([N, N], dtype=object)
    vec = np.zeros([N], dtype=object)
    for i in range(N):
        ni = solved_nodes[i]
        incoming_paths = grf.get_incoming_paths(G, ni)
        rhs_paths = [path for path in incoming_paths if path[0] in source_nodes]
        rhs_ops = []
        for rhs_path in rhs_paths:
            path_op = get_path_operator(model, rhs_path)
            source_op = get_source_operator(model, rhs_path[0])
            rhs_op = path_op @ source_op
            rhs_ops.append(rhs_op)
        vec[i] = np.sum([op.scalar for op in rhs_ops])
        for j in range(N):
            nj = solved_nodes[j]
            mat_paths = grf.get_paths(G, nj, ni)
            mat_op = get_paths_operator(model, mat_paths)
            mat[i, j] = mat_op.scalar
    return mat, vec


###############################################################################
#
#                 Simulation Object
#
###############################################################################


class LCTSimulation(object):
    def __init__(self, model, residual_nodes=None, LCT_params=None):
        self._model = model
        self._base_LCT_params = LCT_params

        if LCT_params is None:
            warnings.warn('can only planewave_solve if LCT_params=None')
        
        if residual_nodes is None:
            residual_nodes = []
        self.residual_nodes = residual_nodes
            
        g1 = grf.copy_graph(model.optical_network)
        g2 = grf.remove_orphans(g1)
        g3 = remove_empty_sources(g2)
        g4 = grf.remove_sinks(g3)

        self.G = grf.annotate_paths_to_graph(g2)

        RG = get_reduced_multigraph(self._model, residual_nodes=self.residual_nodes)
        self.RG = RG
        self.nontrivial_nodes = grf.get_nontrivial_nodes(self.RG)
        self.source_nodes = grf.get_source_nodes(self.RG)
        
        self.node_dict = {}
        for node in self._model.optical_nodes:
            self.node_dict[node] = nt.Namespace()
            self.node_dict[node.full_name] = self.node_dict[node]
            if not LCT_params is None:
                self.node_dict[node].LCT_params = lctf.LCTNodeParams(**LCT_params.__dict__)
                self.node_dict[node].LCT_params.wx = node.qx.w
                self.node_dict[node].LCT_params.wy = node.qy.w

    def planewave_solve_G(self):
        """Mostly used for debugs
        """
        solved_nodes = list(self.G.nodes)
        M, rhs, _ = get_scalar_matrix_vec(self._model, self.G, solved_nodes)
        I_M = np.eye(len(M)) - map_eval(M)
        v = map_eval(rhs)
        scalar_soln = np.linalg.solve(I_M, v)
        return scalar_soln

    def planewave_solve_RG(self):
        """Mostly used for debugs
        """
        solved_nodes = grf.get_nontrivial_nodes(self.RG)
        M, rhs, _ = get_scalar_matrix_vec(self._model, self.RG, solved_nodes)
        I_M = np.eye(len(M)) - map_eval(M)
        v = map_eval(rhs)
        scalar_soln = np.linalg.solve(I_M, v)
        return scalar_soln
    
    def prepare_LCT_solve(self):
        M_mat, rhs_nodes = get_operator_matrix(self._model, self.RG, sim=self)
        self.rhs_nodes = rhs_nodes
        self.M_mat = M_mat
        
        Nn = np.sum([self.node_dict[node].LCT_params.N_points for node in rhs_nodes])
        v0 = np.zeros(Nn)
        I_MA = opf.ifo_matvec(M_mat, v0)
        I_MA_linop = scipy.sparse.linalg.LinearOperator(shape=[Nn, Nn], matvec=I_MA)
        self.I_MA_linop = I_MA_linop
        return True
    
    def finesse_transverse_field(self, node, soln=None, LCT_params=None):
        model = self._model
        node = get_node(self._model, node)
        projector = HGModes(node.q, self._model.homs, zero_tem00_gouy=False, reverse_gouy=True)

        lctpx = self.node_dict[node].LCT_params
        lctpy = self.node_dict[node].LCT_params
        if not LCT_params is None:
            ddx = {**lctpx.__dict__, **gef.filter_dict_by_value(LCT_params.__dict__, None)}
            ddy = {**lctpy.__dict__, **gef.filter_dict_by_value(LCT_params.__dict__, None)}
            lctpx = lctf.LCTNodeParams(**ddx)
            lctpy = lctf.LCTNodeParams(**ddy)

        xs = lctpx.xs
        ys = lctpy.ys

        fd_name = f"E_{node.full_name.replace('.','_')}"
        
        field = np.sum(projector.compute_2d_modes(xs,ys) * soln[fd_name], axis=0)
        return field
    
    def get_source_field(self, source_node):
        source_node = get_node(self._model, source_node)
        component = source_node.component

        field = np.zeros([ys.size, xs.size], dtype=complex)
        if isinstance(component, finesse.components.Laser):
            hom_vec = component.get_output_field(self._model.homs)

            xs = self.node_dict[source_node].LCT_params.xs
            ys = self.node_dict[source_node].LCT_params.ys

            qx = source_node.qx.q
            qy = source_node.qy.q

            for n_m, a in zip(self._model.homs, hom_vec):
                n, m = n_m
                u = a * of.u_nm_q(xs, ys, qx, qy, n, m, lam=self._model.lambda0, include_gouy=False)
                field += u
        return field
    
    def get_rhs_operators(self):
        rhs_nodes = self.rhs_nodes
        source_nodes = grf.get_source_nodes(self.RG)

        rhs_mat = np.zeros([len(rhs_nodes), len(source_nodes)], dtype=object)

        for i, rhs_node in enumerate(rhs_nodes):
            for j, source_node in enumerate(source_nodes):
                rhs_paths = grf.get_paths(self.RG, source_node, rhs_node)
                rhs_op = get_paths_operator(self._model, rhs_paths, sim=self)
                rhs_mat[i, j] = rhs_op

        return rhs_mat
    
    def build_source_indexing_interval(self):
        source_nodes = self.source_nodes
        source_indices = {}
        slc = slice(0,0)
        for source_node in source_nodes:
            N = self.node_dict[source_node].LCT_params.N_points
            next_slc = slice(slc.stop, slc.stop+N)
            source_indices[source_node] = next_slc
            slc = next_slc
        return source_indices
    
    def get_source_vec(self):
        source_nodes = self.source_nodes
        rhs_nodes = self.rhs_nodes
        Nn = np.sum([self.node_dict[node].LCT_params.N_points for node in source_nodes])
        source_vec = np.zeros(Nn, dtype=complex)
        for source_node in source_nodes:
            field = self.get_source_field(source_node)
            slc = self.source_indices[source_node]
            source_vec[slc] = field.ravel()
            
        return source_vec
              
    def prepare_LCT_rhs(self):
        self.rhs_mat = self.get_rhs_operators()
        self.source_indices = self.build_source_indexing_interval()
        self.source_vec = self.get_source_vec()
        self.rhs_fun = opf.opmat_matvec(self.rhs_mat, self.source_vec)
        return True
    
    def solve_LCT(self, rhs_vec=None):
        if rhs_vec is None:
            rhs_vec = self.rhs_fun(self.source_vec)
        soln_vec, return_code = scipy.sparse.linalg.bicgstab(self.I_MA_linop, rhs_vec)
        solns = []
        for i, rhs_node in enumerate(self.rhs_nodes):
            Nx = self.node_dict[rhs_node].LCT_params.Nx
            Ny = self.node_dict[rhs_node].LCT_params.Ny
            soln_field = np.reshape(opf.vec_slice(soln_vec, Nx*Ny, i), (Ny,Nx))
            solns.append(soln_field)
        return solns