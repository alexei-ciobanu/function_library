import random
import functools

import networkx as nx
import numpy as np

from .graph import copy_graph, get_multiedge, get_loop_nodes
from .fvs import mdfvs_sinkhorn
from .manipulate import remove_multiedge

from general_funs import flip_dict

def absorb_node_weighted(G, node, node_weight_key='weight', edge_weight_key='weight', inplace=True, debug=False):
    '''Only performs node-preserving product graph reduction rule
    '''
    if not inplace:
        G = copy_graph(G)
    
    pred = list(G.predecessors(node))
    succ = list(G.successors(node))
    
    # can only reduce if there is no self-loop
    if node not in pred:
        # apply reduction
        for pn in pred:
            for sn in succ:   
                edges_10 = get_multiedge(G, pn, node)
                edges_21 = get_multiedge(G, node, sn)
                for e10 in edges_10:
                    for e21 in edges_21:
                        w10 = G.edges[e10][edge_weight_key]
                        w21 = G.edges[e21][edge_weight_key]
                        edge_dict = {edge_weight_key: w21@w10}
                        G.add_edge(pn, sn, **edge_dict)
                
        # turn absorbed node into a sink
        for sn in succ:
            # propagate information of absorbed node
            # to successors
            v1 = G.nodes[node][node_weight_key]
            v2 = G.nodes[sn][node_weight_key]
            edges_21 = get_multiedge(G, node, sn)
            ws_21 = [G.edges[e][edge_weight_key] for e in edges_21]
            op_21 = functools.reduce(lambda x,y: x + y, ws_21)
            # print(v2.shape, v1.shape)
            v3 = v2 + op_21@v1
            # print(v3.shape)
            G.nodes[sn][node_weight_key] = v3
            
            remove_multiedge(G, node, sn, inplace=True)
        
        if debug:
            # assert node is now a sink
            assert(G.out_degree[node] == 0)
    return G

def sum_parallel_edges(G, inplace=True, debug=False):
    if not inplace:
        copy_graph(G)
    if not isinstance(G, nx.MultiDiGraph):
        raise TypeError(f'addition reduction rule only defined for nx.MultiDiGraph, not {type(G)}')
    multiedge_list = list({e[0:2] for e in G.edges})
    for me in multiedge_list:
        elist = get_multiedge(G, *me)
        # print(elist, len(elist))
        if len(elist) > 1:
            # print(elist)
            ws = [G.edges[e]['weight'] for e in elist]
            op = functools.reduce(lambda x,y: x + y, ws)
            remove_multiedge(G, *me, inplace=True)
            G.add_edge(*me, weight=op)
            
    for me in multiedge_list:
        elist = get_multiedge(G, *me)
        assert(len(elist) == 1)
    return G

def multigraph_reduction(G, nodes=None, residual_nodes=None, node_weight_key='weight', edge_weight_key='weight', inplace=False, debug=False):
    '''Performs weighted MultiDiGraph reduction using a random node order and sinkhorn mdfvs
    '''
    if isinstance(G, nx.MultiDiGraph):
        if not inplace:
            G = copy_graph(G)
    elif isinstance(G, nx.DiGraph):
        G = nx.MultiDiGraph(G)
    else:
        raise TypeError(f'input graph can only be DiGraph or MultiDiGraph, not {type(G)}')      
    if nodes is None:
        nodes = list(G.nodes)
        random.shuffle(nodes)
    if residual_nodes is None:
        residual_nodes = mdfvs_sinkhorn(G)
    if debug: print(nodes)
    for node in nodes:
        if node not in residual_nodes:
            absorb_node_weighted(G, node, node_weight_key=node_weight_key, edge_weight_key=edge_weight_key)

    sum_parallel_edges(G)
    # G = nx.DiGraph(G) # does this copy node and edge weights?
    return G

def graph_node_ind_mapping(G, copy=True, sort=True):
    if sort:
        node_list = sorted(G.nodes)
    else:
        node_list = G.nodes
    mapping = {n:i for i,n in enumerate(node_list)}
    G = nx.relabel_nodes(G, mapping, copy=copy)
    return G

def graph_to_dense_matrix(G, node_weights=None, edge_weights=None, copy=False, dtype=complex):
    '''assumes nodes are sorted integer order
    '''
    if node_weights is None:
        node_weights = {n: G.nodes[n]['weight'] for n in G.nodes}
    if edge_weights is None:
        edge_weights = {e: G.edges[e]['weight'] for e in G.edges}

    # print(G.edges)
    # print(list(edge_weights.keys()))

    # Declaration
    ##################
    N = 0
    sz = []
    for n in node_weights:
        N += node_weights[n].size
        sz.append(N)
    sz.pop(-1)

    # Construction
    ##################
    A = np.zeros([N,N], dtype=dtype)

    edge_views = {}
    sa = np.split(A, sz, axis=1)
    for i,a in enumerate(sa):
        sb = np.split(a, sz, axis=0)
        for j,b in enumerate(sb):
            edge_views[i,j] = b

    # Assignment
    ##################
    for e in edge_weights:
        w = edge_weights[e]
        ## convert sparse to dense
        # if isinstance(w, scipy.sparse.base.spmatrix):
        #     w = w.todense()
        edge_views[e[0],e[1]][:] -= w

    # I = np.eye(N, dtype=dtype)
    idx, idy = np.diag_indices(N)
    A[idx,idy] += np.ones(N, dtype=dtype)

    return A

def graph_to_dense_rhs(G, node_weights=None, dtype=complex, copy=False):
    '''assumes nodes are sorted integer order
    '''
    if node_weights is None:
        node_weights = {n: G.nodes[n]['weight'] for n in G.nodes}

    # Declaration
    ##################
    N = 0
    sz = []
    for n in node_weights:
        N += node_weights[n].size
        sz.append(N)
    sz.pop(-1)

    # Construction
    ##################
    v = np.zeros(N, dtype=dtype)

    rhs_views = {}
    idn = np.split(v, sz, axis=0)
    for i,slc in enumerate(idn):
        rhs_views[i] = slc

    # Assignment
    ##################
    for n in node_weights:
        rhs_views[n][:] = np.ravel(node_weights[n])

    return v

def solve_weighted_graph(G, method='dense', copy=True):
    if copy:
        G = graph_node_ind_mapping(G)
    if method == 'dense':
        B = graph_to_dense_matrix(G)
        v = graph_to_dense_rhs(G)
        s = np.linalg.solve(B,v)
    else:
        raise ValueError(f'unknown weighted graph solving method: {method}')
    return s

def graph_factor(G, residual_nodes=None, inplace=True):
    '''Apply graph reduction rules and propagate the node and edge weights appropriately
    '''
    Gr = multigraph_reduction(G, residual_nodes=residual_nodes, inplace=True)
    return Gr

def graph_solve(Gr, copy=False):
    '''Solves the self loops and propagates the solution to the rest of
    the nodes. Assumes the input graph was factored.
    '''
    def atleast_2d(v):
        '''numpy one turns 1D into row vecs instead of column vec
        '''
        if len(np.shape(v)) < 2:
            v = np.atleast_2d(v).T
        return v
    
    if copy:
        Gr = copy_graph(Gr)
    nl = get_loop_nodes(Gr)
    Gl = Gr.subgraph(nl) # subgraphs are frozen
    # subgraph nodes need to be relabeled to be ordered integers to work with solver
    Gl = nx.relabel.convert_node_labels_to_integers(Gl, label_attribute='original_node')

    sl = solve_weighted_graph(Gl, method='dense', copy=False)
    sl = atleast_2d(sl) # makes it a proper column vector
    N = 0
    sz = []
    for n in Gl.nodes:
        N += Gl.nodes[n]['weight'].size
        sz.append(N)
    sz.pop(-1)
    sl_views = np.split(sl, sz)

    # replace the rhs field in the loop nodes with the solved field
    for i,nl in enumerate(Gl.nodes):
        n = Gl.nodes[nl]['original_node']
        Gr.nodes[n]['weight'] = sl_views[i]

    # remove the loop graph edges from the reduced graph
    # since their contribution has been collapsed into the loop nodes
    for el in Gl.edges:
        u = Gl.nodes[el[0]]['original_node']
        v = Gl.nodes[el[1]]['original_node']
        remove_multiedge(Gr, u, v, inplace=True)
        
    # propagate the solved loop node fields to the rest of the graph
    for nl in Gl.nodes:
        n = Gl.nodes[nl]['original_node']
        absorb_node_weighted(Gr, n)
    
    return Gr

def graph_factor_solve(G, residual_nodes=None, copy=False):
    '''Convenience method for factoring followed by solving.
    '''
    if copy:
        G = copy_graph(G)
    Gr = graph_factor(G, residual_nodes=residual_nodes, inplace=True)
    Grs = graph_solve(Gr, copy=False)
    return Grs