import numpy as np
import scipy.sparse
import networkx as nx

import finesse.cmatrix

def model_build(model):
    '''this is a placeholder for setting up the environment to access edge and node
    weights from a finesse model
    '''
    with model.built() as sim:
        sim.carrier.clear_rhs()
        sim.carrier.fill_rhs()

def graph_node_ind_mapping(G, inplace=False, sort=True):
    if sort:
        node_list = sorted(G.nodes)
    else:
        node_list = G.nodes
    mapping = {n:i for i,n in enumerate(node_list)}
    copy = not inplace
    G = nx.relabel_nodes(G, mapping, copy=copy)
    return G

def model_to_graph(model, return_klumatrix=False, graph_type=None):
    '''Makes a MSFG out of a finesse3 model. This graph can be used for graph reductions.
    '''
    if graph_type is None:
        graph_type = nx.MultiDiGraph
    if issubclass(graph_type, nx.MultiDiGraph):
        edge_selector = lambda G,u,v: G.edges[u,v,0]
    elif issubclass(graph_type, nx.DiGraph):
        edge_selector = lambda G,u,v: G.edges[u,v]
    else:
        raise TypeError(f'graph_type can only be DiGraph or MultiDiGraph, not {graph_type}')
    with model.built() as sim:
        sim.carrier.clear_rhs()
        sim.carrier.fill_rhs()
        G1 = nx.subgraph(model.optical_network, sim.carrier.nodes)
        if not isinstance(G1, graph_type):
            G1 = graph_type(G1)
        node_ind_mapping = {}
        for n in G1.nodes:
            node_ind_mapping[n] = sim.carrier.findex(n, 0)
        G2 = nx.relabel_nodes(G1, node_ind_mapping)

        for key, mat in sim.carrier._submatrices.items():
            u,v = mat.from_idx, mat.to_idx
            if isinstance(mat, finesse.cmatrix.SubCCSMatrixViewDiagonal):
                idx = np.diag_indices(len(mat.view))
                b = scipy.sparse.coo_matrix((mat.view, idx), dtype=complex)
                edge_selector(G2, u, v)['weight'] = -b
            elif isinstance(mat, (finesse.cmatrix.SubCCSMatrixView, np.ndarray)):
                edge_selector(G2, u, v)['weight'] = -mat.view
            
        for n, mat in sim.carrier._diagonals.items():
            G2.nodes[n]['weight'] = np.array(mat.from_rhs_view).T

    if return_klumatrix:
        out = G2, sim.carrier.M()
    else:
        out = G2
    return out


def graph_to_klumatrix(G, node_weights=None, edge_weights=None):
    '''Converts a directed signal flow graph into a sparse matrix.
    The edge and node weights can be passed as a key-value dictionary, otherwise
    the weights are assumed to be on the graph object.
    
    Scalar and vector node weights are supported. Each node can have a different
    weight size.
    
    Scalar and matrix edge weights are supported. Edge weight matrix shapes 
    are assumed to be consistent with their corresponding node weight sizes.
    '''
    
    if node_weights is None:
        node_weights = {n: G.nodes[n]['weight'] for n in sorted(G.nodes)}
    if edge_weights is None:
        edge_weights = {e: G.edges[e]['weight'] for e in G.edges}
    
    M = finesse.cmatrix.KLUMatrix("M")

    # Declaration
    ##################

    node_views = {}
    for i, node_weight in node_weights.items():
        # every node must be listed in node_weights, even nodes that has zero weight
        sz = np.size(node_weight)
        if (i, i) in G.edges:
            # allocate the entire block if there is a self-loop
            node_views[i] = M.declare_equations(sz, i, str(i), is_diagonal=False)
        else:
            # otherwise just allocate the diagonal and save some memory
            node_views[i] = M.declare_equations(sz, i, str(i), is_diagonal=True)

    edge_views = {}
    for i_j, v in edge_weights.items():
        i,j = i_j
        if i == j:
            # self loops have to reuse the diagonal view otherwise that 
            # memory gets double allocated
            edge_views[i,j] = node_views[i]
        else:
            # allocate off-diagonals as normal
            edge_views[i,j] = M.declare_submatrix_view(i, j, str(i_j), False)

    # Construction
    ##################

    M.construct()

    # Assignment
    ##################

    for i, n in node_weights.items():
        node_views[i].from_rhs_view[:] = n

    for i_j, v in edge_weights.items():
        i,j = i_j
        if i == j:
            # add the diagonal identity elements and subtract the self loop contribution
            # identity elements were not included because we did M.add_diagonal_elements(is_diagonal=False)
            # assigning to views inserts a minus sign so we have to negate here
            edge_views[i,j][:] = -(np.eye(len(np.atleast_1d(v))) - v)
        else:
            # automatically inserts minus sign (must be some hook in the view assignment)
            edge_views[i,j][:] = v
            
    return M

def graph_to_dense_matrix(G, node_weights=None, edge_weights=None, dtype=complex):

    G = graph_node_ind_mapping(G)

    if node_weights is None:
        node_weights = {n: G.nodes[n]['weight'] for n in sorted(G.nodes)}
    if edge_weights is None:
        edge_weights = {e: G.edges[e]['weight'] for e in G.edges}

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
        # if isinstance(w, scipy.sparse.base.spmatrix):
        #     w = w.todense()
        edge_views[e[0],e[1]][:] -= w

    # I = np.eye(N, dtype=dtype)
    idx, idy = np.diag_indices(N)
    A[idx,idy] += np.ones(N, dtype=dtype)
    return A

def graph_to_sparse_matrix(G, node_weights=None, edge_weights=None, format='csc', dtype=complex):

    G = graph_node_ind_mapping(G)

    if node_weights is None:
        node_weights = {n: G.nodes[n]['weight'] for n in sorted(G.nodes)}
    if edge_weights is None:
        edge_weights = {e: G.edges[e]['weight'] for e in G.edges}

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
    A = scipy.sparse.lil_matrix((N,N), dtype=complex)
    Mh,Mv = np.meshgrid(np.arange(N),np.arange(N))

    edge_indices = {}
    ssrow = np.split(Mh, sz, axis=1)
    sscol = np.split(Mv, sz, axis=1)
    for i,srow_scol in enumerate(zip(ssrow,sscol)):
        srow, scol = srow_scol
        row = np.split(srow, sz, axis=0)
        col = np.split(scol, sz, axis=0)
        for j,row_col in enumerate(zip(row,col)):
            row, col = row_col
            edge_indices[i,j] = col, row

    diag_indices = np.diag_indices(N)
    A[diag_indices] += np.ones(N, dtype=complex)

    # Assignment
    ##################
    for e in edge_weights:
        A[edge_indices[e[0],e[1]]] -= edge_weights[e]
        
    out_method = 'to'+format.lower()
    B = getattr(A, out_method)()
    return B

def graph_to_sparse_rhs(G, node_weights=None, format='csc', dtype=complex):
    G = graph_node_ind_mapping(G)

    if node_weights is None:
        node_weights = {n: G.nodes[n]['weight'] for n in sorted(G.nodes)}

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
    v = scipy.sparse.lil_matrix((N,1), dtype=dtype)
    idx = np.arange(N)

    rhs_indices = {}
    idn = np.split(idx, sz, axis=0)
    for i,slc in enumerate(idn):
        rhs_indices[i] = slc

    # Assignment
    ##################
    for n in node_weights:
        v[rhs_indices[n]] = node_weights[n]

    out_method = 'to'+format.lower()
    v = getattr(v, out_method)()

    return v

def graph_to_dense_rhs(G, node_weights=None, dtype=complex):
    G = graph_node_ind_mapping(G)

    if node_weights is None:
        node_weights = {n: G.nodes[n]['weight'] for n in sorted(G.nodes)}

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

def solve_weighted_graph(G, method='dense'):
    if method == 'dense':
        B = graph_to_dense_matrix(G)
        v = graph_to_dense_rhs(G)
        s = np.linalg.solve(B,v)
    elif method == 'klu':
        M = graph_to_klumatrix(G)
        M.factor()
        s = M.solve()
    elif method == 'sparse':
        C = graph_to_sparse_matrix(G)
        v = graph_to_sparse_rhs(G)
        s = scipy.sparse.linalg.spsolve(C,v)
    # print(B.shape, v.shape, s.shape)
    return s