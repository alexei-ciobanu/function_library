import collections
import random
import functools

import numpy as np

from .graph import copy_graph, get_loop_nodes, strongly_connected_components, adjacency_matrix
from .manipulate import levy_low_reduction

def sinkhorn_transform(A, N_iter=None, return_transform=False, copy=True):
    '''Iterative algorithm for sinkhorn scaling transformation.
    Finds two diagonal matrices D1, D2 such that B = D1@A@D2 is doubly stochastic.
    Doubly stochastic means that each row and column sums to 1.
    
    This B sinkhorn matrix has applications in graph theory, optimal transform, 
    and machine learning.
    
    The algorithm works by repeatedly normalizing the rows then the columns of the
    matrix. Convergence is monotonic and N*log(N) iterations is typically deemed
    sufficient.
    
    https://math.nist.gov/mcsd/Seminars/2014/2014-06-10-Shook-presentation.pdf
    '''
    if copy:
        A = A.copy()
    n = A.shape[0]
    if N_iter is None:
        N_iter = int(np.round(4*n*np.log(n)))
    if return_transform:
        D1 = np.ones(n)
        D2 = np.ones(n)
    At = A.T
    for i in range(N_iter):
        r = np.sum(A, 0)
        A /= r
        c = np.sum(A, 1)
        At /= c
        if return_transform:
            D1 *= c
            D2 *= r
        # print(np.sum(np.abs(r - np.ones_like(r))), np.sum(np.abs(c - np.ones_like(c))))
    if return_transform:
        return np.diag(1/D1), np.diag(1/D2)
    else:
        return A
    
def mdfvs_max_deg(G, fvs=None, copy=True):
    '''A reasonable hueristic but not as good as some of the 
    more sophisticated ones.
    '''
    if fvs is None:
        fvs = []
    if copy:
        G = copy_graph(G, copy_attributes=False)
        
    # levy_low can produce self-loops
    levy_low_reduction(G)
    self_loops = get_loop_nodes(G)
    while self_loops:
        # self-loops are trivially part of mdfvs
        fvs.extend(self_loops)
        G.remove_nodes_from(self_loops)
        levy_low_reduction(G) # can create more self-loops so we remove those too
        self_loops = get_loop_nodes(G)
    
    # any nodes that are part of a strongly connected component are 
    # in a cycle
    L_sets = strongly_connected_components(G)
    L = functools.reduce(lambda x,y: x|y, L_sets, set())
    # print(L, fvs, loops)
    if len(L) == 0:
        return fvs

    # pick the node with the largest total degree as part of fvs
    totals = collections.defaultdict(list)
    for n in L:
        # total degree is the minimum of the in degree and out degree
        total_deg = min(G.in_degree[n], G.out_degree[n])
        totals[total_deg].append(n)
    k = max(totals.keys())
    v = random.choice(totals[k])
    
    # print(v,k)
    fvs.append(v)
    G.remove_node(v)
    return mdfvs_max_deg(G, fvs=fvs, copy=False)

def mdfvs_random(G, fvs=None, copy=True):
    '''Pretty much the dumbest mdfvs, definitly not optimal.
    Will just randomly pick nodes to try to kill all the cycles.
    '''
    if fvs is None:
        fvs = []
    if copy:
        G = copy_graph(G, copy_attributes=False)
    levy_low_reduction(G)
    
    # levy_low can produce self-loops
    levy_low_reduction(G)
    self_loops = get_loop_nodes(G)
    while self_loops:
        # self-loops are trivially part of mdfvs
        fvs.extend(self_loops)
        G.remove_nodes_from(self_loops)
        levy_low_reduction(G) # can create more self-loops so we remove those too
        self_loops = get_loop_nodes(G)
    
    # any nodes that are part of a strongly connected component are 
    # in a cycle
    L_sets = strongly_connected_components(G)
    L = functools.reduce(lambda x,y: x|y, L_sets, set())
    if len(L) == 0:
        return fvs
    
    # pick a strongly connected node at random
    v = random.choice(tuple(L))
    fvs.append(v)
    G.remove_node(v)
    return mdfvs_random(G, fvs=fvs, copy=False)
    
def mdfvs_bogo(G, Niter=100, method='random'):
    '''A convenience method for running the random
    mdfvs methods in a loop and returning the best result found
    '''
    mdfvs = []
    mdfvs_len = np.inf
    for i in range(Niter):
        if method == 'random':
            fvs = mdfvs_random(G)
        elif method == 'max_deg':
            fvs = mdfvs_max_deg(G)
        fvs_len = len(fvs)
        if fvs_len < mdfvs_len:
            mdfvs_len = fvs_len
            mdfvs = fvs
    return mdfvs

def sinkhorn_node_order(G, Nsinkhorn=None, debug=False):
    A = adjacency_matrix(G)
    A = A.astype(float)
    A = A + np.eye(len(A))
    H = sinkhorn_transform(A, N_iter=Nsinkhorn)
    H_diag = np.diag(H)
    idx = np.argmin(H_diag)
    full_sort = np.argsort(H_diag)[::-1]
    sorted_nodes = np.array(list(G.nodes))[full_sort].tolist()
    if debug:
        sorted_H = H_diag[full_sort]
        print(np.stack([sorted_nodes, sorted_H]).T)
    return sorted_nodes
    
def mdfvs_sinkhorn(G, Nsinkhorn=None, fvs=None, copy=True, debug=False):
    '''This is nuts. Really fast and seems to always find the smallest fvs of any method tried.
    Have to compare it with brute-force at some point.
    
    Source: https://math.nist.gov/mcsd/Seminars/2014/2014-06-10-Shook-presentation.pdf
    '''
    if fvs is None:
        fvs = []
    if copy:
        G = copy_graph(G, copy_attributes=False)
    
    # levy_low can produce self-loops
    levy_low_reduction(G)
    self_loops = get_loop_nodes(G)
    while self_loops:
        # self-loops are trivially part of mdfvs
        fvs.extend(self_loops)
        G.remove_nodes_from(self_loops)
        levy_low_reduction(G) # can create more self-loops so we remove those too
        self_loops = get_loop_nodes(G)
    
    # any nodes that are part of a strongly connected component are 
    # in a cycle
    L_sets = strongly_connected_components(G)
    L = functools.reduce(lambda x,y: x|y, L_sets, set())
    if len(L) == 0:
        return fvs
    
    A = adjacency_matrix(G)
    A = A.astype(float)
    A = A + np.eye(len(A))
    H = sinkhorn_transform(A, N_iter=Nsinkhorn)
    H_diag = np.diag(H)
    idx = np.argmin(H_diag)
    v = list(G.nodes)[idx]
    if debug:
        full_sort = np.argsort(H_diag)
        sorted_nodes = np.array(list(G.nodes))[full_sort]
        sorted_H = H_diag[full_sort]
        print(np.stack([sorted_nodes, sorted_H]).T)
    
    fvs.append(v)
    G.remove_node(v)
    return mdfvs_sinkhorn(G, fvs=fvs, copy=False)