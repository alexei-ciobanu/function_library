import collections
import random

import networkx as nx

import general_funs as gef
# from .manipulate import remove_orphans, remove_sources, remove_sinks

def strongly_connected_components(G):
    '''Excludes strongly connected components that only have a single node
    '''
    sccs = [scc for scc in nx.strongly_connected_components(G) if len(scc) > 1]
    return sccs

def has_cycle(G, return_cycle=False):
    '''Every node that is part of a strongly connected component
    is part of a cycle.

    Computing strongly connected components is very efficient and so this
    is a very efficient way of detecting if a graph has at least 1 cycle.

    In networkx this strongly connected component needs to be a set
    with more than one node.
    '''
    out = False
    for scc in nx.strongly_connected_components(G):
        if len(scc) > 1:
            if return_cycle:
                out = nx.find_cycle(G, scc)
            else:
                out = True
            break
    return out

def is_connected(G):
    G = nx.Graph(G)
    return nx.is_connected(G)

def graph_to_digraph(G):
    '''Turns an undirected graph into a directed one by taking a random
    direction in the undirected edges
    '''
    dG = nx.DiGraph()
    for e in G.edges:
        flip = gef.random_bool()
        if flip:
            e = e[::-1]
        dG.add_edge(*e)
    return dG

def degree(G, flip=True, sort=True):
    out = dict(G.degree)
    if flip:
        out = gef.flip_dict(out)
        if sort:
            d = collections.defaultdict(list)
            for k in sorted(out.keys())[::-1]:
                d[k] = out[k]
            out = d
    return out

def in_degree(G, flip=True, sort=True):
    out = dict(G.in_degree)
    if flip:
        out = gef.flip_dict(out)
        if sort:
            d = collections.defaultdict(list)
            for k in sorted(out.keys())[::-1]:
                d[k] = out[k]
            out = d
    return out

def out_degree(G, flip=True, sort=True):
    out = dict(G.out_degree)
    if flip:
        out = gef.flip_dict(out)
        if sort:
            d = collections.defaultdict(list)
            for k in sorted(out.keys())[::-1]:
                d[k] = out[k]
            out = d
    return out

def edge_multiplicity(G, flip=True, sort=True):
    out = collections.defaultdict(int)
    for e in G.edges:
        u,v,k = e
        out[u,v] += 1
    if flip:
        out = gef.flip_dict(out)
        if sort:
            d = collections.defaultdict(list)
            for k in sorted(out.keys())[::-1]:
                d[k] = out[k]
            out = d
    return out

def copy_graph(G, copy_attributes=True):
    '''
    A trick I often see in networkx's codebase to copy a graph.
    type(G) returns the class of G (e.g. nx.DiGraph) which can accept a graph to
    make a copy of.
    Useful for writing 'pure' functions on graphs.
    '''
    if copy_attributes:
        G2 = type(G)(G)
    else:
        G2 = type(G)(G.edges)
    return G2

def get_sink_nodes(G):
    '''
    A sink node is a node with only incoming edges.
    '''
    out_degree_dict = gef.flip_dict(dict(G.out_degree))
    degree_dict = gef.flip_dict(dict(G.degree))
    try:
        sink_nodes = set(out_degree_dict[0]) - set(degree_dict[0])
    except KeyError:
        sink_nodes = []
    return list(sink_nodes)

def get_source_nodes(G):
    '''
    A source node is a node with only outgoing edges.
    '''
    in_degree_dict = gef.flip_dict(dict(G.in_degree))
    degree_dict = gef.flip_dict(dict(G.degree))
    try:
        source_nodes = set(in_degree_dict[0]) - set(degree_dict[0])
    except KeyError:
        source_nodes = []
    return list(source_nodes)

def get_orphan_nodes(G):
    '''
    An orphan node is a node with no edges.
    '''
    degree_dict = gef.flip_dict(dict(G.degree))
    try:
        orphan_nodes = list(degree_dict[0])
    except KeyError:
        orphan_nodes = []
    return orphan_nodes

def get_loop_nodes(G):
    '''
    An loop node has itself as its own successor.
    '''
    loop_nodes = []
    for n in G.nodes:
        if n in G.succ[n]:
            loop_nodes.append(n)
    return loop_nodes

def get_path_nodes(G):
    '''A path node is a node that has one incoming edge and one outgoing edge
    and is not a loop node.
    '''
    in1 = set(in_degree(G)[1])
    out1 = set(out_degree(G)[1])
    path_nodes = in1 & out1
    n_lst = list(path_nodes)
    for n in n_lst:
        # check if any of the path nodes are a loop node
        if n in G.succ[n]:
            path_nodes.remove(n)
    return list(path_nodes)

def get_multiedge(G, u, v):
    edges = []
    if isinstance(G, nx.MultiGraph):
        for multi_key in G[u][v]:
            edges.append((u, v, multi_key))
    else:
        edges.append((u,v))
    return edges

def adjacency_matrix(G, sparse=False, weight='None'):
    '''
    networkx returns adjacency matrix in sparse format by default
    '''
    M = nx.adjacency_matrix(G, weight=weight)
    if not sparse:
        M = M.asformat('array')
    return M

def minimum_feedback_vertex_set(G, fvs=None, copy=True):
    '''This may not actually be a minimum I just use a hueristic that 
    seems to work.
    
    The hueristic is to compute all simple cycles and to select the node
    that appears the most in all of the cycles. If there is more than
    one node that appears the most then just pick a random one out of them.
    
    Once the node is deleted recursively run the algorithm again until there
    are no more cycles.

    This seems to produce the optimally small feedback vertex set for directed
    graphs with degree (in + out) <= 3. For degree four or more random regular 
    directed graphs this algorithm will not produce a FVS of consistent size.
    Run it again and it can randomly change.
    '''
    if copy:
        G = copy_graph(G)
    if fvs is None:
        fvs = []
    
    cycles = list(nx.simple_cycles(G))
    if cycles == []:
        return fvs
    
    d = collections.defaultdict(int)
    for l in cycles:
        for n in l:
            d[n] += 1
            
    df = gef.flip_dict(d)
    k = max(df.keys())
    fv = random.choice(df[k])
    
    fvs.append(fv)
    G.remove_node(fv)
    return minimum_feedback_vertex_set(G, fvs, copy=False)

# def get_non_sink_nodes(G):
#     G2 = remove_sinks(G, inplace=False)
#     non_sink_nodes = list(G2.nodes)
#     return non_sink_nodes

# def get_non_source_nodes(G):
#     G2 = remove_sources(G, inplace=False)
#     non_source_nodes = list(G2.nodes)
#     return non_source_nodes

# def get_non_orphan_nodes(G):
#     G2 = remove_orphans(G, inplace=False)
#     non_orphan_nodes = list(G2.nodes)
#     return non_orphan_nodes

# def get_nontrivial_nodes(G):
#     G2 = remove_orphans(G, inplace=False)
#     remove_sinks(G2, inplace=True)
#     remove_sources(G2, inplace=True)
#     return list(G2.nodes)