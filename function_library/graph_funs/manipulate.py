import random
import collections
import functools

import networkx as nx

from .graph import copy_graph, get_sink_nodes, get_source_nodes, get_orphan_nodes, get_path_nodes, get_multiedge

def _absorb_node(G, node, keep_node=False, inplace=True):
    if not inplace:
        G = copy_graph(G)
    pred = list(G.predecessors(node))
    succ = list(G.successors(node))
    if node not in pred:
        # apply reduction
        for pn in pred:
            for sn in succ:
                G.add_edge(pn,sn)
        if keep_node:
            # turn node into a sink
            for sn in succ:
                G.remove_edge(node, sn)
        else:
            G.remove_node(node)
    return G

def remove_sinks(G, recursive=True, inplace=False):
    '''
    Removes nodes with out degree 0 from graph G.
    Sometimes removing a out degree 0 node creates a new out degree 0 node.
    So it is necessary to remove out degree 0 nodes recursively.
    '''
    if not inplace:
        G = copy_graph(G)
    sink_nodes = get_sink_nodes(G)
    if sink_nodes == []:
        # Graph has no sink nodes; break recursion 
        return G
    G.remove_nodes_from(sink_nodes)
    if recursive:
        return remove_sinks(G, recursive=True, inplace=True)
    return G

def remove_sources(G, recursive=True, inplace=False):
    '''
    Removes nodes with in degree 0 from graph G.
    Sometimes removing a in degree 0 node creates a new in degree 0 node.
    So it is necessary to remove in degree 0 nodes recursively.
    '''
    if not inplace:
        G = copy_graph(G)
    source_nodes = get_source_nodes(G)
    if source_nodes == []:
        # Graph has no sink nodes; break recursion 
        return G
    G.remove_nodes_from(source_nodes)
    if recursive:
        return remove_sources(G, recursive=True, inplace=True)
    return G

def remove_orphans(G, inplace=False):
    '''
    Removes nodes with in and out degree 0 from graph G.
    This should not need to be recursive.
    '''
    if not inplace:
        G = copy_graph(G)
    orphan_nodes = get_orphan_nodes(G) 
    if len(orphan_nodes) > 0:
        G.remove_nodes_from(orphan_nodes)
    return G

def remove_multiedge(G, u, v, inplace=False):
    if not inplace:
        G = copy_graph(G)
    if isinstance(G, nx.MultiGraph):
        edges = get_multiedge(G, u, v)
        G.remove_edges_from(edges)
    else:
        G.remove_edge(u,v)
    return G

def integer_node_relabel(G, inplace=True, return_mapping=False):
    copy = not inplace
    mapping = {n:i for i,n in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, mapping, copy=copy)
    if return_mapping:
        return G, mapping
    else:
        return G

def simple_reduction(G, inplace=True):
    if not inplace:
        G = copy_graph(G)
    
    remove_orphans(G, inplace=True)
    remove_sinks(G, inplace=True)
    remove_sources(G, inplace=True)

    return G

def compress_paths(G, inplace=True, recursive=True):
    if not inplace:
        G = copy_graph(G)
    
    path_nodes = get_path_nodes(G)
    if path_nodes == []:
        return G
        
    for n in path_nodes:
        _absorb_node(G, n, inplace=True)
    
    if recursive:
        return compress_paths(G, inplace=True, recursive=True)

def levy_low_reduction(G, inplace=True):
    '''A bog standard reduction that most graph algorithms assume
    has already been done.
    
    The reduction is just removing all sinks, sources, and paths.
    '''
    if not inplace:
        G = copy_graph(G)
    
    remove_orphans(G, inplace=True)
    remove_sinks(G, inplace=True)
    remove_sources(G, inplace=True)
    compress_paths(G, inplace=True)
    
    return G

def fork_and_merge(e, new_node):
    e1 = e[0], new_node
    e2 = new_node, e[1]
    e3 = e[0], e[1]
    return e1, e2, e3

def grow_path(e, new_node):
    e1 = e[0], new_node
    e2 = new_node, e[1]
    return e1, e2

def add_sink(e, new_node, pos=None):
    if pos is None:
        pos = random.randint(0, 1)
    e1 = e[pos], new_node
    return e, e1

def add_source(e, new_node, pos=None):
    if pos is None:
        pos = random.randint(0, 1)
    e1 = new_node, e[pos]
    return e, e1 

def grow_graph(
    G=None,
    N_steps=31,
    rule_list=(fork_and_merge, grow_path, add_sink, add_source), 
    rule_probability=(0.4, 1.0, 0.1, 0.1),
    inplace=False
):

    if G is None:
        G = nx.DiGraph([(0,1)])

    edge_list = list(G.edges)
    maxnode = max(G.nodes)

    for i in range(maxnode+1, maxnode+N_steps+1):
        j = random.randint(0, len(edge_list)-1)
        e = edge_list.pop(j)
        found_rule = False
        while not found_rule:
            k = random.randint(0, len(rule_list)-1)
            if rule_probability[k] > random.random():
                found_rule = True
                rule = rule_list[k]
        new_edges = rule(e, i)
        edge_list.extend(new_edges)

    if inplace:
        dis = set(G.edges) - set(edge_list)
        G.remove_edges_from(dis)
        G.add_edges_from(edge_list)
    else:
        G = nx.DiGraph(edge_list)
    return G