import networkx as nx

from .graph import graph_to_digraph
from .manipulate import remove_sources, remove_sinks, levy_low_reduction

def random_k_out_graph(N, k=3, alpha=2.5, self_loops=True):
    G = nx.random_k_out_graph(N, k, alpha, self_loops)
    G = nx.DiGraph(G) # remove multi-edges
    return G

def random_regular_graph(N, k=3, directed=True):
    G = nx.random_regular_graph(k, N)
    if directed:
        G = graph_to_digraph(G)
    return G

def random_graph(N=100, k=3, N_min=5):
    G = nx.DiGraph()
    while len(G.nodes) < N_min:
        G = random_regular_graph(N, k=k)
        remove_sinks(G, inplace=True)
        remove_sources(G, inplace=True)
    return G

def random_low_levy_graph(N=100, k=3, N_min=5):
    G = nx.DiGraph()
    while len(G.nodes) < N_min:
        G = random_graph(N=N, k=k, N_min=N_min)
        levy_low_reduction(G)
    return G 
