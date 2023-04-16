import networkx as nx
import numpy as np

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

def random_dag(n_nodes, edge_frac=None, tree_generator=None, tree_generator_args=None, debug=False):
    '''Create a random DAG. The DAG is created by iteratively adding edges between random pairs of nodes of a tree while guaranteeing that the resulting graph at every step is a DAG.

    Parameters
    ----------
    n_nodes : int
        The number of nodes in the DAG
    edge_frac : float, optional
        The fraction of edges to add, by default None. If None, a random number between 0 and 1 is used. 0 corresponds to a tree, 1 corresponds to a complete graph.
    tree_generator : function, optional
        The function to use to generate the tree, by default None. If None, nx.gn_graph is used.
    tree_generator_args : dict, optional
        The arguments to pass to the tree generator, by default None. If None, {'kernel': lambda x: 1/x} is used.
    seed : int, optional
        The random seed, by default None

    Returns
    -------
    networkx.DiGraph
        The DAG
    '''
    if tree_generator is None:
        tree_generator = nx.gn_graph
    if tree_generator_args is None:
        tree_generator_args = {'kernel': lambda x: 1/x}

    # create a random tree with n_nodes
    dag = nx.DiGraph(tree_generator(n_nodes, **tree_generator_args))
    
    # list all pairs of nodes that don't have an edge
    pairs = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if not dag.has_edge(i, j) and not dag.has_edge(j, i):
                pairs.append((i, j))

    # determine the number of edges to add
    edge_iter_max = len(pairs)
    if edge_frac is None:
        edge_frac = np.random.rand()
    edge_iter = int(np.ceil(edge_frac * edge_iter_max))

    if debug:
        print(edge_iter)
    if debug:
        print(len(pairs))

    for _ in range(edge_iter):
        # pop a random pair
        ni, nj = pairs.pop(np.random.randint(len(pairs)))
        if nx.has_path(dag, ni, nj):
            # If the pair is connected, add a short circuit. Guaranteed to not create a cycle.
            dag.add_edge(ni, nj)
        else:
            # If the pair is not connected, add an edge going backwards. This is guaranteed to not create a cycle if the dag previously had no cycles.
            dag.add_edge(nj, ni)
    if debug:    
        print(pairs)
    return dag

def fully_connected_DAG(n):
    '''
    Create a fully connected DAG.

    Parameters
    ----------
    n : int
        Number of nodes.

    Returns
    -------
    networkx.DiGraph
        Fully connected DAG.
    '''
    DAG = nx.DiGraph()
    DAG.add_edges_from([(u,v) for u in range(n) for v in range(n) if u < v])
    return DAG

def random_DAG(n, p, keep_all_nodes=False):
    '''
    Create a random DAG.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Probability of adding an edge.
    keep_all_nodes : bool, optional
        If True, all nodes will be kept in the DAG even if they have no edges. The default is False.

    Returns
    -------
    networkx.DiGraph
        Random DAG.
    '''
    DAG = nx.DiGraph()
    if keep_all_nodes:
        DAG.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u+1, n):
            if np.random.rand() < p:
                DAG.add_edge(u, v)
    return DAG

def random_connected_DAG(n, p):
    '''
    Create a random connected DAG.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Probability of adding an edge.

    Returns
    -------
    networkx.DiGraph
        Random connected DAG.
    '''
    DAG = fully_connected_DAG(n)
    for u in range(n):
        for v in range(u+1, n):
            if np.random.rand() < 1-p:
                DAG.remove_edge(u, v)
                # check if DAG is still connected
                if not nx.is_weakly_connected(DAG):
                    DAG.add_edge(u, v)
    return DAG

def random_origin_terminal_DAG(n, p, origin_name='a', terminal_name='b'):
    '''
    Create a random DAG with a single source and sink.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Probability of adding an edge.
    origin_name : str, optional
        Name of the origin node. The default is 'a'.
    terminal_name : str, optional
        Name of the terminal node. The default is 'b'.

    Returns
    -------
    networkx.DiGraph
        Random DAG with a single source and sink.
    '''
    DAG = random_DAG(n, p, keep_all_nodes=True)
    source_nodes = [n for n in DAG.nodes if DAG.in_degree(n) == 0]
    sink_nodes = [n for n in DAG.nodes if DAG.out_degree(n) == 0]
    # add an edge from origin node to every source node
    DAG.add_edges_from([(origin_name, n) for n in source_nodes])
    # add an edge from every sink node to terminal node
    DAG.add_edges_from([(n, terminal_name) for n in sink_nodes])
    return DAG