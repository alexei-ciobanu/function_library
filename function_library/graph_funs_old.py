import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import random
import collections

import func_funs as ff
import general_funs as gef
from func_funs import list_roll

def is_connected(G):
    G = nx.Graph(G)
    return nx.is_connected(G)

def spring_electrical_potential(G, layout, node_size=0.01):
    C = 1.0 # Intensity of coulomb's force
    K = 1 # Intensity of Hook's force
    energy = collections.defaultdict(float)
    for u in G.nodes:
        (x,y) = layout[u]
        u_neighbours = list(nx.all_neighbors(G, u))
        f_u = 0
        f0 = 0
        f1 = 0
        for v in G.nodes:
            if u != v:
                (a,b) = layout[v]
                d = np.array([a-x, b-y])
                d_norm = np.sqrt(d[0]**2 + d[1]**2)
                d_norm = np.maximum(node_size, d_norm) # prevent nodes accidently flying off
                # print(d_norm)
                # electrical energy
                f0 += C / d_norm
                if v in u_neighbours:
                    # spring energy
                    f1 += K * d_norm**2
                # print(np.linalg.norm(f0), np.linalg.norm(f1))
        energy[u] = f0 + f1
    total_energy = np.sum([np.linalg.norm(v)**2 for v in energy.values()])

    return total_energy

def spring_electrical_energy_2(G, layout, coulomb=1.0, hook=0.1, node_size=0.001, dt=0.9, update_layout=False):
    C = -1.0 # Intensity of coulomb's force
    K = 1 # Intensity of Hook's force

    force = collections.defaultdict(float)
    for u in G.nodes:
        (x,y) = layout[u]
        for v in G.nodes:
            if u != v:
                (a,b) = layout[v]
                d = np.array([a-x, b-y])
                d_norm = np.linalg.norm(d)
                # coulomb's law
                force[u] += -C * d / d_norm**2
                if v in nx.all_neighbors(G, u):
                    force[u] += K * d
    kinetic = np.sum([np.linalg.norm(v)**2 for v in force.values()])
    # print(layout)
    if update_layout:
        for u in G.nodes:
            x,y = layout[u]
            x,y = np.array([x,y]) + force[u]*dt
            layout[u] = x,y

    return kinetic

def spring_electrical_energy_3(G, layout, node_size=0.01, dt=0.005, update_layout=False):
    C = 1.0 # Intensity of coulomb's force
    K = 1 # Intensity of Hook's force

    energy = collections.defaultdict(float)
    for u in G.nodes:
        (x,y) = layout[u]
        u_neighbours = list(nx.all_neighbors(G, u))
        f_u = 0
        f0 = 0
        f1 = 0
        for v in G.nodes:
            if u != v:
                (a,b) = layout[v]
                d = np.array([a-x, b-y])
                d_norm = np.sqrt(d[0]**2 + d[1]**2)
                d_norm = np.maximum(node_size, d_norm) # prevent nodes accidently flying off
                # print(d_norm)
                # electrical energy
                f0 += -C * d / d_norm**2
                if v in u_neighbours:
                    # spring energy
                    f1 += K * d * d_norm
                # print(np.linalg.norm(f0), np.linalg.norm(f1))
        energy[u] = f0 + f1
    total_energy = np.sum([np.linalg.norm(v)**2 for v in energy.values()])*dt
    # print(layout)
    if update_layout:
        for u in G.nodes:
            x,y = layout[u]
            x,y = np.array([x,y]) + energy[u]*dt
            layout[u] = x,y

    return total_energy

def spring_electrical_energy(G, layout, coulomb=1.0, hook=0.1, node_size=0.001, dt=0.5, update_layout=False):
    """coulomb = 1.0 # Intensity of coulomb's force
hook = 0.1 # Intensity of Hook's force
node_size = 0.001
dt = 0.5 # Time step
    """
    kinetic = 0.0  # kinetic energy
    for u in G.nodes:
        # Compute the acceleration of u
#             print(kinetic)
        (x,y) = layout[u]
        (ax, ay) = (0,0)
        for v in G.nodes:
            if u != v:
                (a,b) = layout[v]
                d = max(node_size, (x-a)*(x-a) + (y-b)*(y-b))
                # coulomb's law
                ax -= coulomb * (a-x)/(d*d)
                ay -= coulomb * (b-y)/(d*d)
        for v in nx.all_neighbors(G, u):
            # Hook's law
            (a,b) = layout[v]
            ax += hook * (a-x)
            ay += hook * (b-y)
        # Update postions
        vx = ax*dt
        vy = ay*dt
        kinetic += vx**2 + vy**2
        if update_layout:
            x = x + vx*dt + 0.5*ax*dt**2
            y = y + vy*dt + 0.5*ay*dt**2
            layout[u] = (x,y)
    return kinetic

def spring_electrical_layout(G, iterations=None, layout=None, debug=False):
    if iterations is None:
        iterations = 500
    if layout is None:
        layout = nx.drawing.layout.kamada_kawai_layout(G, scale=2)
        # layout = {k: 12*v for k,v in layout.items()} # scale by 12 since it seems to work better
    niter = 0
    while iterations > 0:
        niter += 1
        iterations = iterations - 1
        kinetic = spring_electrical_energy_3(G, layout, update_layout=True)
        if kinetic < 1e-12:
            iterations = 0
    if debug:
        print(niter, kinetic)
    return layout

def draw(G, debug=False, figsize=[6,6], **kwargs):
    fig, ax = plt.subplots(1, figsize=figsize)
    '''disconnected graph will fly off'''
    if kwargs.get('pos', None) is None:
        if not is_connected(G):
            print('Warning: graph is not fully connected, using kamada kawai instead of spring-electrical embedding')
            kwargs['pos'] = nx.drawing.layout.kamada_kawai_layout(G)
        else:
            niter = kwargs.pop('iterations', None)
            kwargs['pos'] = spring_electrical_layout(G, iterations=niter, debug=debug)
    nx.draw(G, **kwargs)
    plt.axis('equal')
    return

def list_flatten(l):
    '''flattens list'''
    if type(l) is not list:
        return [l]
    elif l == []:
        return l
    else:
        if type(l[0]) is list:
            return list_flatten(l[0]) + list_flatten(l[1:])
        else:
            return l

def recursive_map(f, x):
    '''
    same as map() except this one traverses the sub lists, applying f() to all
    non-list elements.

    l = [1,[2,3],[4,5,6]]
    list(map(lambda x: [x] + [1], l)) == [[1,1], [[2,3], 1], [[4,5,6], 1]]
    recursive_map(lambda x: [x] + [1], l) == [[1,1], [[2,1], [3,1]], [[4,1], [5,1], [6,1]]]
    '''
    return [f(y) if type(y) is not list else recursive_map(f, y) for y in x]

def recursive_list_map(f, x):
    '''
    same as recursive_map() but f acts on lists instead of list elements.
    
    l = [[6,7,8,9],[3,4,5,6,2]]
    recursive_list_map(lambda x: np.sum(x), l) == [30,20]
    recursive_map(lambda x: np.sum(x), l) == [[6,7,8,9],[3,4,5,6,2]]
    '''
    if type(x[0]) is list:
        return [recursive_list_map(f, y) for y in x]
    else:
        return f(x)

def path_to_edge_list(path):
    edge_list = []
    for pair in zip(path, path[1:]):
        edge_list.append(pair)
    return edge_list

def edge_list_to_path(edge_list):
    path = []
    for edge in edge_list:
        path.append(edge[0])
    return path

def get_node_obj(G, node_str):
    node_obj = G.nodes[node_str]['weakref']()
    return node_obj

def get_optical_network(model):
    optical_network = model.network.copy()
    for node_str in model.network.nodes:
        node_obj = get_node_obj(optical_network, node_str)
        # 0 being the enum for optical node
        if node_obj.type.value == 0:
            # keep optical node
            pass
        else:
            # remove non optical node
            # the edges should be updated accordingly by networkx
            optical_network.remove_node(node_str)
    return optical_network

def ordered_intersect(A, B):
    # assumes paths A,B have no duplicates
    if len(A) < len(B):
        A, B = B, A
    t = []
    for j, a in enumerate(A):
        for i, b in enumerate(B):
            if a is b:
                tt = []
                for k in range(0, len(B)):
                    if B[(i-k)%len(B)] is A[(j-k)%len(A)]:
                        tt = [A[(j-k)%len(A)]] + tt 
                    else:
                        break
                for k in range(1, len(B)):
                    if B[(i+k)%len(B)] is A[(j+k)%len(A)]:
                        tt = tt + [A[(j+k)%len(A)]]
                    else:
                        break
                t.append(tt)
                break
    return t

def ordered_difference(A, B):
    # assumes paths A,B have no duplicates
    if len(A) < len(B):
        A, B = B, A
    t = []
    for j,a in enumerate(A):
        for i,b in enumerate(B):
            if a is b:
                del_ind = []
                for k in range(0,len(B)):
                    if B[(i-k)%len(B)] is A[(j-k)%len(A)]:
                        del_ind.append((j-k)%len(A)) 
                    else:
                        break
                for k in range(1,len(B)):
                    if B[(i+k)%len(B)] is A[(j+k)%len(A)]:
                        del_ind.append((j+k)%len(A)) 
                    else:
                        break
                t.append([A[i] for i in range(len(A)) if i not in del_ind])
#                 print(del_ind)
                break
    return t
    
def get_edge_data(G,u,v,data=True):
    if data is True:
        # return all dict members
        return G.edges[u,v]
    else:
        # return the "data" member
        return G.edges[u,v][data]

def get_edge(G, u, v, k=None):
    '''
    Returns the dictionary containing edge data between nodes u and v.
    If multiple edges between u and v exist a multiplicity index k has to be specified.

    The edge dictionary is a reference of the one stored in a graph and so mutable changes
    to the edge dictionary will be propagated to the graph.
    '''
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        if k is None:
            raise Exception('multiplicity index cannot be None for a MultiGraph')
        edge = G.edges[u, v, k]
    elif isinstance(G, (nx.Graph, nx.DiGraph)):
        edge = G.edges[u, v]
    else:
        raise Exception(f'unknown graph type {G}')
    return edge

def get_edges(G, u, v, k=None):
    '''
    Returns a list of dictionaries containing the edge data between nodes
    u and v. By default all edges between u and v are returned. A multiplicity
    index k can be specified. 
    
    If k is a list then it is interpreted as a list of multiplicity keys as a 
    list is not hashable and hence cannot be a multiplicity key in networkx.
    '''
    edges = []
    if isinstance(G, (nx.MultiGraph, nx.MultiGraph)):
        if k is None:
            for multi_key in G[u][v]:
                edges.append(get_edge(G, u, v, multi_key))
        else:
            if isinstance(k, list):
                for multi_key in k:
                    edges.append(get_edge(G, u, v, multi_key))
    else:
        edges = [get_edge(G, u, v, k)]
    return edges

def get_edge_operator(G,u,v,op_type='LCT'):
    edge_owner_weakref = get_edge_data(G,u,v,'owner')
    return edge_owner_weakref().get_operator(u,v,op_type=op_type)

def get_path_operators(G,path,op_type='LCT'):
    if path == []:
        return []
    elif type(path[0]) is list:
        return [get_path_operators(G,path[0],op_type=op_type)] + get_path_operators(G,path[1:],op_type=op_type)
    else:
        t = []
        N = len(path)
        for i in range(1,N):
            u = path[i-1]
            v = path[i]
            t.append(get_edge_operator(G,u,v,op_type=op_type))
        return t
    
def reduce_operator(op_list):
    acc = op_list[0]
    for op in op_list[1:]:
        acc = op@acc
    return acc

def get_reduced_path_operators(G,path,op_type='LCT'):
    return recursive_list_map(reduce_operator,get_path_operators(G,path,op_type=op_type))

def get_edge_operator_description(G,u,v):
    edge_owner_weakref = get_edge_data(G,u,v,'owner')
    return edge_owner_weakref().get_operator(u,v,op_type='str')

def get_path_operator_descriptions(G,path):
    if path == []:
        return []
    elif type(path[0]) is list:
        return [get_path_operator_descriptions(G,path[0])] + get_path_operator_descriptions(G,path[1:])
    else:
        t = []
        N = len(path)
        for i in range(1,N):
            u = path[i-1]
            v = path[i]
            t.append(get_edge_operator_description(G,u,v))
        return t

def get_cycle_operators(G,cycle):
    t = []
    for u,v in zip(cycle,list_roll(cycle, inplace=False)):
        t.append(get_edge_operator(G,u,v))
    return t

def get_all_cycle_couple_paths(G,A,B):
    A = list_flatten(A)
    B = list_flatten(B)
    return list(nx.all_simple_paths(G,A[0],B[0]))

def get_cycle_couple_path(G, A, B):
    # just get the first one
    return get_all_cycle_couple_paths(G, A, B)[0]

def get_all_node_paths(G, node1, node2):
    return list(nx.all_simple_paths(G, node1, node2))

def are_nodes_unique(path):
    seen = []
    for node in path:
        if node not in seen:
            seen.append(node)
        else:
            return False
    return True

def only_unique_nodes(path):
    seen = []
    for node in path:
        if node not in seen:
            seen.append(node)
    return seen

def cycle_contains_cycle(A, B):
    if len(A) < len(B):
        return False
    else:
        o_isec = ordered_intersect(A, B)
        if o_isec == []:
            return False
        m = max(o_isec, key=len)
        if len(m) == len(B):
            return True
        else:
            return False

def alexei_cycles_prune(cycles):
    cycles = cycles.copy()
    cycles.sort(key=len, reverse=True) # sort so longest cycles are first
    del_ind = []
    # find all cycles that contain another cycle
    for i in range(len(cycles)):
        for j in range(i, len(cycles)):
            if cycle_contains_cycle(cycles[i], cycles[j]):
                del_ind.append(i)
                break
    return [x for i, x in enumerate(cycles) if i not in del_ind]

def get_overlapping_node(A, B):
    for i, ni in enumerate(A):
        for j, nj in enumerate(B):
            if ni is nj:
                return ni, i, j
    return None, None, None

def append_node(node, g_path):
    if g_path == []:
        return []
    else:
        if type(g_path[0]) is list:
            return [append_node(node, g_path[0])] + append_node(node, g_path[1:])
        else:
            return g_path + [node]
        
def cycle2path(g_cycle):
    c = g_cycle[0]
    while type(c) is list:
        c = c[0]
    return append_node(c, g_cycle)

def alexei_cycle_group(cycles):  
    # group all the pruned cycles that share a node
    # also make sure the grouped cycles start at the shared node
    cycles = cycles.copy() # avoid making changes to state
    if cycles == []:
        return []
    N = len(cycles)
    #print(N,cycles)
    ci = cycles[0]
    for j, cj in enumerate(cycles[1:]):
        nk, k, m = get_overlapping_node(ci, cj)
        if nk is not None:
            # roll the cycles so that they both start on the overlapping node
            ti = list_roll(ci, k)
            tj = list_roll(cj, m)
            assert(nk is ti[0] is tj[0])
            if N < 2:
                # base case for recursion
                return [[ti, tj]]
            else:
                # append the coupled cycles and recurse
                del cycles[j]
                return [[ti, tj]] + alexei_cycle_group(cycles[1:])
    if N < 2:
        return [[ci]]
    else:
        return [[ci]] + alexei_cycle_group(cycles[1:])

def alexei_cycle_group2(a_cycles):
    ungrouped_cycles = a_cycles.copy()
    grouped_cycles = []
    oset = set()
    while ungrouped_cycles:
        ci = ungrouped_cycles.pop()
        for j, cj in enumerate(ungrouped_cycles):
            oset = set(ci) & set(cj)
            if len(oset) > 0:
                overlapping_node = random_set_element(oset)
                ci = anchor_cycle(ci, overlapping_node)
                cj = anchor_cycle(cj, overlapping_node)
                ungrouped_cycles.pop(j)
                grouped_cycles.append([ci,cj])
                break
        # fallen out of for loop
        # check if it was because of break or not
        if len(oset) == 0:
            # ci found no groups, append it as its own group
            grouped_cycles.append([ci])
    return grouped_cycles

def alexei_cycle_group3(p_cycles, anchor_nodes=[]):
    '''
    Takes a list of primitive/pruned cycles and groups them by a shared node.

    TODO separate logic of determining the shared nodes and the logic of grouping the cycles
    '''
    anchor_nodes = anchor_nodes.copy()
    ungrouped_cycles = p_cycles.copy()
    grouped_cycles = []
    oset = set()
    while ungrouped_cycles:
        ci = ungrouped_cycles.pop()
        group = [ci]
        # look in all other cycles for an overlapping node to form a group
        for j, cj in enumerate(ungrouped_cycles):
            oset = set(ci) & set(cj)
            if len(oset) > 0:
                # print(oset)
                # found a potential group, check if it contains a given anchor node
                anchor_oset = set(anchor_nodes) & oset
                if anchor_oset:
                    group_node = random_set_element(anchor_oset)
                    anchor_nodes.remove(group_node)
                else:
                    group_node = random_set_element(oset)
                anchor_cycle(ci, group_node)
                # now find all cycles that belong to that group
                for k, ck in enumerate(ungrouped_cycles):
                    if group_node in ck:
                        c = anchor_cycle(ck, group_node, inplace=False)
                        group.append(c)
                        ungrouped_cycles.remove(ck)
                # found all possible groups for group_node move to next ci
                break
        if len(group) == 1:
            # ci didn't find a match
            # still want to apply an anchor node in this case
            anchor_oset = set(anchor_nodes) & set(ci)
            if anchor_oset:
                group_node = random_set_element(anchor_oset)
                anchor_nodes.remove(group_node)
            else:
                group_node = random_set_element(set(ci))
            anchor_cycle(ci, group_node)
        grouped_cycles.append(group)
    if anchor_nodes:
        print(f'warning: unused anchor nodes remain {anchor_nodes}')
    return grouped_cycles


def get_cycle_couple_matrix(graph, g_cycles):
    # g_cycles are the pruned and grouped cycles
    # returns a matrix of the 
    N = len(g_cycles)
    m_couple = [[None for x in range(N)] for y in range(N)] # preallocate list
    for i,ci in enumerate(g_cycles):
        for j,cj in enumerate(g_cycles):
            if i == j:
                # append the first node to diagonal entries to turn them from cycles to paths
                m_couple[i][j] = cycle2path(ci)
            else:
                all_paths = get_all_cycle_couple_paths(graph,ci,cj)
                remaining_cycles = [c for ind,c in enumerate(g_cycles) if ind not in [i,j]]
                del_ind = []
                for k,path in enumerate(all_paths):
                    if any([node in list_flatten(remaining_cycles) for node in path if node not in list_flatten([ci,cj])]):
                        # remove any paths that go through any cavity other than ci or cj
                        del_ind.append(k)
                m_couple[j][i] = [path for ind,path in enumerate(all_paths) if ind not in del_ind]
    return m_couple

def get_connectivity_matrix(m_couple):
    '''Mostly for debugging. The elements of the connectivity elements
    indicate the amount of terms for that coupling.'''
    N = len(m_couple)
    m_conn = [[None for x in range(N)] for y in range(N)]
    m_head = [[None for x in range(N)] for y in range(N)]
    for i,col in enumerate(m_couple):
        for j,el in enumerate(col):
            if el == []:
                m_conn[i][j] = 0
            elif type(el[0]) is list:
                m_conn[i][j] = len(el)
                m_head[i][j] = el[0][0]
            else:
                m_conn[i][j] = 1
                m_head[i][j] = el[0]
    return np.array(m_conn),m_head

def alexei_cycles_align(p_cycles,align_nodes=[]):
    a_cycles = []
    align_nodes = only_unique_nodes(align_nodes)
    for path in p_cycles:
        incidence_count = np.sum([align_node in path for align_node in align_nodes])
        if incidence_count > 1:
            raise Exception(f'Multiple align nodes from {align_nodes} in path {path}. Cannot align path to multiple nodes.')
        elif incidence_count == 0:
            a_cycles.append(path)
        elif incidence_count == 1:
            align_node = [align_node for align_node in align_nodes if align_node in path][0]
            while path[0] != align_node:
                path = list_roll(path,1)
            a_cycles.append(path)
        else:
            raise Exception(f'Undefined alignment')
    return a_cycles

def graph2n_couple(graph,align_nodes=[]):
    cycles = list(nx.simple_cycles(graph))
    p_cycles = alexei_cycles_prune(cycles)
    if align_nodes == []:
        print('Warning: no alignment nodes given. Node order will be \
determined randomly by networkx. Networkx generates random seed \
for node ordering at import time.')
    a_cycles = alexei_cycles_align(p_cycles, align_nodes)
    g_cycles = alexei_cycle_group(a_cycles)
    n_couple = get_cycle_couple_matrix(graph, g_cycles)
    return n_couple

def get_n_couple_diag(n_couple):
    diag = []
    for i in range(len(n_couple)):
        diag.append(n_couple[i][i][0][0])
    return diag

def model2n_couple(ifo):
    graph = get_optical_network(ifo)
    return graph2n_couple(graph)

def build_m_couple(ifo):
    graph = get_optical_network(ifo)
    n_couple = graph2n_couple(graph)
    p_couple = get_path_operator_descriptions(graph,n_couple)
    return p_couple

def prune_invalid_path(path, solved_nodes, del_invalid=False):
    if any(node in path[1:-1] for node in solved_nodes):
        if del_invalid:
            return None
        else:
            return [path[0], None, path[-1]]
    else:
        return path

def prune_invalid_paths(paths, solved_nodes, del_invalid=False):
    out_paths = recursive_list_map(
        lambda path: prune_invalid_path(
            path, solved_nodes, del_invalid),
        paths)

    if del_invalid:
        out_paths = ff.recursive_list_comprehension(
            lambda x: x, out_paths, lambda x: x is not None)

    return out_paths

def get_all_valid_paths(G, sources, targets, del_invalid=False):
    '''
    Returns an (N, M) matrix where N is the number of target nodes M is the 
    number of source nodes. The elements of the matrix are lists of paths 
    that sum to that matrix element.

    The intention is to do a recursive_list_map on this matrix to get
    the path operator matrix, which is then acted on the source
    field vector to get the target field vector which can be built.

    * first index: row of path matrix
    * second index: column of path matrix
    * third index: list of paths that sum to get that matrix element
    '''
    out_list = []
    for target_node in targets:
        temp = []
        for source_node in sources:
            if source_node == target_node:
                pruned_paths = [[source_node, target_node]]
            else:
                if target_node in sources:
                    paths = [[source_node, None, target_node]]
                else:
                    paths = list(nx.all_simple_paths(G, source_node, target_node))
                    if paths == []:
                        paths = [[source_node, None, target_node]]
                pruned_paths = prune_invalid_paths(paths, sources, del_invalid)
            temp.append(pruned_paths)
        out_list.append(temp)

    # return prune_invalid_paths(out_list, sources, del_invalid)
    return out_list

def overlap_set_matrix(sets):
    N = len(sets)
    out = np.zeros([N,N], dtype=object)
    for i in range(N):
        for j in range(N):
            out[i][j] = set(sets[i]) & set(sets[j])
    return out

def lower_off_diagonal_indices(M):
    N = len(M)
    inds = np.tril_indices(N, k=-1)
    return inds

def overlap_of_overlap_sets(sets):
    '''
    Computes the overlap set of nonempty overlap sets.

    Useful for checking if there are more than two sets 
    that share the same element.
    '''
    oset = overlap_set_matrix(sets)
    ldiag_oset_ind = lower_off_diagonal_indices(oset)
    ldiag_oset = [x for x in oset[ldiag_oset_ind] if len(x) > 0]
    return overlap_set_matrix(ldiag_oset)

def random_set_element(s):
    '''
    Python 3.9 deprecated doing random sample on a set. Now they recommend to 
    cast it to a list.
    '''
    el = random.sample(list(s),1)[0]
    return el

def prune_path(path):
    return [path[0], None, path[-1]]

def permute_cycle(cycle):
    N = len(cycle)
    i = np.random.randint(0,len(cycle))
    return list_roll(cycle, i, inplace=False)

def shuffle_permute_cycles(cycles):
    cycles = cycles.copy()
    random.shuffle(cycles)
    return [permute_cycle(cycle) for cycle in cycles]

def anchor_cycle(cycle, anchor_node, inplace=True):
    if anchor_node not in cycle:
        raise Exception(f'Anchor node {anchor_node} not in cycle {cycle}')
    ind = cycle.index(anchor_node)
    return list_roll(cycle, ind, inplace=inplace)

def random_node(G, N=1):
    '''
    Get a random sample of N nodes from graph G without replacement.
    '''
    nodes = random.sample(list(G.nodes()), k=N)
    return nodes

def get_outgoing_edges(G, node):
    edges = []
    for out_node in G[node]:
        edges.extend(get_edges(G, node, out_node))
    return edges

def get_incoming_edges(G, node):
    pred = G.predecessors(node)
    edges = []
    for pn in pred:
        edges.extend(get_edges(G, pn, node))
    return edges

def get_multiplicity_indices(G, u, v):
    return list(G[u][v].keys())

def remove_edge(G, u, v, k=None, inplace=True):
    if not inplace:
        G = copy_graph(G)
    if isinstance(G, (nx.MultiGraph, nx.MultiGraph)):
        if k is None:
            raise Exception('multiplicity index cannot be None for a MultiGraph')
        try:
            G.remove_edge(u, v, k)
        except KeyError:
            pass
    elif isinstance(G, (nx.Graph, nx.DiGraph)):
        try:
            G.remove_edge(u, v)
        except KeyError:
            pass
    else:
        raise Exception(f'unknown graph type {G}')
    return G

def remove_edges(G, u, v, inplace=True):
    if not inplace:
        G = copy_graph(G)
    ks = get_multiplicity_indices(G, u, v)
    for k in ks:
        G.remove_edge(u, v, k)
    return G

def get_sink_nodes(G):
    '''
    A sink node is a node with no outgoing edges.
    '''
    out_degree_dict = gef.flip_dict(dict(G.out_degree))
    try:
        sink_nodes = out_degree_dict[0]
    except KeyError:
        sink_nodes = []
    return sink_nodes

def get_source_nodes(G):
    '''
    A source node is a node with no incoming edges.
    '''
    in_degree_dict = gef.flip_dict(dict(G.in_degree))
    try:
        source_nodes = in_degree_dict[0]
    except KeyError:
        source_nodes = []
    return source_nodes

def get_orphan_nodes(G):
    '''
    An orphan node is a node with no edges.
    '''
    in_degree_dict = gef.flip_dict(dict(G.in_degree))
    out_degree_dict = gef.flip_dict(dict(G.out_degree))
    try:
        orphan_nodes = list(set(in_degree_dict[0]) & set(out_degree_dict[0]))
    except KeyError:
        orphan_nodes = []
    return orphan_nodes

def copy_graph(G):
    '''
    A trick I often see in networkx's codebase to copy a graph.
    type(G) returns the class of G (e.g. nx.DiGraph) which can accept a graph to
    make a copy of.
    Useful for writing 'pure' functions on graphs.
    '''
    return type(G)(G)

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
        return remove_sinks(G, recursive=True)
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
        return remove_sources(G, recursive=True)
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

def get_non_sink_nodes(G):
    G2 = remove_sinks(G, inplace=False)
    non_sink_nodes = list(G2.nodes)
    return non_sink_nodes

def get_non_source_nodes(G):
    G2 = remove_sources(G, inplace=False)
    non_source_nodes = list(G2.nodes)
    return non_source_nodes

def get_non_orphan_nodes(G):
    G2 = remove_orphans(G, inplace=False)
    non_orphan_nodes = list(G2.nodes)
    return non_orphan_nodes

def get_nontrivial_nodes(G):
    G2 = remove_orphans(G, inplace=False)
    remove_sinks(G2, inplace=True)
    remove_sources(G2, inplace=True)
    return list(G2.nodes)

def edge_existsQ(G, ni, no):
    try:
        G.edges[ni, no]
    except KeyError:
        return False
    return True

def annotate_paths_to_graph(G, inplace=False):
    if not inplace:
        G = copy_graph(G)
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        for edge in set(G.edges()):
            ni, no = edge
            edges = get_edges(G, ni, no)
            for edge in edges:
                edge['path'] = [ni, no]
    elif isinstance(G, (nx.DiGraph, nx.Graph)):
        for edge in G.edges():
            ni, no = edge
            G.edges[ni,no]['paths'] = [[ni, no]]
    return G

def add_edge_with_paths(G, ni, no, new_paths=None, inplace=True):
    '''
    networkx doesn't support multiple edges between the same nodes.
    This is a hack to store degenerate edges as edge metadata
    '''
    if not inplace:
        G = copy_graph(G)

    if new_paths is None:
        new_paths = [[ni, no]]
    
    # check if edge exists in graph
    if edge_existsQ(G, ni, no):
        G.edges[ni, no]['paths'] += new_paths
    else:
        G.add_edge(ni, no, paths=new_paths)
    return G

def get_paths(G, ni, no):
    paths = [[ni, None, no]]
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        try:
            edges = get_edges(G, ni, no)
            paths = [e['path'] for e in edges]
        except KeyError:
            pass
    elif isinstance(G, (nx.DiGraph, nx.Graph)):
        try:
            paths = G.edges[ni, no]['paths']
        except KeyError:
            pass
    else:
        raise Exception(f'undefined graph type {G}')
    return paths

def get_incoming_paths(G, node):
    pred = G.predecessors(node)
    out = []
    for pn in pred:
        out += get_paths(G, pn, node)
    return out
    
def adjacency_matrix(G, sparse=False):
    '''
    networkx returns adjacency matrix in sparse format by default
    '''
    M = nx.adjacency_matrix(G)
    if not sparse:
        M = M.asformat('array')
    return M

def path_counting_matrix(G):
    '''
    Like an adjacency matrix except counts multiplicity of edges using 'paths'
    '''
    nodes = list(G.nodes)
    N = len(nodes)
    path_matrix = np.zeros([N, N], dtype=np.int32)
    for i, ni in enumerate(nodes):
        for j, no in enumerate(nodes):
            try:
                num_paths = len(G.edges[ni, no]['paths'])
            except KeyError:
                num_paths = 0
            path_matrix[i,j] = num_paths
    return path_matrix

def absorb_node(G, node, residual_nodes=[], keep_node=False, inplace=False):
    if not inplace:
        G = copy_graph(G)
    pred = list(G.predecessors(node))
    succ = list(G.successors(node))
    sources = get_source_nodes(G)
    # check if node has no self-loops or is not a source or a residual node
    if not (node in pred or node in succ or node in residual_nodes or node in sources):
        # apply reduction
        for pn in pred:
            for sn in succ:
                pn_paths = G.edges[pn, node]['paths']
                sn_paths = G.edges[node, sn]['paths']
                new_paths = [pp[:-1]+sp for sp in sn_paths for pp in pn_paths]
                add_edge_with_paths(G, pn, sn, new_paths=new_paths)
        if keep_node:
            # turn node into a sink
            for sn in succ:
                G.remove_edge(node, sn)
        else:
            G.remove_node(node)
    return G

def absorb_node_multigraph(G, node, residual_nodes=[], inplace=False):
    if not inplace:
        G = copy_graph(G)
    pred = list(G.predecessors(node))
    succ = list(G.successors(node))
    sources = get_source_nodes(G)
    # check if node has no self-loops or is not a source or a residual node
    if not (node in pred or node in succ or node in residual_nodes or node in sources):
        # apply reduction predecessors need to be in the inner loop
        for sn in succ:
            for pn in pred:
                pks = get_multiplicity_indices(G, pn, node)
                sks = get_multiplicity_indices(G, node, sn)
                # loop over all multiedges
                for pk in pks:
                    for sk in sks:
                        p_path = get_edge(G, pn, node, pk)['path']
                        s_path = get_edge(G, node, sn, sk)['path']
                        G.add_edge(pn, sn, path=p_path[:-1]+s_path)
            remove_edges(G, node, sn)
    return G

def remove_nodes_from(G, seq, inplace=False):
    '''
    Same as networkx but has option to not modify state
    '''
    if not inplace:
        G = copy_graph(G)
    G.remove_nodes_from(seq)
    return G

def graph_reduction(G, residual_nodes=[], keep_nodes=False, inplace=False):
    if not inplace:
        G = copy_graph(G)
    all_nodes = list(G.nodes)
    while all_nodes:
        node = random.choice(all_nodes)
        absorb_node(G, node, residual_nodes=residual_nodes, keep_node=keep_nodes, inplace=True)
        all_nodes.remove(node)
    return G

def multigraph_reduction(G, residual_nodes=[], inplace=False):
    if not inplace:
        G = copy_graph(G)
    all_nodes = list(G.nodes)
    while all_nodes:
        node = random.choice(all_nodes)
        absorb_node_multigraph(G, node, residual_nodes=residual_nodes, inplace=True)
        all_nodes.remove(node)
    return G