import numpy as np
import networkx as nx

from .graph import copy_graph
from .manipulate import remove_orphans
import general_funs as gef

def graphviz_draw(
    network=None, 
    draw_orphans=False, 
    draw_labels=True, 
    angle=0, 
    overlap=True, 
    scale=1.0, 
    ratio=None, 
    size=(13, 7), 
    pad=(1.0, 1.0), 
    format='svg', 
    layout='neato', 
    ipython_display=True, 
    graph_attrs=dict(), 
    node_attrs=dict(), 
    edge_attrs=dict(), 
    out_keys=None, 
    **kwargs
):
    """Draws a |graphviz| figure using |neato| layout.
    The default settings are tested to produce a passable drawing of the 
    aLIGO DRMI graph.

    Parameters
    ----------
    angle : float or bool
        The angle parameter rotates the graph by |angle| degrees relative to the 
        first edge in the graph, which most of the time is the edge coming out 
        of the laser. Set |angle=False| to disable rotation and let graphviz
        decide how to rotate the graph.

    overlap : bool or str
        Setting for how graphviz deals with node overlaps. Set to False for
        graphviz to attempt to remove overlaps. Note that overlap removal runs as 
        a post-processing step after initial layout and usually makes the graph look
        worse.

    ratio : float
        Post processing step to stretch the graph. Used for stretching horizontally
        to compoensate for wider nodes to fit node labels.

    The svg format sometimes crops the image too hard, which results in clipped
    nodes or edges, if that happens increase the |pad| graph_attr or use the
    |png| format.

    By default |neato| performs the graph energy minimization using stress 
    majorization from MDS theory. Setting the |mode| graph attribute to |sgd| 
    performs the energy minimization using stochastic gradient descent instead. 
    SGD seems to converge to a global minimum faster than majorization but SGD 
    iterations are also more expensive. Overall it seems like SGD wins, which is 
    why I set it as the default.

    The |scale| graphviz parameter only seems to work for neato layouts.
    The |ratio| graphviz parameter seems to override any manual edge positioning.
    Instead we manually calculate new positions based on |scale| and |ratio|.
    
    Examples
    --------
    import finesse.ligo
    import finesse.plotting
    kat = finesse.ligo.make_aligo()
    finesse.plotting.graph.graphviz_draw(kat)
    """
    if not draw_orphans:
        G = remove_orphans(network, inplace=False)
    else:
        G = copy_graph(network)
    A = nx.drawing.nx_agraph.to_agraph(G)

    # remove unnecessary metadata from DOT file
    for node in A.nodes():
        for k in node.attr.keys():
            node.attr.clear()
    for edge in A.edges():
        for k in edge.attr.keys():
            edge.attr.clear()

    mode = kwargs.pop('mode', 'sgd')
    maxiter = kwargs.pop('maxiter', 300)
    angle = kwargs.pop('angle', 0)
    overlap = kwargs.pop('overlap', True)

    A.graph_attr['mode'] = mode
    A.graph_attr['maxiter'] = maxiter
    A.graph_attr['normalize'] = angle
    A.graph_attr['overlap'] = overlap
    
    # perform initial node layout
    # A = pygraphviz_node_layout(G=G, prog=layout, mode=mode, maxiter=maxiter, angle=angle, overlap=overlap)
            
    A.graph_attr['size'] = f'{size[0]},{size[1]}'
    A.graph_attr['pad'] = f'{pad[0]},{pad[1]}'

    # the first attributes applied propagate to the rest of the nodes/edges
    A.node_attr.update(**node_attrs)
    A.edge_attr.update(**edge_attrs)
            
    if draw_labels:
        if ratio is None: ratio = 1.4
        for n in A.nodes():
            n.attr['shape'] = gef.default_key(kwargs, 'shape', 'oval')
            n.attr['style'] = gef.default_key(kwargs, 'style', 'filled')
    else:
        if ratio is None: ratio = 1.0
        for n in A.nodes():
            n.attr['style'] = gef.default_key(kwargs, 'style', 'filled')
            n.attr['shape'] = gef.default_key(kwargs, 'shape', 'circle')
            n.attr['label'] = gef.default_key(kwargs, 'label', ' ')

    # apply remaining graphviz attributes (overides already existing ones)
    A.graph_attr.update(**graph_attrs)

    if layout is not None:
        A.layout(layout)

        # hack for scale attribute not working in fdp and sfdp layouts
        # https://gitlab.com/graphviz/graphviz/-/issues/2129
        for n in A.nodes():
            new_pos = np.array(n.attr['pos'].split(','), dtype=float) * scale
            new_pos[0] *= np.sqrt(ratio)
            new_pos[1] /= np.sqrt(ratio)
            n.attr['pos'] = f'{new_pos[0]},{new_pos[1]}'
        for e in A.edges():
            # remove existing edge position to force them to be automatic
            e.attr.pop('pos')

        # manually force node attributes
        for n in A.nodes():
            if (_ := gef.default_key(kwargs, 'node_margin', None)) is not None:
                n.attr['margin'] = _
            if (_ := gef.default_key(kwargs, 'node_width', None)) is not None:
                n.attr['width'] = _
            if (_ := gef.default_key(kwargs, 'node_height', None)) is not None:
                n.attr['height'] = _

        byt = A.draw(format=format)

        if ipython_display:
            from IPython.display import Image, SVG
            if format == 'svg':
                disp = SVG(byt)
            elif format in ['png', 'bmp', 'jpg', 'jpeg']:
                disp = Image(byt)
            else:
                raise ValueError(f'unknown {format=}')
            out = disp
        else:
            out = byt
        # TODO add option to write to file

    if out_keys is not None:
        out_dict = {}
        for key in out_keys:
            out_dict[key] = locals().pop(key, None)
        out = out_dict
    
    return out 

def pygraphviz_node_layout(
    G=None, 
    A=None, 
    prog='neato', 
    mode='sgd', 
    maxiter=300, 
    angle=0,
    overlap=True
):
    """Extracts the node positions from a pygraphviz AGraph object
    in a networkx layout format.
    
    Note that pygraphviz only allows nodes to be strings. If the orignal
    networkx graph used non-string node identifiers (e.g. int) then the 
    corresponding nodes will be cast to strings in pygraphviz.
    
    Example
    ----------
    A = nx.drawing.nx_agraph.to_agraph(G)
    A.layout('neato')
    layout = get_pygraphviz_node_layout(A)
    nx.draw(G, pos=layout)
    """
    if A is None:
        A = nx.drawing.nx_agraph.to_agraph(G)
    
    # remove unnecessary metadata from DOT file
    for node in A.nodes():
        for k in node.attr.keys():
            node.attr.clear()
    for edge in A.edges():
        for k in edge.attr.keys():
            edge.attr.clear()

    A.graph_attr['mode'] = mode
    A.graph_attr['maxiter'] = maxiter
    A.graph_attr['normalize'] = angle
    A.graph_attr['overlap'] = overlap

    A.layout(prog)

    for e in A.edges():
        # remove existing edge position to force them to be automatic
        e.attr.clear()
    
    for n in A.nodes():
        for k in n.attr.keys():
            if k != 'pos':
                del n.attr[k]
    return A

def pygraphviz_extract_node_pos(A):
    layout = {}
    for node in A.nodes():
        pos_str = node.attr['pos']
        pos = [float(p) for p in pos_str.split(',')]
        layout[node.name] = pos
    return layout

def networkx_pygraphviz_draw(A, **kwargs):
    G = nx.drawing.nx_agraph.from_agraph(A)
    layout = get_pygraphviz_node_layout(A)
    return nx.draw(G, pos=layout)