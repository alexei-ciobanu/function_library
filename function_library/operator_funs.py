import numpy as np
import pygraphviz as pgv
import anytree
import networkx as nx

import new_types as nt
import LCT_funs as lctf
import graph_funs as grf
import finesse3_funs as f3f

from IPython.display import Image
from anytree.exporter import UniqueDotExporter

def vec_slice(vec, N, i=0):
    inds = slice(N*i, N*(i+1))
    return vec[inds]

def opmat_matvec(Mop, v, dtype=np.complex128):
    '''Constructs an operator matrix matvec function. This implentation assumes
    that every operator matrix element is the same size.

    Mop : A dtype=object numpy array where each element is a single argument 
    operator
    v : A numeric vector that is used to determine the correct sizes for 
    arguments to operator matrix elements

    TODO
    * Make operator matrix work for matrix elements with different sizes
    similar to how the np.block function works. Will likely need force matrix
    elements to be scipy.sparse.linalg.LinearOperator, which have a size attributes
    that can be read.

    Example
    ----------
    import numpy as np

    N = 5
    r1,r2,r3,r4 = np.random.randn(4,N)

    Mop = np.zeros([2,2], dtype=object)
    Mop[0,0] = lambda x: r1*x
    Mop[0,1] = lambda x: r2*x
    Mop[1,0] = lambda x: r3*x
    Mop[1,1] = lambda x: r4*x
    v0 = np.zeros(2*N) # only need this to for vector size
    opmatvec = opmat_matvec(Mop, v0)
    v1 = np.ones(2*N) # make some vector

    # construct the matrix by hand
    R1,R2,R3,R4 = np.diag(r1), np.diag(r2), np.diag(r3), np.diag(r4)
    M = np.block([[R1,R2],[R3,R4]])

    # check that opmatvec gives the same result as matrix multiplication
    v2a = M@v1
    v2b = opmatvec(v1)
    print(v2a)
    print(v2a-v2b) # this should be a vector of zeros
    '''
    P,Q = np.shape(Mop)
    N = len(v)//Q
    out = np.zeros(N*P, dtype=dtype) # this should be allocated once and reused for all matvecs
    def matvec_apply(v):
        out[:] = 0 # clear state
        for i in range(P):
            for j in range(Q):
                inds_i = slice(N*i, N*(i+1))
                inds_j = slice(N*j, N*(j+1))
                res = Mop[i,j](v[inds_j])
                out[inds_i] += res
        return out
    return matvec_apply

def ifo_matvec(Mop, v, dtype=np.complex128, debug=False):
    matvec_apply = opmat_matvec(Mop, v, dtype=np.complex128)
    iter_count = [0]
    def ifo_matvec_apply(v):
        '''
        Identical to (I - M_rt) @ v
        '''
        iter_count[0] += 1
        if debug:
            gef.inplace_print(iter_count[0])
        out = v - matvec_apply(v)
        return out
    return ifo_matvec_apply

class Operator(object):
    def __init__(self, **kwargs):
        self.ni = kwargs.pop('ni', None)
        self.no = kwargs.pop('no', None)
        self.scalar = kwargs.pop('scalar', 1)
        self.ABCD_x = kwargs.pop('ABCD_x', np.eye(2))
        self.ABCD_y = kwargs.pop('ABCD_y', np.eye(2))
        self.separable = kwargs.pop('separable', True)
        self.composable = kwargs.pop('composable', True)
        self.map = kwargs.pop('map', None)
        # self.LCT_params = kwargs.pop('LCT_params', None)
        self.sim = kwargs.pop('sim', None)
        self.operators = [self]
        
    def kernel(self, LCT_params=None, direction='x'):
        if not self.separable:
            # TODO: much later
            raise NotImplementedError("probably don't want to build a nonseparable operator")
        if LCT_params is None:
            # external params not supplied, use internal ones
            LCT_params = self.LCT_params
        if LCT_params is None:
            raise ValueError('LCT_params is None, cannot build kernel')
        if direction == 'x':
            x1s = self.sim.node_dict[self.ni].LCT_params.xs
            x2s = self.sim.node_dict[self.no].LCT_params.xs
            D = lctf.LCT1D(x1s, x2s, M_abcd=self.ABCD_x)
        elif direction == 'y':
            y1s = self.sim.node_dict[self.ni].LCT_params.ys
            y2s = self.sim.node_dict[self.no].LCT_params.ys
            D = lctf.LCT1D(y1s, y2s, M_abcd=self.ABCD_y)
        else:
            raise ValueError(f'undefined {direction=}')
        return D

    @property
    def LCT_params(self):
        """Used in debugs to trigger planewave models
        """
        if self.sim is None:
            return None
        if self.sim._base_LCT_params is None:
            return None
        else:
            return True
        
    def __matmul__(self, other):
        if type(other) is Operator:
            if self.composable and other.composable:
                if not (other.no == self.ni):
                    raise ValueError(f"matmul failed because input/output nodes don't match")
                ns = nt.Namespace()
                ns.ni = other.ni
                ns.no = self.no
                ns.scalar = self.scalar * other.scalar
                ns.ABCD_x = self.ABCD_x @ other.ABCD_x
                ns.ABCD_y = self.ABCD_y @ other.ABCD_y
                ns.separable = self.separable and other.separable
                ns.composable = True
                ns.sim = self.sim
                # a,b = self.LCT_params, other.LCT_params
                # ns.LCT_params = self.merge_LCT_params(a,b)
                op = Operator(**ns.__dict__)
            else:
                # other is not composable, need to apply their kernels sequentially
                op = OperatorChain(self, other)
        elif isinstance(other, Operator):
            # trigger __ratmul__
            return NotImplemented
        else:
            # other is not an Operator, build the kernel
            scalar = f3f.map_eval(self.scalar)
            if self.separable:
                Dx = self.kernel(direction='x')
                Dy = self.kernel(direction='y')
                other = np.reshape(other, (Dy.shape[1], Dx.shape[1]))
                op = scalar * Dy @ other @ Dx.T
                op = op.ravel()
            else:
                if self.map is not None:
                    op = scalar * self.map * other
                else:
                    # TODO: much later
                    raise NotImplementedError   
        return op
    
    def __add__(self, other):
        """all inherited can use this
        """
        if isinstance(other, Operator):
            out = OperatorSum(self, other)
        else:
            raise ValueError
        return out

    def __radd__(self, other):
        print("hello there")

    def __call__(self, other):
        """all inherited can use this
        """
        return self @ other

    def build_tree(self, parent=None):
        self_node = anytree.Node("o", parent=parent)
        return self_node

    def draw_tree(self):
        """all inherited can use this
        """

        def node_dot_attr(node):
            if node.name == 's':
                out = "shape=triangle height=0.2 width=0.5"
            elif node.name == 'c':
                out = "shape=circle fixedsize=shape width=0.2"
            elif node.name == 'o':
                out = "shape=point width=0.1"
            return out

        root_node = self.build_tree()
        ude = UniqueDotExporter(root_node, nodeattrfunc=node_dot_attr)

        dot_str = ''
        for line in ude:
            dot_str += line + '\n'
            if "digraph tree {" in line:
                dot_str += "    node [label=\"\"]\n"

        im = Image(pgv.AGraph(dot_str).draw(format='png', prog='dot'))
        return im

    def build_graph(self, G, layer=0):
        """all inherited can use this
        """
        G.add_node(self, layer=layer)
        layer += 1
        if not getattr(self, 'operators', None) is None:
            for op in self.operators:
                G.add_edge(op, self, layer=layer)
                if isinstance(op, Operator):
                    op.build_graph(G, layer=layer)
        return G

    def draw_graph(self):
        """all inherited can use this
        """
        G = nx.MultiDiGraph()
        self.build_graph(G, layer=0)
        pos = nx.multipartite_layout(G, subset_key='layer', align='horizontal', scale=-1)
        nx.draw(G, pos, node_size=20)
        return G
    
class OperatorChain(Operator):
    """Collection of non-composable operators that need to be applied sequentially
    """
    def __init__(self, *args, **kwargs):
        self.operators = list(args)
        self.composable = False
        op_pairs = zip(self.operators, self.operators[1:])
        for op_a, op_b in op_pairs:
            if op_a.ni != op_b.no:
                raise ValueError("incompatible input/output nodes for OperatorChain")

    @property
    def LCT_params(self):
        if self.operators[0].LCT_params is None:
            return None
        else:
            return True

    @property
    def ni(self):
        return self.operators[-1].ni
    
    @property
    def no(self):
        return self.operators[0].no
    
    @property
    def scalar(self):
        scalar = 1.0
        for op in self.operators:
            scalar *= op.scalar
        return scalar
    
    @property
    def separable(self):
        sep = True
        for op in self.operators:
            sep = sep and op.separable
        return sep
    
    def kernel(self, LCT_params=None, direction='x'):
        if not self.separable:
            # TODO: much later
            raise NotImplementedError("probably don't want to build a nonseparable operator")
        D = self.operators[0].kernel(LCT_params, direction)
        for op in self.operators[1:]:
            B = op.kernel(LCT_params, direction)
            D = D@B
        return D
    
    def __matmul__(self, other):
        if isinstance(other, OperatorChain):
            a = self.operators[-1]
            b = other.operators[0]
            new_ops = self.operators[:-1] + (a@b).operators  + other.operators[1:]
            out = OperatorChain(*new_ops)
        elif isinstance(other, OperatorSum):
            new_ops = []
            for op in other.operators:
                new_ops.append(self @ op)
            out = OperatorSum(*new_ops)
        elif isinstance(other, Operator):
            a = self.operators[-1]
            b = other.operators[0]
            new_ops = self.operators[:-1] + (a@b).operators
            out = OperatorChain(*new_ops)
        else:
            out = other
            for op in self.operators[::-1]:
                out = op @ out
        return out

    def __rmatmul__(self, other):
        if isinstance(other, Operator):
            out = self.__class__.__matmul__(other, self)
        else:
            # shouldn't be reached
            raise ValueError

        return out
    
    def build_tree(self, parent=None):
        node_a = anytree.Node("c", parent=parent)
        root_node = node_a
        for op_b in self.operators[1:]:
            if isinstance(op_b, Operator):
                node_b = op_b.build_tree(parent=node_a)
            node_a = node_b
        return root_node
    
class OperatorSum(Operator):
    """Collection of OperatorSum/OperatorChain/Operator that need to be summed
    """
    def __init__(self, *args, **kwargs):
        self.operators = list(args)
        self.composable = False
        ni_s = [op.ni for op in self.operators]
        no_s = [op.no for op in self.operators]
        if not all(x==ni_s[0] for x in ni_s) or not all(x==no_s[0] for x in no_s):
            raise ValueError('incompatible input/output nodes for OperatorSum')
        # self.LCT_params = self.operators[0].LCT_params
        
    @property
    def LCT_params(self):
        if self.operators[0].LCT_params is None:
            return None
        else:
            return True

    @property
    def ni(self):
        return self.operators[0].ni
    
    @property
    def no(self):
        return self.operators[0].no
    
    @property
    def separable(self):
        sep = True
        for op in self.operators:
            sep = sep and op.separable
        return sep
    
    @property
    def scalar(self):
        """This should only be used in purely scalar/planewave models
        """
        if not self.LCT_params is None:
            raise ValueError('OperatorSum.scalar should not be used outside of purely scalar models')
        s = 0
        for op in self.operators:
            s += op.scalar
        return s

    def kernel(self, LCT_params=None, direction='x'):
        if not self.separable:
            # TODO: much later
            raise NotImplementedError("probably don't want to build a nonseparable operator")
        D = self.operators[0].kernel(LCT_params, direction)
        for op in self.operators[1:]:
            raise NotImplementedError
            # this doesn't do scalars properly
            B = op.scalar * op.kernel(LCT_params, direction)
            D += B
        return D
    
    def __matmul__(self, other):
        new_ops = []
        ops_a = self.operators
        if isinstance(other, OperatorSum):
            ops_b = other.operators
            for op_a in ops_a:
                for op_b in ops_b:
                    new_op = op_a @ op_b
                    new_ops.append(new_op)
            out = OperatorSum(*new_ops)
        elif isinstance(other, Operator):
            # expect either Operator or OperatorChain
            for op_a in ops_a:
                new_op = op_a @ other
                new_ops.append(new_op)
            out = OperatorSum(*new_ops)
        else:
            out = np.zeros_like(other, dtype=complex)
            for op in self.operators:
                out += op @ other
        return out

    def __rmatmul__(self, other):
        # only pure Operator is supported here
        if type(other) is Operator:
            out = self.__class__.__matmul__(other, self)
        else:
            # shouldn't be reached
            raise ValueError

        return out

    def __mul__(self, other):
        return self @ other

    def build_tree(self, parent=None):
        self_node = anytree.Node("s", parent=parent)
        if not getattr(self, 'operators', None) is None:
            for op in self.operators:
                if isinstance(op, Operator):
                    child_tree = op.build_tree(parent=self_node)
        return self_node