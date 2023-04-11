import numpy as np
import sympy as sp

def operator(name,n,m):
    M = sp.Matrix(np.zeros([n,m]))
    for i in range(n):
        for j in range(m):
            M[i,j] = sp.Symbol(f'{name}_{{{i},{j}}}', complex=True)
    return M

class FunctionOperator(object):
    '''
    Should behave exactly as an operator (matrix of symbols), but whose elements are symbolic functions. Calling a FunctionOperator instance M with an argument x as M(x) will pass that argument to all of the matrix element functions.
    '''
    def __init__(self, name, n, m=None, arg=None):
        if m is None:
            m = n
        M = sp.Matrix(np.zeros([n,m]))
        for i in range(n):
            for j in range(m):
                M[i,j] = sp.Function(f'{name}_{{{i},{j}}}', complex=True)
                if arg is not None:
                    M[i,j] = M[i,j](arg)
        self.M = M
        self.shape = n,m
        self.name = name
        self.arg = arg
        
    def __call__(self, x):
        n,m = self.shape
        newM = FunctionOperator(self.name, *self.shape, arg=self.arg)
        for i in range(n):
            for j in range(m):
                newM.M[i,j] = newM.M[i,j](x)
        return newM
    
    def __getitem__(self, key):
        return self.M.__getitem__(key)
    
    def __matmul__(self, other):
        if isinstance(other, FunctionOperator):
            return self.M @ other.M
        else:
            return self.M @ other
        
    def __rmatmul__(self, other):
        if isinstance(other, FunctionOperator):
            return other.M @ self.M
        else:
            return other @ self.M
                        
    def __repr__(self):
        return repr(self.M)

def symbolic_DFT(N):
    n = np.arange(N)-sp.Number(N-1)/2
    arg = np.outer(n,n)/N
    sp_exp = np.vectorize(lambda x: sp.exp(-sp.I*2*sp.pi*x))
    M = sp.Matrix(sp_exp(arg)) / sp.sqrt(N)
    return M

def mapsubs(expr_list, subs_dict):
    return [expr.subs(subs_dict) for expr in expr_list]