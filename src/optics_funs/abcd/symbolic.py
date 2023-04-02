import sympy as sp

class SymbolFactory(object):
    def __init__(self, prefix='x', **kwargs):
        self.symbol_properties = kwargs
        self.prefix = prefix
        self.counter = 0
    
    def new_symbol(self):
        symbol_name = self.prefix + '_' + str(self.counter)
        self.counter += 1
        return sp.Symbol(symbol_name, **self.symbol_properties)

_space_symbol = SymbolFactory('@d', real=True)
_focal_length_symbol = SymbolFactory('@f', real=True)
_focal_power_symbol = SymbolFactory('@p', real=True)
_radius_of_curvature_symbol = SymbolFactory('@R_c', real=True)

def space(d=None):
    if d is None:
        d = _space_symbol.new_symbol()
    return sp.Matrix(\
                   [[1,d],\
                    [0,1]]\
                   )
                   
def mirror(Rc=None):
    if Rc is None:
        Rc = _radius_of_curvature_symbol.new_symbol()
    return sp.Matrix(\
                   [[1,0],\
                    [-2/Rc,1]]\
                   )
                   
def lens(f=None):
    if f is None:
        f = _focal_length_symbol.new_symbol()
    return sp.Matrix(\
                   [[1,0],\
                    [-1/f,1]]\
                   )
                   
def lens_p(p=None):
    if p is None:
        p = _focal_power_symbol.new_symbol()
    return sp.Matrix(\
                   [[1,0],\
                    [-p,1]]\
                   )
                   
def eig(M):
    A,B,C,D = M
    one_on_q_eig = (-(A-D) - sp.sp.sqrt((A-D)**2 + 4*B*C))/(2*B)
    return  1/(one_on_q_eig)
    
def q_propag(q, M):
    A,B,C,D = M
    return (A*q + B)/(C*q + D)