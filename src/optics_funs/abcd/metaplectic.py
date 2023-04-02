import numpy as np

from ..misc import metaplectic_sign_flip_matmul
from .. import abcd

def better_vectorize(pyfync, *args, **kwargs):
    '''
    np.vectorize has an intersting "feature" where if you pass in a non-array input to the vectorized function it will return the output of the non-vectorized function wrapped in a zero dimensional array instead of just returning the same output as the non-vectorized version.

    Example: 
        np.vectorize(f)(np.array([x,y,z])) == np.array([f(x),f(y),f(z)])
        np.vectorize(f)(x) == np.array(f(x))

    This function essentially fixes that so that instead:
        better_vectorize(f)(np.array([x,y,z])) == np.array([f(x),f(y),f(z)])
        better_vectorize(f)(x) == f(x)
    '''
    def wrap(*args2, **kwargs2):
        out = np.vectorize(pyfync, *args, **kwargs)(*args2, **kwargs2)
        if np.shape(out) == ():
            # unpack the zero dimensional array
            # alternative is out.item(0)
            # see https://stackoverflow.com/q/36392032/13161738
            out = out.flat[0]
        return out
    return wrap

def object_vectorize(f):
    def wrap(*args, **kwargs):
        return better_vectorize(f, otypes=[object])(*args, **kwargs)
    return wrap

class Metaplectic(object):
    def __init__(self, m=np.eye(2), s=1):
        self.m = m # abcd matrix
        self.s = s # metaplectic sign
    def __matmul__(self, other):
        s3 = (-1)**metaplectic_sign_flip_matmul(self.m, other.m) * self.s * other.s
        m3 = self.m@other.m
        return Metaplectic(m=m3, s=s3)
    def __repr__(self):
        repr_str = self.m.__repr__()
        if self.s == 1:
            repr_str = '(+)'+repr_str
        elif self.s == -1:
            repr_str = '(-)'+repr_str
        # pad all newlines to fix the alignment for the signs
        repr_str = repr_str.replace('\n','\n   ')
        return repr_str

    @classmethod
    def parity(cls):
        return cls(-np.eye(2))

    @classmethod
    @object_vectorize
    def space(cls, d):
        return cls(abcd.space(d))

    @classmethod
    @object_vectorize
    def lens(cls, f):
        return cls(abcd.lens(f))

    @classmethod
    @object_vectorize
    def lens_p(cls, p):
        return cls(abcd.lens_p(p))

    @classmethod
    @object_vectorize
    def mirror(cls, Rc):
        return cls(abcd.mirror(Rc))

    @classmethod
    @object_vectorize
    def frt(cls, r):
        return cls(abcd.frt(r))

    @classmethod
    @object_vectorize
    def scaling(cls, s):
        return cls(abcd.scaling(s))

    @classmethod
    @object_vectorize
    def cm(cls, q):
        return cls(abcd.cm(q))

    @classmethod
    @object_vectorize
    def cc(cls, z):
        return cls(abcd.cc(z))