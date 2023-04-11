import numpy as np
import scipy
import pylops
import sparse

import functools

def remove_singleton_dims(shape):
    out_shape = []
    for d in shape:
        if not isinstance(d, (int, np.integer)):
            out_shape.append(d)
    return tuple(out_shape)

def atleastiter(x):
    try:
        iter(x)
        out = x
    except TypeError as e:
        out = tuple([x])
    return out

def slice2range(slc, N=None):
    start, stop, step = slc.start, slc.stop, slc.step
    if stop is None:
        stop = N
    if start is None:
        start = 0
    if step is None:
        step = 1
    return range(start, stop, step)

def ravel(x):
    if isinstance(x, sparse._dok.DOK):
        out = ravel(x.to_coo())
    elif isinstance(x, (scipy.sparse._base.spmatrix, sparse._coo.COO)):
        out = x.reshape((1, np.product(x.shape)))
    elif isinstance(x, np.ndarray):
        out = x.ravel()
    elif isinstance(x, (tuple, list)):
        out = np.array(x).ravel()
    elif isinstance(x, (int, np.integer)):
        out = x
    return out

def index_shape(ind, shape):
    '''
    Get the resulting array shape after indexing an array of
    shape `shape` with `ind`.
    
    Example:
    arr = np.zeros([7,5,4])
    arr_shape = arr.shape
    ind = (slice(None), 1, (0, 1)) # equivelant to [:, 1, 0:2]
    i_shape = index_shape(ind, arr_shape)
    assert(i_shape == arr[ind].shape)
    '''
    s = []
    for i, x in enumerate(ind):
        if isinstance(x, slice):
            _ = slice2range(x, shape[i])
            s.append(_)
        elif isinstance(x, (range, list, tuple, np.ndarray)):
            s.append(x)
        elif isinstance(x, (int, np.integer)):
            pass
        else:
            raise TypeError(f'Unknown index object type {type(x)}')
    return tuple([len(x) for x in s])

class TensorIndexMap:
    '''
    Essentially a lookup table for an arbitrary ND array into its ravel'd index.
    Useful if you want to work in the ND index and then translate it to the ravel'd index.

    As far as I can tell it's not deprecated by pydata's `sparse` library, but it surely
    should have something like this in it.
    
    Does not support negative indexing.
    
    Example:
    arr = np.random.randn(4,3,2)
    stim = TensorIndMap(arr.shape)
    inds = stim[1:3, 0, :]
    print(arr.ravel()[inds])
    print(arr[1:3, 0, :].ravel())
    '''
    def __init__(self, shape):
        if isinstance(shape, (int, np.integer)):
            shape = tuple([shape])
        self._shape = tuple(shape)
        
    def __getitem__(self, args):
        args = atleastiter(args)
        if len(args) != len(self._shape):
            raise ValueError(f'dimension mismatch')
        new_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, slice):
                sl = slice2range(arg, self._shape[i])
            elif isinstance(arg, (int, np.integer)):
                sl = range(arg, arg+1)
            elif isinstance(arg, range):
                sl = arg
            elif isinstance(arg, (list, tuple)):
                sl = np.array(arg)
                if not issubclass(sl.dtype.type, np.integer):
                   raise ValueError(f'invalid index array {arg} only integer arrays are allowed')
            elif isinstance(arg, np.ndarray):
                sl = arg
                if not issubclass(sl.dtype.type, np.integer):
                   raise ValueError(f'invalid index array {arg} only integer arrays are allowed')
            else:
                raise ValueError(f'index object {arg} of type {type(arg)} is not valid for {self}')
            new_args.append(sl)
        slcs = tuple(new_args)
        # print(arg)
        # print(slcs)
        mg = np.array(np.meshgrid(*slcs, indexing='ij'))
        # print(mg.shape)
        mgl = mg.reshape(mg.shape[0], np.product(mg.shape[1:]))
        # print(len(mgl))
        # print(mgl)
        # print(self._shape)
        r = np.ravel_multi_index(mgl, self._shape)
        return r

    def __repr__(self):
        dim_str = f'{self._shape[0]}'
        for n in self._shape[1:]:
            dim_str += f'x{n}'
        repr_str = f"<{dim_str} sparse array index map>"
        return repr_str

class SparseTensor:
    '''
    This is now deprecated by pydata's `sparse` library which provides sparse tensors
    wrapped around scipy's sparse matrices.

    This is an attempt at hacking together a sparse tensor since scipy only provides sparse matrices.
    This is just a wrapper around a LIL or DOK sparse vector with some index trickery using `np.mgrid`
    and `np.ravel_multi_index` from SparseTensorIndexMap to simulate the indexing of a 
    sparse ND array.
    
    So far this is only for storage. No mathematical operations are supported.
    '''
    def __init__(self, shape, sparse_format='dok'):
        self._shape = shape
        self._N = int(np.product(shape))
        self._tim = TensorIndexMap(shape)
        
        if sparse_format == 'dok':
            self._sa = scipy.sparse.dok_array((1, self._N))
        elif sparse_format == 'lil':
            self._sa = scipy.sparse.lil_array((1, self._N))
        else:
            raise ValueError(f'unkown sparse format {sparse_format=}')

    def ravel(self):
        return ravel(self._sa)
        
    def __getitem__(self, ind):
        ind = atleastiter(ind)
        i_shape = index_shape(ind, self._shape)
        r = self._tim[ind]
        # print(f'{r=}')
        if i_shape:
            a = self._sa[[0]*len(r), r]
            # print(f'{a=}')
            i_ind = remove_singleton_dims(ind)
            b = SparseTensor(i_shape)
            # print(f'{b=}')
            b[...] = ravel(a)
            out = b
        else:
            a = self._sa[0, r[0]]
            out = a
        return out
    
    def __setitem__(self, ind, val):
        if ind is Ellipsis:
            self._sa[0, :] = ravel(val)
        else:
            r = self._tim[ind]
            self._sa[[0]*len(r), r] = ravel(val)
        
    def __repr__(self):
        dim_str = f'{self._shape[0]}'
        for n in self._shape[1:]:
            dim_str += f'x{n}'
        repr_str = f"<{dim_str} sparse array of type {self._sa.dtype.type}\n\t with {self._sa.nnz} stored elements in {self._sa.format} format>"
        return repr_str

def sum_operator(X=None, shape=None, axis=None):
    '''
    Should satisfy out_op@X.ravel() == np.sum(X, axis=axis).ravel()
    
    Should work in all cases np.sum does
    
    Example:
    X = np.random.randn(4,3,5)
    sum_op = sum_operator(X, axis=(2,1))
    print(sum_op@X.ravel())
    print(np.sum(X, axis=(2,1)).ravel())
    '''
    if shape is not None and X is not None:
        raise ValueError("Can't pass both an array and a shape")
    
    if shape is None:
        if X is None:
            raise ValueError('Must pass either an array or a shape')
        shape = X.shape
    
    if axis is None:
        axis = np.arange(X.shape)
    
    shape = list(shape) # make a mutable copy of shape
    axis = np.sort(np.atleast_1d(axis))
    m = len(axis)
    ops = []
    for i, n in enumerate(axis):
        # print(shape, n, i, n-i)
        op = pylops.basicoperators.Sum(tuple(shape), n-i)
        # print(op)
        ops.append(op)
        shape.pop(n-i)
    out_op = functools.reduce(lambda x,y: y@x, ops)
    return out_op

def select_operator(inds, N=None, M=None):
    '''Operator that selects specific elements from a vector (usually for comparison).
    '''
    if N is None:
        N = len(inds)
    if M is None:
        M = np.max(inds) + 1
    r_idx = np.arange(len(inds))
    c_idx = np.atleast_1d(inds)
    mat = scipy.sparse.coo_array((np.ones_like(inds), (r_idx, c_idx)), shape=(N, M))
    scipy_op = scipy.sparse.linalg.aslinearoperator(mat)
    pylops_op = pylops.LinearOperator(Op=scipy_op)
    return pylops_op

@functools.wraps(scipy.sparse.coo_array)
def pylops_from_coo(*args, **kwargs):
    mat = scipy.sparse.coo_array(*args, **kwargs)
    scipy_op = scipy.sparse.linalg.aslinearoperator(mat)
    pylops_op = pylops.LinearOperator(Op=scipy_op)
    return pylops_op