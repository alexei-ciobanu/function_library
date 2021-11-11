import types
import debug_funs as dgf
import numpy as np

class attr_dict(types.SimpleNamespace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return self.__dict__.__repr__()
    
class Namespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return "{}({})".format(type(self).__name__, ", ".join(items))

class HashableNamespace(Namespace):
    '''
    Like types.SimpleNamespace but hashable so it can be used as a dict key
    '''
    def __init__(self, **kwargs):
        try:
            [self.__hash(x) for x in kwargs.values()]
        except TypeError as e:
            raise e
        self.__dict__.update(kwargs)

    @staticmethod
    def __hash(obj):
        if isinstance(obj, np.ndarray):
            return hash(obj.tobytes())
        else:
            return hash(obj)

    def __setattr__(self, name, value):
        try:
            self.__hash(value)
        except TypeError as e:
            raise e
        self.__dict__.update({name: value})
    
    def __hash__(self):
        hash_ = 0
        for x,y in self.__dict__.items():
            hash_ ^= hash(x) ^ self.__hash(y)
        return hash_

# make_namespace = namespace
def locals_to_namespace(locals):
    keys = tuple(locals.keys())
    ns = Namespace()
    for k in keys:
        ns.__dict__[k] = locals[k]
    return ns

def namespace_pull(names, namespace):
    out = []
    for name in names:
        out.append(namespace.__dict__[name])
    return out

def add_to_namespace(namespace, *args, **kwargs):
    names = list(map(dgf.var_outer_name, args))
    for name, arg in zip(names, args):
        namespace.__dict__[name] = arg
    namespace.__dict__.update(**kwargs)
    return namespace

# class attr_dict2(dict):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
            
#     def __setitem__(self, key, item):
#         exceptions = ['__class__']
#         if key in [x for x in dir(dict) if x not in exceptions]:
#             raise NameError(f'{key} is a protected name')
#         else:
#             super().__setitem__(key, item)
            
#     def __getattr__(self, attr):
#         return self[attr]
    
#     def __setattr__(self, attr, value):
#         self.__setitem__(attr, value)
    
#     def __dir__(self):
#         dict_dir = super().__dir__()
#         dict_keys = list(self.keys())
#         return dict_dir+dict_keys

class weak_value(object):
    def __init__(self, value=None):
        self._value = value
        
    def __getattr__(self, attr):
        try:
            return self._value.__getattr__(attr)
        except AttributeError as e:
            return self._value.__getattribute__(attr)
#     def __add__(self, other):
#         return self._value.__add__(other)
    def __dir__(self):
        return self._value.__dir__()
    def __repr__(self):
        return self._value.__repr__()