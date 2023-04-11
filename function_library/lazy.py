'''
lazy evaluation module
'''
import numpy as np

list_eager = list
getattr_eager = getattr

class lazy(object):
    '''
    A lazy wrapper around python's function calling syntax.
    Functions can be passed directly or as a string which is then evaluated
    to the function. Passing function as a string has the benefit of it being 
    displayed in the __repr__.
    
    args and kwargs can be passed directly or wrapped up in a 
    lazy object themselves (either with the identity or any other function).
    
    Evaluating the lazy object will force it to evaluate any lazy objects embedded 
    in args or kwargs. Embedding lazy objects inside self.fun is not yet supported
    (I didn't need it at the time)
    '''
    def __init__(self, fun, *args, fun_str=None, **kwargs):
        if isinstance(fun, str):
            self._fun = eval(fun)
            self._fun_str = fun
        else:
            self._fun = fun
            self._fun_str = str(fun)
            
        if fun_str is not None: 
            self._fun_str = fun_str
        
        self._args = list_eager(args)
        self._kwargs = kwargs
        
    @property
    def fun(self):
        return self._fun
        
    @fun.setter
    def fun(self, f):
        if isinstance(f, str):
            self._fun_str = f
            self._fun = eval(f)
        else:
            self._fun = f
            self._fun_str = str(f)
        
    def __call__(self, *args, **kwargs):
        '''note that the function reference may be mutated'''
        return lazy(self._fun, *args, fun_str=self._fun_str, **kwargs)
    
    def eval(self, debug=False, recurse=3):
        fun = self._fun
        args = []
        kwargs = {}
        if debug:
            print(self._fun_str, self._args, self._kwargs)
        if hasattr(fun, 'eval'):
            fun = fun.eval(debug=debug, recurse=recurse)
        for arg in self._args:
            if hasattr(arg, 'eval'):
                args.append(arg.eval(debug=debug, recurse=recurse))
            else:
                args.append(arg)
        for key in self._kwargs:
            if hasattr(self._kwargs[key], 'eval'):
                kwargs[key] = self._kwargs[key].eval(debug=debug, recurse=recurse)
            else:
                kwargs[key] = self._kwargs[key]

        out = fun(*args, **kwargs)
        if recurse:
            if hasattr(out, 'eval'):
                if debug:
                    print(f'recursing {recurse} on {self._fun_str}')
                out = out.eval(debug=debug, recurse=recurse-1)
        if debug:
            print(f'returning {out}')
        return out
    
    def __repr__(self):
        '''__repr__ should be executable when possible'''
        fun_str = self._fun_str
            
        if len(self._args) > 0:      
            arg_str = ', *' + str(tuple(self._args))
        else:
            arg_str = ''
            
        if len(self._kwargs) > 0:      
            kwarg_str = ', **' + str(self._kwargs)
        else:
            kwarg_str = ''
        return f'lazy({fun_str}{arg_str}{kwarg_str})'
    
    def tree(self, depth=0):
        '''returns a string which presents the lazy expression as a tree'''
        tree_str = ''
        indent_size = 4
      
        fun_str = '\n'+' '*depth*indent_size+'  - fun='
        if hasattr(self._fun, 'tree'):
            fun_str += self._fun.tree(depth=depth+1)
        else:
            if self._fun_str is not None:
                fun_str += self._fun_str
            else:
                fun_str += str(self._fun)
            
        arg_len = len(self._args)
        if len(self._args) > 0:   
            arg_str = '\n'+' '*depth*indent_size+'  - args=['
            for i, item in enumerate(self._args):
                if hasattr(item, 'tree'):
                    arg_str += item.tree(depth=depth+1)
                else:
                    arg_str += str(item)
                if i < arg_len-1:
                    arg_str += ', '
            arg_str += ']'
        else:
            arg_str = ''
            
        kwarg_len = len(self._kwargs)
        if kwarg_len  > 0:
            kwarg_str = '\n'+' '*depth*indent_size+'  - kwargs={'
            for i, key in enumerate(self._kwargs):
                kwarg_str += f"'{key}': "
                if hasattr(self._kwargs[key], 'tree'):
                    kwarg_str += self._kwargs[key].tree(depth=depth+1)
                else:
                    kwarg_str += str(self._kwargs[key])
                if i < kwarg_len-1:
                    kwarg_str += ', '
            kwarg_str += '}'
        else:
            kwarg_str = ''
            
        
        tree_str += f'lazy({fun_str}{arg_str}{kwarg_str}'
        tree_str += '\n'+' '*depth*indent_size+')'
        return tree_str
        
eye = lazy('lambda x: x')
add = lazy('lambda x,y: x+y')
sub = lazy('lambda x,y: x-y')
mul = lazy('lambda x,y: x*y')
div = lazy('lambda x,y: x/y')
pow = lazy('lambda x,y: x**y')
sqrt = lazy('np.sqrt')
sum = lazy('np.sum')
exp = lazy('np.exp')
log = lazy('np.log')
sin = lazy('np.sin')
cos = lazy('np.cos')
tan = lazy('np.tan')
angle = lazy('np.angle')
array = lazy('np.array')

getattr = lazy('lambda obj, attr: getattr_eager(obj, attr)')
list = lazy('lambda *args: list_eager(args)')

getkey = lazy('lambda obj, key: obj[key]')