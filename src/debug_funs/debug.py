import sys
import ast
import inspect
import types
import builtins
import time
import functools
import importlib
import warnings
import datetime

import numpy as np
import scipy.interpolate
import networkx as nx

import graph_funs as grf
import general_funs as gef
import new_types

class Id:
    def __call__(self, x):
        return x
    
    def __repr__(self):
        return 'Id'

class Pair:
    def __call__(self, x, y):
        return x, y
    
    def __repr__(self):
        return 'Pair'

def is_hashable(obj):
    try:
        hash(obj)
        return True
    except TypeError:
        return False

def cache_last_result(init_cache=None):
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            result = f(*args, **kwargs)
            wrapped._cache = result
            return result
        wrapped._cache = init_cache
        return wrapped
    return decorator

def identity_decorator(f):
    new_f = functools.wraps(f)(f)
    return new_f

def add_attr(obj, **kwargs):
    for k,v in kwargs.items():
        setattr(obj, k, v)
    return obj

def add_func_attr(**kwargs):
    '''
    decorator for addint atributes to functions
    '''
    return lambda f: add_attr(f, **kwargs)

def nested_getattr(obj, attr):
    attrs = attr.split('.')
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

def attr_exists(obj, attr, nested=True):
    '''
    checks if (potentially nested) attribute exists in object
    
    e.g. attr_exists(np, 'fft.fft') == True
    '''
    try:
        if nested:
            nested_getattr(obj, attr)
        else:
            getattr(obj, attr)
        return True
    except AttributeError as e:
        return False

def get_external_vars(function):
    return function.__code__.co_names

def get_internal_vars(function):
    return function.__code__.co_varnames

def get_ast(function):
    ast_obj = ast.parse(inspect.getsource(function))
    return ast_obj

def ast_to_str(ast_obj):
    '''
    Returns the string representation of an ast object
    '''
    return ast.dump(ast_obj, indent=2)

def make_function(f, globals):
    # print(f.__code__, globals, f.__defaults__)
    new_f = types.FunctionType(f.__code__, globals=globals, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    new_f = functools.wraps(f)(new_f)
    return new_f

def add_globals(**kwargs):
    '''
    add some stuff into the decorated function's globals
    '''
    def decorator(f):
        new_globals = {**f.__globals__, **kwargs} 
        new_f = make_function(f, new_globals)
        return new_f
    return decorator

def allow_self(f):
    '''
    allow a decorated function to reference iteself with self
    '''
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        return f(f, *args, **kwargs)
    return wrapped

def no_globals(f):
    '''
    A function decorator that prevents functions from looking up variables in outer scope.
    '''
    # need builtins in globals otherwise can't import inside function
    new_globals = {'__builtins__': builtins} 
    new_f = make_function(f, new_globals)
    return new_f

def explicit_globals(**kwargs):
    '''
    A function decorator that prevents variable lookup to outer scope.
    Globals can be explicitly passed in using the globals argument.
    '''
    new_globals = {'__builtins__': builtins, **kwargs} 
    return lambda f: make_function(f, globals=new_globals)

def timer(f):
    @functools.wraps(f)
    def wrapped_function(*args, **kwargs):
        start_time = time.perf_counter_ns()
        result = f(*args, **kwargs)
        end_time = time.perf_counter_ns()
        print(f"\n{'-'*72}\nExecution took {(end_time-start_time)*1e-9} seconds\n{'-'*72}")
        return result
    try:
        wrapped_function.__signature__ = inspect.signature(f)
    except ValueError:
        pass
    return wrapped_function

def var_outer_name(var):
    '''
    Call inside function to get the name of var in outer scope.
    If no matching outer scope name is found look for match in local scope.

    Very dangerous function.
    '''
    outer_scope = inspect.stack()[2][0].f_locals
    out_names = []
    for name in outer_scope:
        if var is outer_scope[name]:
            out_names.append(name)
    if len(out_names) == 0:
        local_scope = inspect.stack()[1][0].f_locals
        for name in local_scope:
            if var is local_scope[name]:
                out_names.append(name)
    elif len(out_names) > 1:
        raise Exception(f'Multiple variable names matched the value {var}')
    return out_names[0]

def try_delattr(obj, attr, default=None):
    try:
        out = delattr(obj, attr)
    except AttributeError as e:
        out = default
    return out

def class_graph(cls, G=None):
    '''Creates a directed graph from a class and all of its base classes.
    object is ignored because every class derives from object and so it just 
    clutters up the graph.
    '''
    if G is None:
        G = nx.DiGraph()
    G.add_node(cls)
    for b in cls.__bases__:
        if b is not object:
            G.add_edge(b, cls)
            class_graph(b, G)
    return G

def module_graph(module, G=None, external_modules=False, include_root_module=False):
    '''Creates a directed graph from a module and all of its submodules.
    External submodules can be included in graph but they will not be traversed 
    since that could cause the graph to walk an absurd number of packages.

    The root module is typically not included since it tends to clutter the graph.
    '''
    def _module_graph_iter(module, G, external_modules):
        G.add_node(module)
        for name, obj in inspect.getmembers(module):
            obj_is_public = name[0] != '_'
            # print(name, obj_is_public and isinstance(obj, types.ModuleType))
            if obj_is_public and isinstance(obj, types.ModuleType):
                obj_is_submodule = module.__name__ in obj.__name__
                if external_modules or obj_is_submodule:
                    G.add_edge(module, obj)
                if obj_is_submodule:
                    _module_graph_iter(obj, G, external_modules)

    if G is None:
        G = nx.DiGraph()
    _module_graph_iter(module, G, external_modules)
    if not include_root_module:
        G.remove_node(module)
    return G

def module_graph3(module, G=None):
    '''Creates a directed graph from a module and all of its submodules

    This one tries to use the public API convention 
    '''
    if G is None:
        G = nx.DiGraph()
    G.add_node(module)
    if getattr(module, '__all__', None) is not None:
        for attr in module.__all__:
            obj = getattr(module, attr, None)
            if obj is None:
                warnings.warn(f'unable to resolve {attr} in {module.__name__}.__all__')
            if isinstance(obj, types.ModuleType):
                G.add_edge(module, obj)
                if module.__name__ in obj.__name__:
                    module_graph(obj, G)
    else:
        for name, obj in inspect.getmembers(module):
            if name[0] != '_' and isinstance(obj, types.ModuleType):
                print(type(obj), obj)
                G.add_edge(module, obj)
                if module.__name__ in obj.__name__:
                    module_graph(obj, G)
    return G

def class_graph_from_module_graph(module_graph, G=None):
    '''Creates a directed graph for all the classes and base classes in the 
    module graph.
    '''
    if G is None:
        G = nx.DiGraph(G)
    for module in module_graph.nodes:
        for name, obj in inspect.getmembers(module):
            if isinstance(obj, type):
                if obj not in G.nodes:
                    class_graph(obj, G)
    return G

def draw_class_graph(cls=None, G=None, scale=1.0, layout='dot', ratio=1.4):
    if G is None:
        G = class_graph(cls)
    out = grf.drawing.graphviz_draw(G, draw_orphans=True, layout=layout, scale=scale, ratio=ratio, shape='box', node_height=0.3, node_width=0, node_margin=0.0)
    return out

def ast_graph_from_source(src=None, ast_obj=None):
    '''Create a directed networkx graph for the ast generated from a given 
    string |src|.
    '''
    import ast
    import networkx as nx
    
    def iter_update(parent, G):
        '''Recursively traverse ast in BFS. Parent can be any ast.AST object.
        '''
        for child in ast.iter_child_nodes(parent):
            G.add_edge(parent, child)
            iter_update(child, G)

    if ast_object is None:
        ast_object = ast.parse(src)
    G = nx.DiGraph()
    iter_update(ast_object, G)
    
    return G

def time_split(sec, format='hms', to_string=True):
    if format == 'hms':
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        out = h,m,s
        if to_string:
            out = ''
            if h > 0:
                out += f'{h}:'
            if m > 0 or h > 0:
                out += f'{m:02d}:'
            out += f'{int(s):02d}'
    return out

def benchmark_adaptive(f, args=None, kwargs=None, time_budget=0.1, minruns=1, maxruns=None, progress=False, sort=False, summarize=True):
    '''Time function execution, repeating as many runs as can be allowed within 
    the time_budget, given in seconds.
    '''
    T0 = time.perf_counter()
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if maxruns is None:
        maxruns = np.inf
    dts = []
    i = 0
    dT = 0
    while (i < minruns) or (dT < time_budget and i < maxruns):
        t0 = time.perf_counter()
        soln = f(*args, **kwargs)
        t1 = time.perf_counter()

        i += 1
        dT = t1 - T0
        dt = t1 - t0
        dts.append(dt)
        if progress:
            time_frac = dT/time_budget
            time_frac = 1 if time_frac > 1 else time_frac
            minrun_frac = i/minruns
            maxrun_frac = i/maxruns
            frac = max(maxrun_frac, min(minrun_frac, time_frac))
            median_time = np.median(dts)
            inplace_print(f'progress: {frac*100:6.2f}% | {i:5d} runs | elapsed time: {dT:6.3f} sec | median time: {median_time:.3g} sec')
    if progress:
        # force print the last line in case it didn't fall into update_interval
        inplace_print(f'progress: {frac*100:6.2f}% | {i:5d} runs | elapsed time: {dT:6.3f} sec | median time: {median_time:.3g} sec', update_interval=0)
        # put cursor on new line so that the next print doesn't go on top of progress
        print() 
    dts = np.array(dts)
    if sort:
        dts = np.sort(dts)
    if summarize:
        summary = new_types.Namespace()
        summary.data = dts
        summary.runs = len(dts)
        summary.mean = np.mean(dts)
        summary.median = np.median(dts)
        summary.min = np.min(dts)
        summary.max = np.max(dts)
        return summary
    return dts

def benchmark_fixed(f, args=None, kwargs=None, Nruns=100, progress=False, sort=False, summarize=True):
    '''Time function execution, repeating N times
    '''
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    dts = np.zeros(Nruns, dtype=float)
    T0 = time.perf_counter()
    for i in range(Nruns):
        T1 = time.perf_counter()
        dT = T1-T0
        if progress:
            gef.inplace_print(f'progress: {(i+1)/Nruns*100:.2f}% | {i+1} runs | elapsed time: {dT:.3f} sec')
        t0 = time.perf_counter()
        soln = f(*args, **kwargs)
        t1 = time.perf_counter()
        dts[i] = t1-t0
    if sort:
        dts = np.sort(dts)
    if summarize:
        summary = new_types.Namespace()
        summary.data = dts
        summary.runs = len(dts)
        summary.mean = np.mean(dts)
        summary.median = np.median(dts)
        summary.min = np.min(dts)
        summary.max = np.max(dts)
        summary.quantile = lambda q: np.quantile(dts, q)
        return summary
    return dts

def time_complexity_benchmark(f=None, f_build=None, Ns=None, maxruns=None, single_run_budget=None, progress=False):
    '''Times a function of parameter N in a loop for various N.
    
    Often it is useful to decouple the setup and execution of a piece of code.
    In that case pass f_build which will take construct the function to be timed.
    '''
    if single_run_budget is None:
        single_run_budget = np.inf
    data = []
    for i, N in enumerate(Ns):  
        if f is None:
            t0 = time.perf_counter()
            runthis = f_build(N)
            t1 = time.perf_counter()
            build_time = t1-t0
        else:
            build_time = np.nan
            runthis = lambda: f(N)
        dt = dgf.debug.benchmark_adaptive(runthis, maxruns=maxruns, time_budget=single_run_budget)
        
        if progress:
            inplace_print(f'run {i:4d} | N {N:4d} | {build_time=:=.3g} | {dt.min=:=.3g}')
        
        dt.N = N
        dt.f = f
        dt.build_time = build_time
        data.append(dt)
        if dt.min > single_run_budget:
            break
    return data

def local_time_complexity(Ns, ts, w=None, s=1.0, Ns_out=None):
    '''Approximates the time complexity from benchmark data
    '''
    Ns_log = np.log(Ns)
    ts_log = np.log(ts)

    if Ns_out is None:
        Ns_out = Ns
        Ns_out_log = Ns_log
    else:
        Ns_out_log = np.log(Ns_out)
    
    # tck = scipy.interpolate.splrep(Ns_log, ts_log, s=s)
    # tck1 = scipy.interpolate.splder(tck)
    # t_t = scipy.interpolate.splev(Ns_out_log, tck)
    # O_t = scipy.interpolate.splev(Ns_out_log, tck1)

    irl = scipy.interpolate.UnivariateSpline(Ns_log, ts_log, s=s, w=w)
    der = irl.derivative(1)
    t_t = irl(Ns_out_log)
    O_t = der(Ns_out_log)
    
    return np.exp(t_t), O_t

@allow_self
@add_func_attr(_cache='', _t=0)
def inplace_print(self, *args, update_interval=1/30):
    t = time.time()
    if t - self._t < update_interval:
        return None
    # clear the previous line
    old_s = ' ' * len(self._cache) + '\r'
    new_s = ''
    for arg in args:
        new_s += str(arg) + ' '
    new_s = new_s[:-1]
    print(old_s+new_s, end='\r')
    self._cache = new_s
    self._t = t
    return None