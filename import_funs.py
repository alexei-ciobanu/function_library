import importlib
import sys
import types
from pathlib import Path

import IPython
from IPython.terminal.embed import InteractiveShellEmbed
import nbformat

def try_import(module_name):
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        print(f'Warning: {e}, skipping')

def rip_source_from_notebook(nb=None, nb_path=None):
    '''Rips all python source code from Jupyter notebook using nbformat into a 
    single continuous string, which could be exectuted (ignoring Jupyter 
    magicks).
    '''
    import nbformat
    
    if nb is None:
        nb = nbformat.read(nb_path, as_version=4)
        
    src = ''
    for i, cell in enumerate(nb['cells']):
        src += f'# Cell {i:d}\n'
        src += cell['source'] + '\n'
    return src

def write_nb_to_file(nb_path, out_path=None):
    '''Basically the same as "nbconvert --to script" but way way faster.
    Honestly don't know why nbconvert is so slow.
    '''
    from pathlib import Path
    import nbformat
    
    nb_path = Path(nb_path)
    nb = nbformat.read(nb_path, as_version=4)
    src = rip_source_from_notebook(nb)
    if out_path is None:
        out_path = nb_path.with_suffix('.py')
    with open(out_path, 'w') as f:
        f.write(src)

def module_from_file(file_path, module_name=None, exec_module=True, import_module=False):
    '''Converts a python file at |file_path| into a module and executes it
    in its own namespace on top of the current one. Useful if you want to run a 
    bunch of python scripts but to reuse the global imports (numpy, scipy, 
    matplotlib), which would otherwise be repeated every time.

    The variables in the python file can be dot-accessed in the |module| after 
    it has been executed.

    Last checked to work with:
    -------------------------
    python=3.9.7
    '''
    import importlib
    import sys
    from pathlib import Path
    
    path = Path(file_path)
    if module_name is None:
        module_name = path.stem
    
    spec = importlib.util.spec_from_file_location(module_name, path.resolve())
    module = importlib.util.module_from_spec(spec)
    
    if exec_module:
        # run the code in the module and populate the module namespace
        spec.loader.exec_module(module)
    
    if import_module:
        # not too sure why you would want to stick it in the global modules list
        # but you can
        sys.modules[module_name] = module
    
    return module

def module_from_string(src, module_name=None, exec_module=True, import_module=False):
    '''Takes a string of python code and executes it in its own nampespace, reusing
    global imports (e.g. numpy, scipy, matplotlib).

    Last checked to work with:
    -------------------------
    python=3.9.7
    '''
    import sys
    import types
    
    if module_name is None:
        module = types.ModuleType('<string>')
    else:
        module = types.ModuleType(module_name)
    
    if import_module:
        if module_name is None:
            raise NameError
        else:
            sys.modules[module_name] = module
    
    if exec_module:
        exec(src, module.__dict__)
    
    return module

class NotebookLoader(importlib.abc.Loader):
    """Module Loader class for Jupyter Notebooks

    Use module_from_notebook() to create modules using this loader class.
    
    Last checked to work with:
    -------------------------
    python=3.9.7
    ipython=7.28.0
    nbformat=5.1.3
    """
    def __init__(self, path=None):
        self.path = path
        
    def create_module(self, spec):
        return None
    
    def exec_module(self, module):
        import nbformat
        import IPython

        # load the notebook object
        with open(self.path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
            
        # create new IPython shell
        ipy = InteractiveShellEmbed()
        # magic commands in module will need the get_ipython() method
        module.__dict__['get_ipython'] = lambda: ipy
        module.__dict__['__file__'] = str(self.path.resolve())

        for cell in nb.cells:
            if cell.cell_type == 'code':
                # transform the magic commands to executable Python
                code = ipy.transform_cell(cell.source)
                # run the code in the module namespace
                exec(code, module.__dict__)

def module_from_notebook(nb_path, exec_module=True, import_module=False):
    import importlib
    import sys
    from pathlib import Path
    
    path = Path(nb_path)
    module_name = path.stem
    
    loader = NotebookLoader(path)
    spec = importlib.util.spec_from_file_location(module_name, path.resolve(), loader=loader)
    module = importlib.util.module_from_spec(spec)

    if exec_module:
        spec.loader.exec_module(module)
    
    if import_module:
        sys.modules[module_name] = module
    
    return module

# def module_from_notebook(nb=None, nb_path=None):
#     '''Takes the python code from Jupyter notebook and executes it in its own 
#     namespace.
#     '''
#     src = rip_source_from_notebook(nb, nb_path)
#     # could filter jupyter stuff here
#     module = module_from_string(src)
#     return module 
