import shutil
import os
import sys
import pathlib
from pathlib import Path

def tree(dir_path: Path, prefix: str=''):
    """
    taken from https://stackoverflow.com/a/49912639/13161738

    A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    
    Example
    ---------
    gen = tree(Path(r"C:\Users"))
    for line in take(20, gen):
        print(line)
    """    
    
    # prefix components:
    space =  '    '
    branch = '│   '
    # pointers:
    tee =    '├── '
    last =   '└── '
    
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir(): # extend the prefix and recurse:
            extension = branch if pointer == tee else space 
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix+extension)