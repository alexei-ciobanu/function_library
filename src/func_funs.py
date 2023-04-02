'''
All functions here are pure and are expecting standard types.
'''

import numpy as np
import networkx as nx
import functools


def compose(functions, order='application'):
    '''
    Takes a list of functions [f1,f2,f3,...,fn] and returns a function fn(...(f3(f2(f1(x)))))
    
    By default assumes the list of functions is given in application order.
    '''
    if order == 'composition':
        functions = functions[::-1]
    elif order == 'application':
        pass
    
    composed_function = functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)
    return composed_function

def list_comprehension(f, iterable, test):
    return (f(x) for x in iterable if test(x))

def recursive_list_comprehension(f, iterable, test):
    if not isinstance(iterable, list):
        return f(iterable)
    else:
        return list(list_comprehension(lambda y: recursive_list_comprehension(f, y, test), iterable, test))


def recursive_map(f, x):
    '''
    same as map() except this one traverses the sub lists, applying f() to all
    non-list elements.

    l = [1,[2,3],[4,5,6]]
    list(map(lambda x: [x] + [1], l)) == [[1,1], [[2,3], 1], [[4,5,6], 1]]
    recursive_map(lambda x: [x] + [1], l) == [[1,1], [[2,1], [3,1]], [[4,1], [5,1], [6,1]]]
    '''
    return [f(y) if not isinstance(y, (tuple, list)) else recursive_map(f, y) for y in x]


def recursive_list_map(f, x, debug=False):
    '''
    same as recursive_map() but f acts on lists instead of list elements.

    l = [[6,7,8,9],[3,4,5,6,2]]
    recursive_list_map(lambda x: np.sum(x), l) == [30,20]
    recursive_map(lambda x: np.sum(x), l) == [[6,7,8,9],[3,4,5,6,2]]
    '''
    if x == []:
        if debug:
            print('x == []')
        return []
    elif type(x[0]) is list:
        if debug:
            print('x is a list of lists, recursing', x)
        return [recursive_list_map(f, y, debug=debug) for y in x]
    else:
        if debug:
            print('x is just a list', x)
        return f(x)


def recursive_reduce(f, x, init=None, debug=False):
    if debug:
        print('x = ', x)
    if init is None:
        if x == []:
            if debug:
                print('reduce base case x == []')
            return []
        elif x[1:] == []:
            if debug:
                print('reduce base case x[1:] == []')
            return f(x[0])
        elif x[2:] == []:
            if debug:
                print('reduce base case x[2:] == []')
            return f(x[0], x[1])
        else:
            if debug:
                print('reduce recurse case (no init)')
            return f(x[0], recursive_reduce(f, x[1:]))
    else:
        if x[1:] == []:
            if debug:
                print('reduce base case x[1:] == []')
            return f(init, x[0])
        else:
            if debug:
                print(f'reduce recurse case (init = {init})')
            return f(init, recursive_reduce(f, x))


def list_roll(l, k=1, inplace=False):
    '''rolls list forward by k'''
    l2 = l.copy()
    N = len(l2)

    if inplace:
        for i in range(len(l2)):
            l[(i-k)%N] = l2[i]
        return l
    else:
        for _ in range(k):
            l2.append(l2.pop(0))
        return l2

def list_flatten(l):
    '''flattens list'''
    if type(l) is not list:
        return [l]
    elif l == []:
        return l
    else:
        if type(l[0]) is list:
            return list_flatten(l[0]) + list_flatten(l[1:])
        else:
            return l


def list_groupby(f, l):
    '''
    Groups items in a list by the return vaule of f().
    Returns a dict, so the return value of f() needs to be hashable.

    I think the group order should be robust enough that it can be
    relied upon.
    '''
    d = dict()
    for item in l:
        d[f(item)] = []
    for item in l:
        d[f(item)].append(item)
    return d


def list_denest(l):
    '''
    removes one level from a nested list
    '''
    return recursive_reduce(lambda x, y: x+y, l)


def ordered_intersect(A, B):
    # assumes paths A,B have no duplicates
    if len(A) < len(B):
        A, B = B, A
    t = []
    for j, a in enumerate(A):
        for i, b in enumerate(B):
            if a is b:
                tt = []
                for k in range(0, len(B)):
                    if B[(i-k) % len(B)] is A[(j-k) % len(A)]:
                        tt = [A[(j-k) % len(A)]] + tt
                    else:
                        break
                for k in range(1, len(B)):
                    if B[(i+k) % len(B)] is A[(j+k) % len(A)]:
                        tt = tt + [A[(j+k) % len(A)]]
                    else:
                        break
                t.append(tt)
                break
    return t


def ordered_difference(A, B):
    # assumes paths A,B have no duplicates
    if len(A) < len(B):
        A, B = B, A
    t = []
    for j, a in enumerate(A):
        for i, b in enumerate(B):
            if a is b:
                del_ind = []
                for k in range(0, len(B)):
                    if B[(i-k) % len(B)] is A[(j-k) % len(A)]:
                        del_ind.append((j-k) % len(A))
                    else:
                        break
                for k in range(1, len(B)):
                    if B[(i+k) % len(B)] is A[(j+k) % len(A)]:
                        del_ind.append((j+k) % len(A))
                    else:
                        break
                t.append([A[i] for i in range(len(A)) if i not in del_ind])
#                 print(del_ind)
                break
    return t


def get_edge_data(G, u, v, data=True):
    if data is True:
        # return all dict members
        return G.edges[u, v]
    else:
        # return the "data" member
        return G.edges[u, v][data]


def is_list_unique(l):
    seen = []
    for node in l:
        if node not in seen:
            seen.append(node)
        else:
            return False
    return True


def only_unique_elements(l):
    seen = []
    for node in l:
        if node not in seen:
            seen.append(node)
    return seen


def list_contains_list(A, B):
    if len(A) < len(B):
        return False
    else:
        o_isec = ordered_intersect(A, B)
        if o_isec == []:
            return False
        m = max(o_isec, key=len)
        if len(m) == len(B):
            return True
        else:
            return False
