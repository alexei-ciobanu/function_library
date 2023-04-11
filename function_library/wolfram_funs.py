import numpy as np
import sympy as sp
import re

re_m_fun = lambda x: re.compile(rf"{x}\[(.*?)\]")
re_m_sqrt = re_m_fun('Sqrt')

def np2wolfram_array(arr):
    w_str = '{'
    lst_arr = arr.tolist()
    for row in lst_arr:
        w_str += str(row).replace('\n', '').replace('[','{').replace(']','}')+','
    w_str = w_str[:-1] + '}'
    return w_str

def wolfram2python(wolfram_expr):
    py_str = wolfram_expr.replace('^','**')
    py_str = re_m_sqrt.sub('np.sqrt(\\1)', py_str)
    return py_str

def wolfram2sympy(wolfram_expr):
    py_str = wolfram_expr.replace('^','**')
    py_str = re_m_sqrt.sub('sp.sqrt(\\1)', py_str)
    return py_str