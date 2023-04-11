from multimethod import multimethod

@multimethod
def add(a, b):
    return a + b

@multimethod
def sub(a, b):
    return a - b

@multimethod
def mul(a, b):
    return a * b

@multimethod
def div(a, b):
    return a / b

@multimethod
def matmul(a, b):
    return a @ b

@multimethod
def pow(a, b):
    return a ** b

class MultiDispatchObject:
    '''A convinience class to register multimethods for basic arithmetic operations.
    This makes it easier to hook into these methods and explicitly write out the 
    logic for how these arithmetic operators work for different combinations of 
    user defined classes. This is an alternative to python's ad hoc solution 
    to pass around NotImplemented objects to trigger the __r<op>__ method in 
    the caller's class, which can get confusing when there are nontrivial 
    interactions between several different classes using these methods.

    Only the bultin binary (2-arity) operators should be listed here. Bultin 
    unary operators are better off implemented the standard way as builtin methods 
    in class definition since python's OOP is perfectly legible for single dispatch.

    If using multiple inheritance it's probably a good idea to have this as the 
    first inherited class (last in the multiple iheritance list) to allow other 
    classes to override the following methods.

    Example
    -----------
    class Foo(Bar, Baz, MultiDispatchObject):
        ...

    '''
    def __add__(self, other):
        return add(self, other)
    
    def __radd__(self, other):
        return add(other, self)
    
    def __sub__(self, other):
        return sub(self, other)
    
    def __rsub__(self, other):
        return sub(other, self)
    
    def __mul__(self, other):
        return mul(self, other)
    
    def __rmul__(self, other):
        return mul(other, self)
    
    def __truediv__(self, other):
        return div(self, other)
    
    def __rtruediv__(self, other):
        return div(other, self)
    
    def __matmul__(self, other):
        return matmul(self, other)
    
    def __rmatmul__(self, other):
        return matmul(other, self)
    
    def __pow__(self, other):
        return pow(self, other)
    
    def __rpow__(self, other):
        return pow(other, self)

################################################################################
# The following block is needed to break infinite recursions when inheriting 
# MultiDispatchObject without implementing the neccessary multiple dispatch methods
################################################################################

@multimethod
def add(a:MultiDispatchObject, b):
    raise NotImplementedError
    
@multimethod
def add(a, b:MultiDispatchObject):
    raise NotImplementedError
    
@multimethod
def add(a:MultiDispatchObject, b:MultiDispatchObject):
    raise NotImplementedError
    
@multimethod
def sub(a:MultiDispatchObject, b):
    raise NotImplementedError
    
@multimethod
def sub(a, b:MultiDispatchObject):
    raise NotImplementedError
    
@multimethod
def sub(a:MultiDispatchObject, b:MultiDispatchObject):
    raise NotImplementedError
    
@multimethod
def mul(a:MultiDispatchObject, b):
    raise NotImplementedError
    
@multimethod
def mul(a, b:MultiDispatchObject):
    raise NotImplementedError
    
@multimethod
def mul(a:MultiDispatchObject, b:MultiDispatchObject):
    raise NotImplementedError
    
@multimethod
def div(a:MultiDispatchObject, b):
    raise NotImplementedError
    
@multimethod
def div(a, b:MultiDispatchObject):
    raise NotImplementedError
    
@multimethod
def div(a:MultiDispatchObject, b:MultiDispatchObject):
    raise NotImplementedError
    
@multimethod
def matmul(a:MultiDispatchObject, b):
    raise NotImplementedError
    
@multimethod
def matmul(a, b:MultiDispatchObject):
    raise NotImplementedError
    
@multimethod
def matmul(a:MultiDispatchObject, b:MultiDispatchObject):
    raise NotImplementedError
    
@multimethod
def pow(a:MultiDispatchObject, b):
    raise NotImplementedError
    
@multimethod
def pow(a, b:MultiDispatchObject):
    raise NotImplementedError
    
@multimethod
def pow(a:MultiDispatchObject, b:MultiDispatchObject):
    raise NotImplementedError

#########################################
# Short exapmle implementation
#########################################

class Example(MultiDispatchObject):
    '''Simple container class
    '''
    def __init__(self, val=1):
        self.val = val
    
@multimethod
def add(a:Example, b):
    return a.val + b

@multimethod
def add(a, b:Example):
    return a + b.val

@multimethod
def add(a:Example, b:Example):
    return a.val + b.val