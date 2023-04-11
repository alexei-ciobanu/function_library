import functools

@functools.total_ordering
class Mutant(object):
    '''
    A very scary class. From the outside indistinguishable from the currently held value.
    AFAIK it can only be detected by checking with type().  
    
    Don't even try using this as a key in a dictionary. Any mutation will likely 
    raise a KeyError because the mutated value was never added as a key.
    
    Truly a duck typing nightmare.
    
    This implementation uses slots, so should be fine to make a bazillion copies of this.
    
    Pretty much any interaction with it will morph it into a regular type.
    
    Useful for mimicking pointer style references. You can think of obj.mutate(x) as replacing
    the pointer inside obj to point at a new value x. At least that's why it was designed.
    '''
    __slots__ = 'value'
    
    def __init__(self, value):
        self.value = value
        
    def __getattribute__(self, attr):
        # __getattribute__ is very scary
        if attr in ['value', 'mutate']:
            # could use super() but this is more clear imo
            return object.__getattribute__(self, attr)
        else:
            return getattr(self.value, attr)
        
    def mutate(self, x):
        self.value = x
        
    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)

    @property
    def __class__(self):
        # this will fool isinstance()
        return self.value.__class__

    def __call__(self, *args, **kwargs):
        # in case you store a function in self.value
        return self.value(*args, **kwargs)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __complex__(self):
        return complex(self.value)
    
    # need __len__ and __iter__ for ipython jupyter for some reason __getattribute__ doesn't pull them correctly
    def __len__(self):
        return len(self.value)
        
    def __iter__(self):
        return iter(self.value)
    
    def __hash__(self):
        return hash(self.value)
    
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
    
    def __eq__(self, other):
        return self.value == other
    
    def __lt__(self, other):
        return self.value < other
    
    def __add__(self, other):
        return self.value + other
    
    def __sub__(self, other):
        return self.value - other
    
    def __mul__(self, other):
        return self.value * other
    
    def __div__(self, other):
        return self.value / other
    
    def __radd__(self, other):
        return other + self.value
    
    def __rsub__(self, other):
        return other - self.value
    
    def __rmul__(self, other):
        return other * self.value
    
    def __rdiv__(self, other):
        return other / self.value