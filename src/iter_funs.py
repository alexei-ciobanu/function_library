import itertools

def take(N, gen):
    '''
    Returns a finite generator containing the first N entries of gen.
    Identical to itertools.islice(gen, N)
    '''
    for g in zip(range(N), gen):       
        yield g[1]