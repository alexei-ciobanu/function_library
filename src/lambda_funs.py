
reduce = lambda f,accum,list: accum if list[:] == [] else reduce(f, f(accum,list[0]), list[1:])
reduce2 = lambda f,list: reduce(f,list[0],list[1:])

map = lambda f,list: [] if list[:] == [] else [f(list[0])] + map(f,list[1:])