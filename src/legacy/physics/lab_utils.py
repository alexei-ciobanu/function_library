import numpy as np
import pickle as pl

def pyrpl_curves(files, cwd='./'):
    metas = []
    xs = []
    ys = []
    for filename in files:
        with open(cwd+filename,'rb') as f:
            dat = pl.load(f)
            idn, meta_dict, xy = dat
            metas.append(meta_dict)
            xs.append(xy[0])
            ys.append(xy[1])
    return metas,xs,ys