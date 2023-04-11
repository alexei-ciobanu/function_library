import numpy as np
import pickle as pl
import pathlib

try:
    import bezier
    import colorspacious
except ModuleNotFoundError:
    pass

cwd = pathlib.Path(__file__).parent.absolute()

def _cmap_metadata_dict(cmap_name):
    out_dict = {
        'path': cwd / (cmap_name+'.pl'),
        'generate' : eval('generate_'+cmap_name)
    }
    return out_dict

def generate_bmr_constant():
    cmap_path = cmap_data['bmr_constant']['path']
    N = 501
    ns = np.linspace(0,1,N)

    ab_nodes = np.array([[-9,-35],[45,-40],[35.5,22]]).T
    ab_curve = bezier.Curve.from_nodes(ab_nodes)
    a,b = ab_curve.evaluate_multi(ns)
    l = np.ones(N)*50
    lab = np.vstack([l,a,b]).T
    rgb = colorspacious.cspace_convert(lab, "CAM02-UCS", "sRGB1")
    
    with open(cmap_path, 'wb') as f:
        pl.dump(rgb, f)
        print(f'written {cmap_path}')

def generate_bgr_constant():
    cmap_path = cmap_data['bgr_constant']['path']
    N = 501
    ns = np.linspace(0,1,N)

    ab_nodes = np.array([[-9,-35],[-13,-20],[-46,45],[0,20],[35.5,22]]).T
    ab_curve = bezier.Curve.from_nodes(ab_nodes)
    a,b = ab_curve.evaluate_multi(ns)
    l = np.ones(N)*50
    lab = np.vstack([l,a,b]).T
    rgb = colorspacious.cspace_convert(lab, "CAM02-UCS", "sRGB1")
    
    with open(cmap_path, 'wb') as f:
        pl.dump(rgb, f)
        print(f'written {cmap_path}')

cmap_names = ['bmr_constant', 'bgr_constant']
cmap_data = {name:_cmap_metadata_dict(name) for name in cmap_names}