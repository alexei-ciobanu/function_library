import matplotlib as mpl
import pickle as pl
import os
import types

from . import generate_colormaps
cmap_dict = generate_colormaps.cmap_data

# generate the maps if they don't exist
for cmap_name, cmap_data in cmap_dict.items():
    if not os.path.isfile(cmap_data['path']):
        cmap_data['generate']()

def _load_colormap(path, colormap_name=''):
    with open(path, 'rb') as f:
        rgb = pl.load(f)
        colormap = mpl.colors.ListedColormap(rgb, colormap_name)
    return colormap

cm_dict = {cmap_name: _load_colormap(cmap_dict[cmap_name]['path']) for cmap_name in cmap_dict.keys()}

cm = types.SimpleNamespace(**cm_dict)