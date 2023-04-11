'''
To do real-time plotting you need to use QTimer or QApplication.processEvents()
Source: https://stackoverflow.com/a/18081255/16989297

Getting QApplication.processEvents() renders as fast as possible.

If you try to use pyqtgraph in jupyter without running |%gui qt| you are going
to have a bad time.

To view the interactive pyqtgraph examples

%gui qt
import pyqtgraph as pg
import pyqtgraph.examples
pg.examples.run()

or alternatively

python -m pyqtgraph.examples

The "remote plotting" is very useful for fast real time plotting
'''

import IPython
ipy = IPython.get_ipython()
if ipy is not None:
    ipy.run_line_magic('gui', 'qt')

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

r = (255,0,0)
g = (0,255,0)
b = (0,0,255)
k = (0,0,0)
w = (255,255,255)
color_cycle = [g,r,w,b]

def plot(x=None, y=None, *args, **kwargs):
    '''
    kwargs
    -----------
    pen : None or pg.QtCore.Qt.DashLine
    symbolSize: Int
    symbol: None, 'o', '+'
    '''
    win = pg.GraphicsWindow()
    win.setWindowTitle('Title')
    p = win.addPlot()

    pen = kwargs.pop('pen', color_cycle[0])
    symbolSize = kwargs.pop('symbolSize', 5)
    
    if x is None and y is None:
        raise Exception
    if x is None:
        x = np.arange(len(y))
    if y is None:
        y = x

    curve = p.plot(x, y, pen=pen, symbolSize=symbolSize, **kwargs)
    p.showGrid(x=True, y=True)

    # p.setLabel('left', "Y Axis", units='Arb.')
    # p.setLabel('bottom', "X Axis", units='Arb.')
    p.setLabel('left', "Y Axis")
    p.setLabel('bottom', "X Axis")

    out_dict = {}
    out_dict['window'] = win
    out_dict['plot'] = p
    out_dict['curve'] = curve

    return out_dict

def add_curve(plot, x=None, y=None):
    if y is None:
        if x is not None:
            x,y = y,x
    if x is None:
        x = np.arange(len(y))

    Ncurves = len(get_curves_from_plot(plot))
    curve = plot.plot(x, y, pen=color_cycle[Ncurves])

    out_dict = {}
    out_dict['curve'] = curve

    return out_dict

def set_curve_ydata(curve, y):
    xold, yold = curve.getData()
    curve.setData(xold, y)

def set_curve_data(curve, x, y):
    curve.setData(x, y) # puts draw event on queue
    pg.QtGui.QApplication.processEvents() # empty the queue

def get_items_from_win(win):
    return win.items()

def get_plots_from_win(win):
    return [x for x in win.items() if isinstance(x, pg.graphicsItems.PlotItem.PlotItem)]

def get_curves_from_plot(plot):
    return plot.curves