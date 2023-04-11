import plotly.graph_objects as go
import numpy as np

def plot(x=None, y=None, fig=None, name=None, width=800, height=500, xlabel=None, ylabel=None):
    if fig is None:
        fig = go.Figure()
    if y is None and x is not None:
        y = x
        x = None
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name))
    fig.update_layout(autosize=False, width=width, height=height,)
    if xlabel is not None:
        fig.update_xaxes(title=xlabel)
    if ylabel is not None:
        fig.update_yaxes(title=ylabel)
    return fig

def complex_plot(x=None, y=None, fig=None, name=None, width=800, height=500):
    if fig is None:
        fig = go.Figure()
    if y is None and x is not None:
        y = x
        x = None
    fig.add_trace(go.Scatter(x=x, y=np.real(y), mode='lines', name='real part'))
    fig.add_trace(go.Scatter(x=x, y=np.imag(y), mode='lines', name='imag part'))
    fig.update_layout(autosize=False, width=width, height=height,)
    return fig

def complex_plot3d(x=None, y=None, fig=None, name=None, width=700, height=600):
    if fig is None:
        fig = go.Figure()
    if y is None and x is not None:
        y = x
        x = None
    fig.add_trace(go.Scatter3d(x=x, y=np.imag(y), z=np.real(y), mode='lines', name=name))
    fig.update_layout(autosize=False, width=width, height=height,)
    return fig