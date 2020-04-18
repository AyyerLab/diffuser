import numpy as np
import pylab as P
from scipy import ndimage
from skopt import plots

def plot_partials(res, diag=True, d1=True, d2=True, ylim=None):
    if diag:
        num_vecs = int((np.sqrt(8*len(res.x)+1) - 1)/2.)
        diags = [i*(i+3)//2 for i in range(num_vecs)]
    else:
        num_vecs = len(res.x)
        diags = np.arange(len(res.x))
    if d1:
        for k in range(len(diags)):
            P.subplot(num_vecs, num_vecs, k*num_vecs + k + 1)
            P.cla()
            P.plot(*plots.partial_dependence(res.space, res.models[-1], diags[k], None, n_points=20))
            if ylim is not None:
                P.ylim(*ylim)
            ax = P.gca()
            ax.xaxis.set_visible(False)
            ax.yaxis.tick_right()
    if d2:
        if ylim is None:
            levels = 100
        else:
            levels = np.linspace(ylim[0], ylim[1], 100)
        for i in range(1,len(diags)):
            for j in range(i):
                P.subplot(num_vecs, num_vecs, i*num_vecs + j + 1)
                P.cla()
                P.contourf(*plots.partial_dependence(res.space, res.models[-1], diags[i], diags[j], n_points=10, n_samples=100), cmap='gist_earth_r', levels=levels)
                P.text(0.5, 0.9, '(%d, %d)'%(diags[j], diags[i]), horizontalalignment='center', transform=P.gca().transAxes, c='r')
                P.axis('off')
                print((diags[i],diags[j]))
    P.subplots_adjust(left=0.01, right=0.91, bottom=0.01, top=0.91)

def plot_conv(res):
    plots.plot_convergence(res)
    P.plot(np.arange(len(res.func_vals))+1, ndimage.gaussian_filter1d(res.func_vals, 4), c='C1')
    P.plot(np.arange(len(res.func_vals))+1, res.func_vals, ls='None', marker='+', c='C0')

def get_cov(res):
    num_vecs = int((np.sqrt(8*len(res.x)+1) - 1)/2.)
    cov = np.zeros((num_vecs,num_vecs))
    n = 0
    for i in range(num_vecs):
        for j in range(i+1):
            if i == j:
                cov[i,j] = res.x[n]**2
            else:
                cov[i,j] = res.x[n]
                cov[j,i] = res.x[n]
            n += 1
    return cov
    
