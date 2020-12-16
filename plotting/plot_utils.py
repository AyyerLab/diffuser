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

def plot_cc(cc):
    P.clf()
    titles = 'xx yy zz xy yz zx'.split()
    for i in range(6):
        s = P.subplot(2,3,i+1)
        im = P.imshow(cc[i], vmin=-1, vmax=1, cmap='coolwarm')
        P.title(r'CC$_{%s}$'%titles[i], fontsize=12)
    P.suptitle('CC matrices', fontsize=14)
    P.subplots_adjust(right=0.88)
    cax = P.gcf().add_axes([0.9,0.125,0.02,0.75])
    P.colorbar(im, cax=cax)

def plot_ccpos(cc, mean_pos, ind):
    P.clf()
    titles = 'xx yy zz xy yz zx'.split()
    for i in range(6):
        s = P.subplot(2,3,i+1)
        im = P.scatter(mean_pos[:,0], mean_pos[:,1], 3, c=cc[i,ind], cmap='coolwarm', vmin=-1, vmax=1)
        P.plot(mean_pos[ind,0], mean_pos[ind,1], marker='o', markersize=14, color='green', markerfacecolor='none')
        if i > 2:
            P.xlabel(r'X ($\AA$)')
        if i % 3 == 0:
            P.ylabel(r'Y ($\AA$)')
        P.title(r'CC_${%s}$'%titles[i])
    P.subplots_adjust(right=0.88)
    cax = P.gcf().add_axes([0.9,0.125,0.02,0.75])
    P.colorbar(im, cax=cax)
