import cupy as cp
import numpy as np
import calc_cc
import plot_multi_cc_q
import BGO_optimize
import skopt
from skopt import load
import matplotlib.pylab as P
from matplotlib.pyplot import cm
import h5py
import os
import os.path as op
import argparse

def main():
    parser = argparse.ArgumentParser(description=' 1. Imc calculator with BGO optimized s vectors; 2. Estimate anisotropic CC between Itarget and Imc; 3. Plotting CC v/s q')
#    parser.add_argument('config_file', help ='Path to config file')
#    parser.add_argument('-mc', '--run_mc', help = 'Calculate Imc with BGO optimized  s vectors. Default = False', action ='store_true')
#    parser.add_argument('-i', '--diff_intens', help='Path to Imc diff_intens files.')
#    parser.add_argument('-c', '--corr_q', help ='Path to cc/radius/q file.')
#    parser.add_argument('-o', '--out_fname', help ='Path to output files.')
#    parser.add_argument('-d', '--device', help = 'GPU device ID(if applicable). Default:0', type = int, default=0)
    parser.add_argument('multi_plot', nargs = '+', help = 'Path to CC_q files') 
    
    args = parser.parse_args()

    '''def multi_cc_q(*args, **kwargs):
        ax = kwargs.get("ax", None)
        q = kwargs.get("q", None)
        cc = kwargs.get("cc", None)

        if ax is None:
            ax = P.gca()

        ax.set_title("CC v/s q")
        ax.set_xlabel("q [$\AA$]")
        ax.set_ylabel("CC")
        ax.grid()

        if cc is not None:
            ax.set_cc(cc)

        colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

        for fname, color in zip(args, colors):
            if isinstance(fname, tuple):
                name, fname = fname
            else:
                name = None

            corr = np.loadtxt(fname, skiprows =1)
            q = corr[:, 1]
            cc = corr[:, 2]
            #P.axes(xlim =(0, 1.4), ylim =(0, 1.0))
            ax.plot(q, cc, c=color, lw=2, label = name)
            P.legend()
            P.show()
        return ax    
      '''  

  
    plot_multi_cc_q.multi_cc_q(args.multi_plot)

if __name__== '__main__':
    main()

