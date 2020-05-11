import cupy as cp
import numpy as np
import calc_cc
import BGO_optimize
import skopt
from skopt import load
import matplotlib.pylab as P
import h5py
import os
import os.path as op
import argparse

def main():
    parser = argparse.ArgumentParser(description=' 1. Imc calculator with BGO optimized s vectors; 2. Estimate anisotropic CC between Itarget and Imc; 3. Plotting CC v/s q')
    parser.add_argument('-c','--config_file', help ='Config file')
    parser.add_argument('-mc', '--run_mc', help = 'Calculate Imc with BGO optimized  s vectors. Default = False', action ='store_true')
    parser.add_argument('-Imc', '--Imc', help='Path to Imc file already exist. Default = False', action = 'store_true')
    parser.add_argument('-CC_q', '--CC_q', help ='Path to CC/radius/q file.')

   # parser.add_argument('-i', '--input_file/input_files', help = 'Files for compare CC v/s q plot from different Imc dataset and for recalculation of CC') 
    
    parser.add_argument('-d', '--device', help = 'GPU device ID(if applicable). Default:0', type = int, default=0)
    
    args = parser.parse_args()

    cp.cuda.Device(args.device).use()

    
    opt = BGO_optimize.CovarianceOptimizer(args.config_file)
        
    pcd = opt.pcd
    num_vecs = opt.num_vecs
    #get S vecs(cov_weights) for run_mc() from BGO optimization
    output_fname = opt.output_fname
    res = skopt.load(output_fname)
    s = np.array(res.x)

    # get only the diagonals of cov_weights
    ''' sdiag =[]
    for n in range(1, num_vecs+1):
        j = int((n-1)* n/2 + (n-1))
        d = s[j]
        sdiag.append(d)
    sdiag = np.array(sdiag)
    #print(sdiag)
    ''' 
    if args.run_mc: 
        Imc = opt.get_mc_intens(s)
        #Imc = opt.get_mc_intens(sdiag)
        
        with h5py.File(op.splitext(output_fname)[0]+'_diffcalc.h5', 'w') as f:    
             f['diff_intens'] = Imc

    elif args.Imc:
        #get Imc from file using -i option
        with h5py.File(args.Imc, 'r') as f:
            Imc = f['diff_intens'][:]

        #get Itarget
        Itarget = opt.Itarget
    
        ###CC between Itarget and Imc
        Vol1 = cp.array(Itarget)
        Vol2 = cp.array(Imc)

        assert Vol1.shape == Vol2.shape
        size = Vol1.shape[-1]
        bin_size = 1
        rad, rbin = calc_cc.calc_rad(size, bin_size)
        num_bins = int(rbin.max() +1)

        calc_cc.subtract_radavg(Vol1, rbin, num_bins)
        calc_cc.subtract_radavg(Vol2, rbin, num_bins) 

        cc = calc_cc.calc_cc(Vol1, Vol2, rbin, num_bins)
        cc = cc.get()    
        # cal q
        res_edge = 1.35
        cen = size //2
        q = np.arange(num_bins) * bin_size / cen / res_edge
        
        calc_cc.save_to_file(op.splitext(output_fname)[0] +'_diffcal_CC.dat', cc, q)

    else:
        Corr = np.loadtxt(args.CC_q, skiprows =1)
        q = Corr[:, 1]
        cc = Corr[:, 2]
    
    
    ##Plotting CC v/s q
    ax = P.axes(xlim=(0, 1.4), ylim=(0, 1.0))

    P.plot(q, cc, 'r-')
    P.xlabel('q [1/$\AA$]')
    P.ylabel ('CC')
   # P.title(str(op.splitext(output_fname)[0][24: ]))
    #P.axvline(q[10], ls='--', c = 'k')
    #P.axvline(q[100], ls='--', c = 'k')
    #P.axvline(q[200], ls ='--', c = 'k')

    P.grid()
    #P.legend()
    P.show()
#    P.savefig('../CypA/Analysis/' + op.splitext(output_fname)[0][24: ] + '.png')


if __name__== '__main__':
    main()









# save to file

#calc_cc.save_to_file('../CypA/xtc/CC_md295_Imd_Ibgo_5vecs_diag1.dat', cc3, q)

#c10 = np.loadtxt('../CypA/xtc/CC_md295_Imd_Ibgo_10vecs_full_1_wosub.dat', skiprows =1)
