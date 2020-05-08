import cupy as cp
import numpy as np
import calc_cc
import BGO_optimize
import skopt
from skopt import load
import matplotlib.pylab as P
import h5py

def main():
    import argparse
    parser = argparse.ArgumentParser(description=' 1. Imc calculator with BGO optimized s vectors; 2. Estimate anisotropic CC between Itarget and Imc; 3. Plotting CC v/s q')
    parser.add_argument('-c','--config_file', help ='Config file')
    parser.add_argument('-mc', '--run_mc', help = 'Calculate Imc with BGO optimized  s vectors. Default = False', action ='store_true')
    parser.add_argument('-o', '--save',  help = 'Saving Imc and CC')
    parser.add_argument('Imc', help='Path to Imc file already exist')
    parser.add_argument('-i', '--input_file/input_files', help = 'Files for compare CC v/s q plot from different Imc dataset and for recalculation of CC') 
    parser.add_argument('-l', '--label', help='define label for plot. Default = out_fname.')
    
    parser.add_argument('-d', '--device', help = 'GPU device ID(if applicable). Default:0', type = int, default=0)
    
    args = parser.parse_args()

    cp.cuda.Device(args.device).use()

    
    opt = BGO_optimize.CovarianceOptimizer(args.config_file)
        
    pcd = opt.pcd
    num_vecs = opt.num_vecs
    #get S vecs(cov_weights) for run_mc() from BGO optimization
    res = skopt.load(opt.output_fname)
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
    else:
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
        
    # cal q
    res_edge = 1.35
    cen = size //2
    q = np.arange(num_bins) * bin_size / cen / res_edge
    
    #Save Imc, CC 
    with h5py.File(opt.output_fname+'_diffcalc_cc.h5', 'w') as f:    #need to edit  file name
         f['diff_intens'] = Imc
         f['cc'] = cc.get()
         f['q'] = q



    ##Plotting CC v/s q
    ax = P.axes(xlim=(0, 1.4), ylim=(0, 1.0))

    P.plot(q, cc, 'r-' )
    P.xlabel('q [1/$\AA$]')
    P.ylabel ('CC')
    P.title('CC v/s q')
    #P.axvline(q[10], ls='--', c = 'k')
    #P.axvline(q[100], ls='--', c = 'k')
    #P.axvline(q[200], ls ='--', c = 'k')

    P.grid()
    P.legend()
    P.show()
    #P.savefig('../CypA/Analysis/output_fname.png')



if __name__== '__main__':
    main()









# save to file

#calc_cc.save_to_file('../CypA/xtc/CC_md295_Imd_Ibgo_5vecs_diag1.dat', cc3, q)

#c10 = np.loadtxt('../CypA/xtc/CC_md295_Imd_Ibgo_10vecs_full_1_wosub.dat', skiprows =1)
