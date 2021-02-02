import cupy as cp
import numpy as np
import calc_cc
import diffuser
import skopt
from skopt import load
import matplotlib.pylab as P
import h5py
import os
import os.path as op
import argparse

def main():
    parser = argparse.ArgumentParser(description=' 1. Diffuse scattering calculator with BGO optimized PCs (s vectors); 2. Estimate anisotropic CC between target and calcuted intensity; 3. Plot CC vs q')
    parser.add_argument('config_file', help='Path to config file')
    parser.add_argument('-mc', '--get_mc_intens', help='Calculate Icalc with BGO optimized s vectors.', default=False, action='store_true')
    parser.add_argument('-i', '--diff_intens', help='Path to Icalc diff_intens files.')
    parser.add_argument('-c', '--corr_q', help='Path to cc/radius/q file.')
    parser.add_argument('-o', '--out_fname', help='Path to output files for intensity and CC. Default: BGO_output_fname_mc_diffcalc.h5, BGO_output_fname_mc_diffcalc_CC.dat')
    parser.add_argument('-d', '--device', help='GPU device ID(if applicable).', type=int, default=0)
    
    args = parser.parse_args()

    cp.cuda.Device(args.device).use()
    
    opt = diffuser.bgo_optimize.CovarianceOptimizer(args.config_file)
        
    pcd = opt.pcd
    num_vecs = opt.num_vecs
   
    # get s vecs(cov_weights) for run_mc() from BGO optimization

    output_fname = opt.output_fname
    res = skopt.load(output_fname)
    s = res.x
    print('s' + str(s))
    obj_func = res.fun
    print('Obj_func' +  str(obj_func))

    ## get only the diagonals of cov_weights
    ''' sdiag =[]
    for n in range(1, num_vecs+1):
        j = int((n-1)* n/2 + (n-1))
        d = s[j]
        sdiag.append(d)
    sdiag = np.array(sdiag)
    #print(sdiag)
    '''
    # get Itarget
    Itarget = opt.i_target
   
    # CC calculator function
    def get_cc_q(Vol1, Vol2):
        '''CC between Itarget and Imc '''
        # get cc
        assert Vol1.shape == Vol2.shape
        mask = ~(np.isnan(Vol1) | np.isnan(Vol2))
        rbin = cp.array(opt.intrad)
        num_bins = int(rbin.max() + 1)
        qbin_size = float(opt.pcd.dgen.qrad[0,0,0] - opt.pcd.dgen.qrad[0,0,1])
         
        calc_cc.subtract_radavg(Vol1, rbin, num_bins, mask)
        calc_cc.subtract_radavg(Vol2, rbin, num_bins, mask) 

        cc = calc_cc.calc_cc(Vol1, Vol2, rbin, num_bins, mask)
        cc = cc.get()    
        # get q
        q = np.arange(num_bins) * qbin_size
        return(cc, q)

    # calculate diffuse scattering and CC between target intensity with option -mc
    if args.get_mc_intens: 
        # calculate diffuse scattering         
        Icalc = opt.get_mc_intens(s)
        #print('Writing Imc to', op.splitext(output_fname)[0]+'_mc_diffcalc.h5')
        
        ## saving calculated diffuse intensity
        if args.out_fname:
           print('Writing Imc to', args.out_fname+'_diffcalc.h5')
           with h5py.File(args.out_fname+'_diffcalc.h5', 'w') as f:
                 f['diff_intens'] = Icalc
        else:   
            print('Writing Imc to', op.splitext(output_fname)[0]+'_mc_diffcalc.h5')
            with h5py.File(op.splitext(output_fname)[0]+'_mc_diffcalc.h5', 'w') as f:    
                 f['diff_intens'] = Icalc
        
        ### CC calculation
        Vol1 = cp.array(Itarget) 
        Vol2 = cp.array(Icalc) 
        cc, q = get_cc_q(Vol1, Vol2)

        #### saving cc/q
        if args.out_fname:
           calc_cc.save_to_file(args.out_fname+'_CC.dat', cc, q)
        else:   
           calc_cc.save_to_file(op.splitext(output_fname)[0] +'_mc_diffcalc_CC.dat', cc, q)
   
    # calculate CC 
    elif args.diff_intens:
        # get calculated diffuse intensity from saved file usindg -i option
        with h5py.File(args.diff_intens, 'r') as f:
            Icalc = f['diff_intens'][:]
        ## CC calculation
        Vol1 = cp.array(Itarget)
        Vol2 = cp.array(Icalc)
        cc, q = get_cc_q(Vol1, Vol2)  
        fname = args.diff_intens 
        
        ### saving cc/q
        if args.out_fname:
            calc_cc.save_to_file(args.out_fname+'_CC.dat', cc, q)
        else:    
            calc_cc.save_to_file(op.splitext(fname)[0] +'_CC.dat', cc, q)
    
    else:
        # get cc and q from saved file using -c in command line 
        corr = np.loadtxt(args.corr_q, skiprows =1)
        q = corr[:, 1]
        cc = corr[:, 2]
        fname = args.corr_q
    
    # Plotting CC v/s q
    ax = P.axes(xlim=(0, q[-1]), ylim=(0, 1))
    P.plot(q, cc, 'r-')
    P.xlabel(r'1/d ($\mathrm{\AA}^{-1}$)')
    P.ylabel ('Anisotropic CC')
    P.title('simulation vs data')
    P.grid()
    P.show()



if __name__== '__main__':
    main()









