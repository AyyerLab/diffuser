import os.path as op
import argparse

import numpy as np
import pylab as P
import h5py
import cupy as cp
import skopt

import diffuser
from diffuser import calc_cc

def get_mc_icalc(opt, out_prefix):
    '''Calculate and save diffuse intensity from BGO-optimized weights'''
    res = skopt.load(opt.output_fname)
    print('s =', res.x)
    print('Objective function =', res.fun)

    # Calculate diffuse scattering
    i_calc = opt.get_mc_intens(res.x)

    ## Save calculated diffuse intensity
    if out_prefix is not None:
        h5_output_fname = out_prefix + '_diffcalc.h5'
    else:
        h5_output_fname = op.splitext(opt.output_fname)[0] + '_diffcalc.h5'
    print('Writing Imc to', h5_output_fname)
    with h5py.File(h5_output_fname, 'w') as fptr:
        fptr['diff_intens'] = i_calc
    return i_calc

def get_cc_q(vol1, vol2, opt):
    '''q-dependent CC between two diffuse intensities'''
    assert vol1.shape == vol2.shape
    mask = ~(np.isnan(vol1) | np.isnan(vol2))
    rbin = cp.array(opt.intrad)
    num_bins = int(rbin.max() + 1)
    qbin_size = float(opt.pcd.dgen.qrad[0,0,0] - opt.pcd.dgen.qrad[0,0,1])

    calc_cc.subtract_radavg(vol1, rbin, num_bins, mask)
    calc_cc.subtract_radavg(vol2, rbin, num_bins, mask)

    cc = calc_cc.calc_cc(vol1, vol2, rbin, num_bins, mask)
    if calc_cc.CUPY:
        cc = cc.get() # pylint: disable=no-member
    q = np.arange(num_bins) * qbin_size

    return(cc, q)

def main():
    '''CLI script to plot CC vs q for a few different data sources'''
    parser = argparse.ArgumentParser(description='Plot CC vs q.\n'
                                     'One of -m, -i or -c are used to specify the data source',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config_file', help='Path to config file')
    parser.add_argument('-m', '--get_mc_intens', help='Calculate Icalc with BGO optimized s-vectors.',
                        default=False, action='store_true')
    parser.add_argument('-i', '--diff_intens', help='Path to Icalc diff_intens files')
    parser.add_argument('-c', '--corr_q', help='Path to cc/radius/q file')
    parser.add_argument('-o', '--out_prefix', help='Prefix to output files for intensity and CC')
    parser.add_argument('-d', '--device', help='GPU device ID(if applicable).', type=int, default=0)
    args = parser.parse_args()

    cp.cuda.Device(args.device).use()

    cc_out_fname = args.out_prefix+'_CC.dat'
    opt = diffuser.bgo_optimize.CovarianceOptimizer(args.config_file)

    if args.get_mc_intens: # Calculate Icalc from BGO s-vector
        i_calc = get_mc_icalc(opt, args.out_prefix)

        cc, q = get_cc_q(cp.array(opt.i_target), cp.array(i_calc), opt)
        if args.out_prefix is None:
            cc_out_fname = op.splitext(opt.output_fname)[0] +'_mc_diffcalc_CC.dat'

        calc_cc.save_to_file(cc_out_fname, cc, q)
    elif args.diff_intens is not None: # Get Icalc from stored diff_intens file
        with h5py.File(args.diff_intens, 'r') as fptr:
            i_calc = fptr['diff_intens'][:]

        cc, q = get_cc_q(cp.array(opt.i_target), cp.array(i_calc), opt)
        if args.out_prefix is None:
            cc_out_fname = op.splitext(args.diff_intens)[0] +'_mc_diffcalc_CC.dat'

        calc_cc.save_to_file(cc_out_fname, cc, q)
    elif args.corr_q is not None: # Get saved cc vs q data
        q, cc = np.loadtxt(args.corr_q, skiprows=1, usecols=(1,2), unpack=True)
    else:
        print('Need one of -m, -i or -c options to be specified')
        return

    # Plotting CC v/s q
    _ = P.axes(xlim=(0, q[-1]), ylim=(0, 1))
    P.plot(q, cc, 'r-')
    P.xlabel(r'1/d ($\mathrm{\AA}^{-1}$)')
    P.ylabel('Anisotropic CC')
    P.grid()
    P.show()

if __name__== '__main__':
    main()
