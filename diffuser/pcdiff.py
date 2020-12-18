'''Calculate diffuse scattering from rigid body motions of electron density'''

import sys
import os.path as op
import argparse
import numpy as np
import cupy as cp
import h5py

from diffuser import DensityGenerator

class PCDiffuse():
    '''Calculate diffuse scattering from random distortions
    along principal component (PC) vectors
    '''
    def __init__(self, config_fname):
        self._parse_config(config_fname)
        self.diff_intens = None

    def _parse_config(self, config_file):
        self.dgen = DensityGenerator(config_file, vecs=True, grid=True)
        self.num_steps = self.dgen.config.getint('parameters', 'num_steps')

        self.out_fname = self.dgen.config.get_path('files', 'out_fname', fallback=None)
        if self.out_fname is None:
            pdb_fname = self.dgen.config.get_path('files', 'pdb_fname')
            self.out_fname = op.splitext(pdb_fname)[0] + '_pc_diffcalc.h5'

    def _init_diffcalc(self):
        mean_fdens = cp.zeros(tuple(self.dgen.size), dtype='c8')
        mean_intens = cp.zeros(tuple(self.dgen.size), dtype='f4')
        return mean_fdens, mean_intens, 0.

    def run_mc(self):
        '''Run Monte Carlo calculation of diffuse intensities from
        random distortions along PC modes
        '''
        mean_fdens, mean_intens, denr = self._init_diffcalc()
        for i in range(self.num_steps):
            dens = self.dgen.gen_random_dens()
            fdens = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(dens)))

            mean_fdens += fdens
            mean_intens += cp.abs(fdens)**2
            denr += 1

            sys.stderr.write('\r%d/%d      ' % (i+1, self.num_steps))
        sys.stderr.write('\n')

        mean_fdens /= denr
        mean_intens /= denr

        self.diff_intens = mean_intens - cp.abs(mean_fdens)**2

    def run_linear(self, mode, sigma):
        '''Calculate diffuse intensity from Gaussian distortions about
        selected PC mode

        Arguments:
            mode - Mode number
            sigma - Standard deviation of mode weight
        '''
        params = cp.linspace(-4*sigma, 4*sigma, self.num_steps, dtype='f4')
        norm_weights = cp.exp(-params**2/2./sigma**2)

        mean_fdens, mean_intens, denr = self._init_diffcalc()
        for i in range(self.num_steps):
            dens = self.dgen.gen_proj_dens(mode, params[i])
            fdens = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(dens)))

            mean_fdens += fdens * norm_weights[i]
            mean_intens += np.abs(fdens)**2 * norm_weights[i]
            denr += norm_weights[i]

            sys.stderr.write('\r%d/%d      ' % (i+1, self.num_steps))
        sys.stderr.write('\n')

        mean_fdens /= denr
        mean_intens /= denr

        self.diff_intens = mean_intens - cp.abs(mean_fdens)**2

    def save(self, out_fname):
        '''Save diffuse intensities to file'''
        if self.diff_intens is None:
            raise ValueError('Calculate diffuse intensities first')

        print('Writing intensities to', out_fname)
        with h5py.File(out_fname, 'w') as fptr:
            fptr['diff_intens'] = self.diff_intens.get().astype('f4')

def main():
    '''Run as console script with given config file'''
    parser = argparse.ArgumentParser(description='Diffuse from PC-distorted molecules')
    parser.add_argument('config_file', help='Path to config file')
    parser.add_argument('-d', '--device', help='GPU device number. Default: 0', type=int, default=0)
    args = parser.parse_args()

    cp.cuda.Device(args.device).use()

    pcd = PCDiffuse(args.config_file)
    pcd.run_mc()
    pcd.save(pcd.out_fname)

if __name__ == '__main__':
    main()
