'''Calculate diffuse scattering from rigid body motions of electron density'''

import sys
import os.path as op
import argparse
import cupy as cp
import h5py

from diffuser import DensityGenerator

class TLSDiffuse():
    '''Calculate diffuse scattering from random distortions
    along TLS-related parameters
    '''
    def __init__(self, config_fname):
        self._parse_config(config_fname)
        self.diff_intens = None

    def _parse_config(self, config_file):
        self.dgen = DensityGenerator(config_file, vecs=False, grid=True)
        self.num_steps = self.dgen.config.getint('parameters', 'num_steps')

        self.out_fname = self.dgen.config.get_path('files', 'out_fname', fallback=None)
        if self.out_fname is None:
            pdb_fname = self.dgen.config.get_path('files', 'pdb_fname')
            self.out_fname = op.splitext(pdb_fname)[0] + '_tls_diffcalc.h5'

    def _init_diffcalc(self):
        mean_fdens = cp.zeros(tuple(self.dgen.size), dtype='c8')
        mean_intens = cp.zeros(tuple(self.dgen.size), dtype='f4')
        return mean_fdens, mean_intens, 0.

    def run_mc(self):
        '''Run Monte Carlo calculation of diffuse intensities from
        random rigid body distortions
        '''
        mean_fdens, mean_intens, denr = self._init_diffcalc()
        for i in range(self.num_steps):
            dens = self.dgen.gen_tls_dens()
            fdens = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(dens)))

            mean_fdens += fdens
            mean_intens += cp.abs(fdens)**2
            denr += 1

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
            fptr['q_voxel_size'] = self.dgen.qvox

def main():
    '''Run as console script with given config file'''
    parser = argparse.ArgumentParser(description='Diffuse scattering from TLS-related distortions')
    parser.add_argument('config_file', help='Path to config file')
    parser.add_argument('-d', '--device', help='GPU device number. Default: 0', type=int, default=0)
    args = parser.parse_args()

    cp.cuda.Device(args.device).use()

    tlsd = TLSDiffuse(args.config_file)
    tlsd.run_mc()
    tlsd.save(tlsd.out_fname)

if __name__ == '__main__':
    main()
