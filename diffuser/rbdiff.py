# pylint: disable=too-many-instance-attributes

import sys
import argparse
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage
import mrcfile
import h5py

from diffuser import DiffuserConfig

class RBDiffuse():
    '''Calculate diffuse intensities from electron density map and rigid-body parameters'''
    def __init__(self, rot_plane=(1,2)):
        self.rot_plane = rot_plane
        self.size = None
        self.diff_intens = None
        self.dens = None
        self.fdens = None
        self.cella = None
        self.mean_fdens = None
        self.mean_intens = None
        self.denominator = None
        self.qvec = None

    def _rot_fdens(self, in_fdens, ang):
        out = cp.empty_like(in_fdens)
        ndimage.rotate(in_fdens.real, ang, axes=self.rot_plane, reshape=False, order=0, output=out.real)
        ndimage.rotate(in_fdens.imag, ang, axes=self.rot_plane, reshape=False, order=0, output=out.imag)
        return out

    def _trans_fdens(self, in_fdens, vec):
        x, y, z = self.qvec
        return in_fdens * cp.exp(-1j * (x*vec[0] + y*vec[1] + z*vec[2]))

    def parse(self, fname):
        '''Parse electron density from file'''
        with mrcfile.open(fname, 'r') as fptr:
            self.dens = cp.array(fptr.data)
            self.cella = fptr.header.cella['x']
        self.fdens = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.dens)))

    def init_diffcalc(self, translate=True):
        '''Initialize accumulation of 1st and 2nd moments for diffuse intensity calculation'''
        self.mean_fdens = cp.zeros_like(self.fdens)
        self.mean_intens = cp.zeros_like(self.fdens, dtype='f4')
        self.denominator = 0.

        if translate and self.size != self.fdens.shape[0]:
            self.size = self.fdens.shape[0]
            cen = self.size // 2
            x, y, z = cp.indices(self.fdens.shape, dtype='f4')
            x = (x - cen) / self.size * 2. * cp.pi
            y = (y - cen) / self.size * 2. * cp.pi
            z = (z - cen) / self.size * 2. * cp.pi
            self.qvec = (x, y, z)

    def run_mc(self, num_steps, sigma_deg, cov_vox, prefix=''):
        '''Run Monte Carlo sampling of rigid body motions to get diffuse intensity

        By default, this adds to the existing 1st and second moments.
        Run init_diffcalc() to reset accumulation.
        '''
        shifts = cp.array(np.random.multivariate_normal(np.zeros(3), cov_vox, size=num_steps))
        angles = cp.random.randn(num_steps) * sigma_deg

        for i in range(num_steps):
            if shifts[i].max() != 0.:
                modfdens = self._trans_fdens(self.fdens, tuple(shifts[i]))
            else:
                modfdens = self.fdens

            if angles[i] != 0.:
                modfdens = self._rot_fdens(modfdens, angles[i])

            self.mean_fdens += modfdens
            self.mean_intens += cp.abs(modfdens)**2
            self.denominator += 1

            sys.stderr.write('\r%s%d/%d      ' % (prefix, i+1, num_steps))
        sys.stderr.write('\n')

    def rotate_weighted(self, num_steps, sigma_deg):
        '''Get diffuse intensities by rotating object about single axis

        By default, this adds to the existing 1st and second moments.
        Run init_diffcalc() to reset accumulation.
        '''
        angles = cp.linspace(-3*sigma_deg, 3*sigma_deg, num_steps)
        weights = cp.exp(-angles**2/2./sigma_deg**2)

        for i in range(num_steps):
            rotfdens = self._rot_fdens(self.fdens, angles[i])
            self.mean_fdens += rotfdens * weights[i]
            self.mean_intens += cp.abs(rotfdens)**2 * weights[i]
            self.denominator += weights[i]
            sys.stderr.write('\r%d/%d: %+.3f deg (%.3e)   ' % (i+1, num_steps, angles[i], weights[i]))
        sys.stderr.write('\n')

    def aggregate(self):
        '''Finish accumulation and calculate diffuse intensities'''
        if self.denominator is None or self.denominator == 0.:
            raise ValueError('Run a few steps before aggregating intensities')
        self.mean_fdens /= self.denominator
        self.mean_intens /= self.denominator

        self.diff_intens = self.mean_intens - cp.abs(self.mean_fdens)**2
        self.diff_intens = self.diff_intens.get()

    def save(self, out_fname):
        '''Save to given HDF5 file'''
        if self.diff_intens is None:
            self.aggregate()
        with h5py.File(out_fname, 'w') as fptr:
            fptr['diff_intens'] = self.diff_intens.astype('f4')

def main():
    '''Run as console script with config file'''
    parser = argparse.ArgumentParser(description='Diffuse scattering from rigid-body motion of electron density')
    parser.add_argument('config_file', help='Path to config file')
    parser.add_argument('-d', '--device', help='GPU device number. Default: 0', type=int, default=0)
    args = parser.parse_args()

    conf = DiffuserConfig(args.config_file)
    fname = conf.get_path('files', 'dens_fname')
    out_fname = conf.get_path('files', 'output_fname')
    num_steps = conf.getint('parameters', 'num_steps')
    sigma_deg = conf.getfloat('parameters', 'sigma_deg', fallback=0.)
    rot_plane = tuple(np.delete([0, 1, 2], conf.getint('parameters', 'rot_axis', fallback=2)))
    sigma_vox = conf.getfloat('parameters', 'sigma_vox', fallback=0.)
    cov_vox = np.identity(3) * sigma_vox**2

    # Instantiate class
    rbd = RBDiffuse(rot_plane=rot_plane)
    # Parse electron density
    rbd.parse(fname)
    # Run MC calculation with translation and rotation
    rbd.run_mc(num_steps, sigma_deg, cov_vox)
    # Collect and save output
    rbd.save(out_fname)

if __name__ == '__main__':
    main()
