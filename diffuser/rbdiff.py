'''Calculate diffuse scattering from rigid body motions of electron density'''

import sys
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage
import mrcfile
import h5py

class RBDiffuse():
    '''Calculate diffuse intensities from electron density map and rigid-body parameters'''
    def __init__(self, rot_plane=(1,2)):
        self.size = None
        self.diff_intens = None
        self.rot_plane = rot_plane
        self.mean_fdens = None
        self.mean_intens = None
        self.denominator = None

    def _rot_fdens(self, in_fdens, ang):
        out = cp.empty_like(in_fdens)
        ndimage.rotate(in_fdens.real, ang, axes=self.rot_plane, reshape=False, order=0, output=out.real)
        ndimage.rotate(in_fdens.imag, ang, axes=self.rot_plane, reshape=False, order=0, output=out.imag)
        return out

    def _trans_fdens(self, in_fdens, vec):
        return in_fdens * cp.exp(-1j * (self.qx*vec[0] + self.qy*vec[1] + self.qz*vec[2]))

    def parse(self, fname):
        '''Parse electron density from file'''
        with mrcfile.open(fname, 'r') as f:
            self.dens = cp.array(f.data)
            self.cella = f.header.cella['x']
        self.fdens = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.dens)))

    def init_diffcalc(self, translate=True):
        '''Initialize accumulation of 1st and 2nd moments for diffuse intensity calculation'''
        self.mean_fdens = cp.zeros_like(self.fdens)
        self.mean_intens = cp.zeros_like(self.fdens, dtype='f4')
        self.denominator = 0.

        if translate and self.size != self.fdens.shape[0]:
            self.size = self.fdens.shape[0]
            cen = self.size // 2
            self.qx, self.qy, self.qz = cp.indices(self.fdens.shape, dtype='f4')
            self.qx = (self.qx - cen) / self.size * 2. * cp.pi
            self.qy = (self.qy - cen) / self.size * 2. * cp.pi
            self.qz = (self.qz - cen) / self.size * 2. * cp.pi

    def run_mc(self, num_steps, sigma_deg, cov_vox, prefix='', reset=False):
        '''Run Monte Carlo sampling of rigid body motions to get diffuse intensity'''
        shifts = cp.array(np.random.multivariate_normal(np.zeros(3), cov_vox, size=num_steps))
        angles = cp.random.randn(num_steps) * sigma_deg
        if reset:
            self.init_diffcalc(translate=True)

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

    def rotate_weighted(self, num_steps, sigma_deg, reset=True):
        '''Get diffuse intensities by rotating object about single axis'''
        angles = cp.linspace(-3*sigma_deg, 3*sigma_deg, num_steps)
        weights = cp.exp(-angles**2/2./sigma_deg**2)
        if reset:
            self.init_diffcalc(translate=False)

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
    '''Run with test data (example code)'''
    fname = 'data/2w5j_cutout_rot_remap.ccp4'
    out_fname = 'data/2w5j_diffcalc.ccp4'
    sigma_deg = 7
    sigma_vox = 0.8
    rot_plane = (1, 2) # Rotate about x-axis
    num_steps = 21
    cov_vox = np.identity(3)*sigma_vox**2

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
