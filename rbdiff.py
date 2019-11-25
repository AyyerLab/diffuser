'''Calculate diffuse scattering from rigid body motions of electron density'''

import sys
import numpy
try:
    import cupy as np
    from cupyx.scipy import ndimage
    CUPY = True
    print('Using CuPy')
    np.disable_experimental_feature_warning = True
except ImportError:
    import numpy as np
    from scipy import ndimage
    CUPY = False
    print('Using NumPy/SciPy')
import mrcfile

class RBDiffuse():
    def __init__(self, rot_plane=(1,2)):
        self.size = None
        self.diff_intens = None
        self.rot_plane = rot_plane

    def rot_fdens(self, in_fdens, ang):
        out = np.empty_like(in_fdens)
        ndimage.rotate(in_fdens.real, ang, axes=self.rot_plane, reshape=False, order=0, output=out.real)
        ndimage.rotate(in_fdens.imag, ang, axes=self.rot_plane, reshape=False, order=0, output=out.imag)
        return out

    def trans_fdens(self, in_fdens, vec):
        return in_fdens * np.exp(-1j * (self.qx*vec[0] + self.qy*vec[1] + self.qz*vec[2]))

    def parse(self, fname, reset=True, **kwargs):
        with mrcfile.open(fname, 'r') as f:
            self.dens = np.array(f.data)
            self.cella = f.header.cella['x']
        if reset:
            self.initialize(**kwargs)

    def initialize(self, translate=True):
        self.fdens = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.dens)))
        self.mean_fdens = np.zeros_like(self.fdens)
        self.mean_intens = np.zeros_like(self.fdens, dtype='f4')
        self.denominator = 0.

        if translate and self.size != self.fdens.shape[0]:
            self.size = self.fdens.shape[0]
            cen = self.size // 2
            self.qx, self.qy, self.qz = np.indices(self.fdens.shape, dtype='f4')
            self.qx = (self.qx - cen) / self.size * 2. * np.pi
            self.qy = (self.qy - cen) / self.size * 2. * np.pi
            self.qz = (self.qz - cen) / self.size * 2. * np.pi

    def run_mc(self, num_steps, sigma_deg, cov_vox):
        shifts = np.random.multivariate_normal(np.zeros(3), cov_vox, size=num_steps)
        angles = np.random.randn(num_steps) * sigma_deg
        weights = np.ones(shifts.shape[0])
        for i in range(num_steps):
            if shifts[i].max() != 0.:
                modfdens = self.trans_fdens(self.fdens, tuple(shifts[i]))
            else:
                modfdens = self.fdens

            if angles[i] != 0.:
                modfdens = self.rot_fdens(modfdens, angles[i])

            self.mean_fdens += modfdens
            self.mean_intens += np.abs(modfdens)**2
            self.denominator += 1

            sys.stderr.write('\r%d/%d' % (i+1, num_steps))
        sys.stderr.write('\n')

    def rotate_weighted(self, num_steps, sigma_deg):
        angles = np.linspace(-3*sigma_deg, 3*sigma_deg, num_steps)
        weights = np.exp(-angles**2/2./sigma_deg**2)
        for i in range(num_steps):
            rotfdens = self.rot_fdens(self.fdens, angles[i])
            self.mean_fdens += rotfdens * weights[i]
            self.mean_intens += np.abs(rotfdens)**2 * weights[i]
            self.denominator += weights[i]
            sys.stderr.write('\r%d/%d: %+.3f deg (%.3e)   ' % (i+1, num_steps, angles[i], weights[i]))
        sys.stderr.write('\n')

    def aggregate(self):
        self.mean_fdens /= self.denominator
        self.mean_intens /= self.denominator

        self.diff_intens = self.mean_intens - np.abs(self.mean_fdens)**2
        # If using GPU, move output back to host
        if CUPY:
            self.diff_intens = self.diff_intens.get()

    def save(self, out_fname):
        if self.diff_intens is None:
            self.aggregate()

        # Save to file
        with mrcfile.new(out_fname, overwrite=True) as f:
            f.set_data(self.diff_intens.astype('f4'))
            for key in f.header.cella.dtype.names:
                f.header.cella[key] = self.cella

def main():
    fname = 'data/2w5j_cutout_rot_remap.ccp4'
    out_fname = 'data/2w5j_diffcalc.ccp4'
    sigma_deg = 7
    sigma_vox = 0.8
    rot_plane = (1, 2) # Rotate about x-axis
    num_steps = 201
    cov_vox = np.diag(np.ones(3)*sigma_vox**2)

    # Instantiate class
    rbd = RBDiffuse()
    # Parse electron density
    rbd.parse(fname)
    # Run MC calculation with translation and rotation
    rbd.run_mc(num_steps, sigma_deg, cov_vox)
    # Collect and save output
    rbd.save(out_fname)

if __name__ == '__main__':
    main()
