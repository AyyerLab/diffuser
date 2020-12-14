'''Calculate diffuse scattering from rigid body motions of electron density'''

import sys
import os.path as op
import configparser
import numpy as np
from scipy import special
import h5py
import cupy as cp
from cupyx.scipy import ndimage
cp.disable_experimental_feature_warning = True
import MDAnalysis as md
import mrcfile

class PCDiffuse():
    def __init__(self, config_fname):
        self._parse_config(config_fname)
        self._define_kernels()

    def _define_kernels(self):
        with open('kernels.cu', 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_gen_dens = kernels.get_function('gen_dens')

    def _parse_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        # Files
        pdb_fname = config.get('files', 'pdb_fname')
        sel_string = config.get('files', 'selection_string', fallback='all')
        vecs_fname = config.get('files', 'vecs_fname')
        self.out_fname = config.get('files', 'out_fname', fallback=None)
        qvox_fname = config.get('files', 'qvox_fname', fallback=None)
        rlatt_fname = config.get('files', 'rlatt_fname', fallback=None)

        # Parameters
        self.size = [int(s) for s in config.get('parameters', 'size').split()]
        res_edge = [float(r) for r in config.get('parameters', 'res_edge', fallback='0.').split()]
        self.sigma_deg = config.getfloat('parameters', 'sigma_deg', fallback=0.)
        sigma_vox = config.getfloat('parameters', 'sigma_vox', fallback=0.)
        self.sigma_uncorr_A = config.getfloat('parameters', 'sigma_uncorr_A', fallback=0.)
        self.cov_vox = np.identity(3) * sigma_vox**2
        self.num_steps = config.getint('parameters', 'num_steps')

        print('Parsed config file')

        # Get volume
        if len(self.size) == 1:
            self.size = np.array(self.size * 3)
        elif len(self.size) != 3:
            raise ValueError('size parameter must be either 1 or 3 space-separated numbers')
        else:
            self.size = np.array(self.size)

        # Get voxel size in 3D
        if qvox_fname is not None and res_edge != [0.]:
            raise ValueError('Both res_edge and qvox_fname defined. Pick one.')
        elif qvox_fname is not None:
            with h5py.File(qvox_fname, 'r') as f:
                self.qvox = f['q_voxel_size'][:]
        elif res_edge != [0.]:
            if len(res_edge) == 1 and res_edge[0] != 0.:
                res_edge = np.array(res_edge * 3)
            elif len(res_edge) != 3:
                raise ValueError('res_edge parameter must be either 1 or 3 space-separated numbers')
            else:
                res_edge = np.array(res_edge)
            self.qvox = np.diag(1. / res_edge / (self.size//2))
        else:
            raise ValueError('Need either res_edge of qvox_fname to define voxel parameters')
        print('q-space voxel size:\n%s' % self.qvox)
        self.a2vox = cp.array((self.qvox * self.size)).astype('f4')

        # Get centered coordinates
        univ = md.Universe(pdb_fname)
        atoms = univ.select_atoms(sel_string)
        self.avg_pos = cp.array(atoms.positions)
        self.avg_pos -= self.avg_pos.mean(0)

        # Get approximate F_0: this hack works for low-Z atoms (mass = 2Z)
        self.atom_f0 = (cp.array([np.around(a.mass) for a in atoms]) / 2.).astype('f4')
        self.atom_f0[self.atom_f0 == 0.5] = 1. # Hydrogens

        # Get PC vectors
        with h5py.File(vecs_fname, 'r') as f:
            self.vecs = cp.array(f['vecs'][:].astype('f4'))
            #self.vec_weights = cp.array(f['weights'][:])
            self.cov_weights = cp.array(f['cov_weights'][:])
            # Check number of components = 3N
            assert self.vecs.shape[0] == self.avg_pos.size
            # Check number of vectors matches number of weights
            assert self.vecs.shape[1] == self.cov_weights.shape[0]
            self.num_vecs = self.cov_weights.shape[0]
            '''
            self.vec_weights = cp.array(f['weights'][:])
            # Check number of vectors matches number of weights
            assert self.vecs.shape[1] == self.vec_weights.shape[0]
            self.num_vecs = self.vec_weights.shape[0]
            '''
            print('Using %d pricipal-component vectors on %d atoms' % (self.vecs.shape[1], self.vecs.shape[0]//3))

        # Get reciprocal lattice if defined
        if rlatt_fname is None:
            self.rlatt = None
            self.lpatt = None
        else:
            with h5py.File(rlatt_fname, 'r') as f:
                self.rlatt = cp.array(f['diff_intens'][:])
            self.lpatt = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.rlatt)))

        # B_sol filter for support mask
        cen = self.size // 2
        x, y, z = np.meshgrid(np.arange(self.size[0], dtype='f4') - self.size[0]//2,
                              np.arange(self.size[1], dtype='f4') - self.size[1]//2,
                              np.arange(self.size[2], dtype='f4') - self.size[2]//2,
                              indexing='ij')
        self.qrad = cp.array(np.linalg.norm(np.dot(self.qvox, np.array([x.ravel(), y.ravel(), z.ravel()])), axis=0).reshape(x.shape))
        # -- 30 A^2 B_sol
        self.b_sol_filt = cp.fft.ifftshift(cp.exp(-30 * self.qrad**2))
        # -- Maximum res_edge
        self.res_max = float(1. / max(self.qrad[0, cen[1], cen[2]], self.qrad[cen[0], 0, cen[2]], self.qrad[cen[0], cen[1], 0]))

        # u-vectors for LLM
        uvox = np.linalg.inv(self.qvox.T) / self.size
        self.urad = cp.array(np.linalg.norm(np.dot(uvox, np.array([x.ravel(), y.ravel(), z.ravel()])), axis=0).reshape(x.shape))

        if self.out_fname is None:
            self.out_fname = op.splitext(pdb_fname)[0] + '_diffcalc.ccp4'
        print('Parsed data')

    def _gen_rotz(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        arr = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        return cp.array(arr).astype('f4')

    def _random_rot(self):
        qq = 2.
        while True:
            quat = np.random.normal(0, 0.1, 4)
            qq = np.linalg.norm(quat)
            if qq < 1:
                quat /= qq
                break
        rotmatrix = np.array([
            [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
            [2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
            [2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]])
        return cp.array(rotmatrix.astype('f4'))

    @staticmethod
    def _random_small_rot(sigma_deg):
        angle = np.random.normal(0, sigma_deg*np.pi/180)
        while True:
            v = np.random.random(3)
            norm = np.linalg.norm(v)
            if norm < 1:
                v /= norm
                break
        tilde = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotmatrix = np.cos(angle)*np.identity(3) + np.sin(angle)*tilde + (1. - np.cos(angle))*(np.dot(tilde, tilde) + np.identity(3))

        return cp.array(rotmatrix.astype('f4'))

    def gen_random_dens(self):
        '''Generate electron density by randomly distorting average molecule

        Applies distortions along principal component vectors followed by
        rigid body translations and rotations
        '''
        # Generate distorted molecule
        if np.linalg.norm(self.cov_weights.get()) > 0.:
            #projs = cp.random.normal(cp.zeros(self.num_vecs), self.vec_weights, size=self.num_vecs, dtype='f4')
            projs = cp.array(np.random.multivariate_normal(np.zeros(self.num_vecs), self.cov_weights.get())).astype('f4')
            curr_pos = self.avg_pos + cp.dot(self.vecs, projs).reshape(3,-1).T
        else:
            curr_pos = cp.copy(self.avg_pos)

        # Apply rigid body motions
        curr_pos += cp.array(np.random.multivariate_normal(np.zeros(3), self.cov_vox)).astype('f4')
        #curr_pos = cp.dot(curr_pos, self._gen_rotz(np.random.randn() * self.sigma_deg * np.pi / 180))
        curr_pos = cp.dot(curr_pos, self._random_small_rot(self.sigma_deg))

        # Apply uncorrelated displacements
        if self.sigma_uncorr_A > 0.:
            curr_pos += cp.random.randn(*curr_pos.shape) * self.sigma_uncorr_A

        return self._calc_dens_pos(curr_pos)

    def gen_proj_dens(self, mode, weight):
        '''Generate electron density by distorting average molecule by given mode and weight

        Applies distortion along specific principal component vector
        No rigid body translations and rotations
        '''
        # Generate distorted molecule
        curr_pos = self.avg_pos + self.vecs[:,mode].reshape(3,-1).T * weight

        return self._calc_dens_pos(curr_pos)

    def get_intens(self):
        dens = self._calc_dens_pos(self.avg_pos)
        fdens = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(dens)))
        return cp.abs(fdens)**2

    def _calc_dens_pos(self, curr_pos):
        '''Calculate density from coordinates in Angstrom units'''
        dsize = cp.array(self.size)

        # Convert to voxel units
        vox_pos = cp.dot(curr_pos, self.a2vox)
        vox_pos += dsize // 2

        # Generate density grid
        n_atoms = len(vox_pos)
        dens = cp.zeros(tuple(self.size), dtype='f4')
        self.k_gen_dens((n_atoms//32+1,), (32,), (vox_pos, self.atom_f0, n_atoms, dsize, dens))

        # Solvent mask filtering
        mask = (dens<0.2).astype('f4')
        mask = cp.real(cp.fft.ifftn(cp.fft.fftn(mask)*self.b_sol_filt)) - 1
        dens += mask

        return dens

    def _initialize(self):
        self.mean_fdens = cp.zeros(tuple(self.size), dtype='c8')
        self.mean_intens = cp.zeros(tuple(self.size), dtype='f4')
        self.denominator = 0.

    def run_mc(self):
        self._initialize()
        for i in range(self.num_steps):
            dens = self.gen_random_dens()
            fdens = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(dens)))

            self.mean_fdens += fdens
            self.mean_intens += cp.abs(fdens)**2
            self.denominator += 1

            sys.stderr.write('\r%d/%d      ' % (i+1, self.num_steps))
        sys.stderr.write('\n')

        self.mean_fdens /= self.denominator
        self.mean_intens /= self.denominator

        self.diff_intens = self.mean_intens - cp.abs(self.mean_fdens)**2
        #self.diff_intens = self.diff_intens.get()

    def run_linear(self, mode, sigma):
        params = cp.linspace(-4*sigma, 4*sigma, self.num_steps, dtype='f4')
        norm_weights = cp.exp(-params**2/2./sigma**2)

        self._initialize()
        for i in range(self.num_steps):
            dens = self.gen_proj_dens(mode, params[i])
            fdens = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(dens)))

            self.mean_fdens += fdens * norm_weights[i]
            self.mean_intens += np.abs(fdens)**2 * norm_weights[i]
            self.denominator += norm_weights[i]

            sys.stderr.write('\r%d/%d      ' % (i+1, self.num_steps))
        sys.stderr.write('\n')

        self.mean_fdens /= self.denominator
        self.mean_intens /= self.denominator

        self.diff_intens = self.mean_intens - cp.abs(self.mean_fdens)**2
        self.diff_intens = self.diff_intens.get()

    def liquidize(self, intens, sigma_A, gamma_A):
        s_sq = (2. * cp.pi * sigma_A * self.qrad)**2
        patt = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(intens)))

        slimits = np.array([np.real(np.sqrt(special.lambertw(-(1.e-3 * special.factorial(n))**(1./n) / n, k=0)) * np.sqrt(n) * -1j) for n in range(1,150)])
        if slimits.max() > 2. * np.pi * sigma_A / self.res_max:
            n_max = np.where(slimits > 2. * np.pi * sigma_A / self.res_max)[0][0] + 1
        else:
            print('No effect of liquid-like motions with these parameters')
            return intens

        liq = cp.zeros_like(intens)
        for n in range(n_max):
            kernel = cp.exp(-n * self.urad / gamma_A)
            weight = cp.exp(-s_sq + n*cp.log(s_sq) - float(special.loggamma(n+1)))
            liq += weight * cp.abs(cp.fft.fftshift(cp.fft.ifftn(patt * kernel)))
            sys.stderr.write('\rLiquidizing: %d/%d' % (n+1, n_max))
        sys.stderr.write('\n')

        return liq

    def liqlatt(self, sigma_A, gamma_A):
        if self.rlatt is None:
            raise AttributeError('Provide rlatt to apply liqlatt')
        s_sq = (2 * cp.pi * sigma_A * self.qrad)**2
        slimits = np.array([np.real(np.sqrt(special.lambertw(-(1.e-3 * special.factorial(n))**(1./n) / n, k=0)) * np.sqrt(n) * -1j)
                            for n in range(1, 150)])
        if slimits.max() > 2 * np.pi * sigma_A / self.res_max:
            n_max = np.where(slimits > 2. * np.pi * sigma_A / self.res_max)[0][0] + 1
        else:
            return self.rlatt

        if n_max == 0:
            return cp.ones_like(self.rlatt)

        liq = cp.zeros_like(self.rlatt)
        for n in range(1, n_max):
            weight = cp.exp(-s_sq + n * cp.log(s_sq) - float(special.loggamma(n+1)))
            kernel = cp.exp(-n * self.urad / gamma_A)
            liq += weight * cp.abs(cp.fft.fftshift(cp.fft.ifftn(self.lpatt * kernel)))
            sys.stderr.write('\rLiquidizing: %d/%d' % (n, n_max-1))
        sys.stderr.write('\n')

        return liq

    def save(self, out_fname):
        print('Writing intensities to', out_fname)
        with mrcfile.new(out_fname, overwrite=True) as f:
            f.set_data(self.diff_intens.astype('f4'))

def main():
    import argparse

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
