'''Calculate diffuse scattering from rigid body motions of electron density'''

import sys
import os.path as op
import configparser
import numpy as np
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
        self.k_gen_dens = cp.RawKernel(r'''
        extern "C" __global__
        void gen_dens(const float *positions,
                      const float *atom_f0,
                      const long long num_atoms,
                      const long long size,
                      float *dens) {
            int n = blockIdx.x * blockDim.x + threadIdx.x ;
            if (n >= num_atoms)
                return ;

            float tx = positions[n*3 + 0] ;
            float ty = positions[n*3 + 1] ;
            float tz = positions[n*3 + 2] ;
            float val = atom_f0[n] ;
            int ix = __float2int_rd(tx), iy = __float2int_rd(ty), iz = __float2int_rd(tz) ;
            if ((ix < 0) || (ix > size - 2) ||
                (iy < 0) || (iy > size - 2) ||
                (iz < 0) || (iz > size - 2))
                return ;
            float fx = tx - ix, fy = ty - iy, fz = tz - iz ;
            float cx = 1. - fx, cy = 1. - fy, cz = 1. - fz ;

            atomicAdd(&dens[ix*size*size + iy*size + iz], val*cx*cy*cz) ;
            atomicAdd(&dens[ix*size*size + iy*size + iz+1], val*cx*cy*fz) ;
            atomicAdd(&dens[ix*size*size + (iy+1)*size + iz], val*cx*fy*cz) ;
            atomicAdd(&dens[ix*size*size + (iy+1)*size + iz+1], val*cx*fy*fz) ;
            atomicAdd(&dens[(ix+1)*size*size + iy*size + iz], val*fx*cy*cz) ;
            atomicAdd(&dens[(ix+1)*size*size + iy*size + iz+1], val*fx*cy*fz) ;
            atomicAdd(&dens[(ix+1)*size*size + (iy+1)*size + iz], val*fx*fy*cz) ;
            atomicAdd(&dens[(ix+1)*size*size + (iy+1)*size + iz+1], val*fx*fy*fz) ;
        }
        ''', 'gen_dens')

    def _parse_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        # Files
        pdb_fname = config.get('files', 'pdb_fname')
        sel_string = config.get('files', 'selection_string', fallback='all')
        vecs_fname = config.get('files', 'vecs_fname')
        self.out_fname = config.get('files', 'out_fname', fallback=None)

        # Parameters
        self.size = config.getint('parameters', 'size')
        self.res_edge = config.getfloat('parameters', 'res_edge')
        self.sigma_deg = config.getfloat('parameters', 'sigma_deg', fallback=0.)
        sigma_vox = config.getfloat('parameters', 'sigma_vox', fallback=0.)
        self.sigma_uncorr_A = config.getfloat('parameters', 'sigma_uncorr_A', fallback=0.)
        self.cov_vox = np.identity(3) * sigma_vox**2
        self.num_steps = config.getint('parameters', 'num_steps')

        print('Parsed config file')

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

        # B_sol filter for support mask
        cen = self.size//2
        ind = np.linspace(-cen, cen, self.size, dtype='f4')
        x, y, z = np.meshgrid(ind, ind, ind, indexing='ij')
        q = np.sqrt(x*x + y*y + z*z) / cen / self.res_edge
        # 30 A^2 B_sol
        self.b_sol_filt = cp.array(np.fft.ifftshift(np.exp(-30 * q * q)))

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

    def _random_small_rot(self, sigma_deg):
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

        # Convert to voxel units
        curr_pos *= 2. / self.res_edge
        curr_pos += self.size // 2

        # Generate density grid
        n_atoms = len(curr_pos)
        dens = cp.zeros(3*(self.size,), dtype='f4')
        self.k_gen_dens((n_atoms//32+1,), (32,), (curr_pos, self.atom_f0, n_atoms, self.size, dens))

        # Solvent mask filtering
        mask = (dens<0.2).astype('f4')
        mask = cp.real(cp.fft.ifftn(cp.fft.fftn(mask)*self.b_sol_filt)) - 1
        dens += mask

        return dens

    def gen_proj_dens(self, mode, weight):
        '''Generate electron density by distorting average molecule by given mode and weight

        Applies distortion along specific principal component vector
        No rigid body translations and rotations
        '''
        # Generate distorted molecule
        curr_pos = self.avg_pos + self.vecs[:,mode].reshape(3,-1).T * weight

        # Convert to voxel units
        curr_pos *= 2. / self.res_edge
        curr_pos += self.size // 2

        # Generate density grid
        n_atoms = len(curr_pos)
        dens = cp.zeros(3*(self.size,), dtype='f4')
        self.k_gen_dens((n_atoms//32+1,), (32,), (curr_pos, self.atom_f0, n_atoms, self.size, dens))

        # Solvent mask filtering
        mask = (dens<0.2).astype('f4')
        mask = cp.real(cp.fft.ifftn(cp.fft.fftn(mask)*self.b_sol_filt)) - 1
        dens += mask

        return dens

    def _initialize(self):
        self.mean_fdens = cp.zeros(3*(self.size,), dtype='c8')
        self.mean_intens = cp.zeros(3*(self.size,), dtype='f4')
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
        self.diff_intens = self.diff_intens.get()

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

    def save(self, out_fname):
        print('Writing intensities to', out_fname)
        with mrcfile.new(out_fname, overwrite=True) as f:
            f.set_data(self.diff_intens.astype('f4'))
            for key in f.header.cella.dtype.names:
                f.header.cella[key] = (self.size // 2) * self.res_edge

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
