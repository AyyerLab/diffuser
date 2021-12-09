import sys
import os.path as op
import argparse
import numpy as np
import h5py
import cupy as cp

from diffuser import DensityGenerator

class MDProcessor():
    '''Process MD trajectory to get Principal Component modes'''
    def __init__(self, config_file):
        self.dgen = DensityGenerator(config_file, vecs=False, grid=False)

        self.traj_fname = self.dgen.config.get_path('files', 'traj_fname')

        self.cov_fname = self.dgen.config.get_path('files', 'cov_fname', fallback=None)
        if self.cov_fname is None:
            self.cov_fname = op.splitext(self.traj_fname)[0] + '_cov.h5'

        self.vecs_fname = self.dgen.config.get_path('files', 'vecs_fname', fallback=None)
        if self.vecs_fname is None:
            self.vecs_fname = op.splitext(self.traj_fname)[0] + '_vecs.h5'

        self.f_weighting = self.dgen.config.getboolean('parameters', 'apply_f_weighting', fallback=True)
        if not self.f_weighting:
            print('Not applying F-weighting to covariance matrix')

        self.cov = None

    def calc_cov(self, num_frames=-1, first_frame=0, frame_stride=1):
        '''Calculate and save displacement covariance matrix'''
        if num_frames + first_frame > len(self.dgen.univ.trajectory) or num_frames == -1:
            num_frames = len(self.dgen.univ.trajectory) - first_frame
        print('Calculating displacement CC for %d frames' % num_frames)

        num_atoms = self.dgen.atoms.n_atoms
        print('Position of %d atoms will be used' % num_atoms)

        print('Calculating mean position over selected frames')
        mean_pos = cp.zeros((num_atoms, 3))
        for i in range(first_frame, num_frames + first_frame, frame_stride):
            _ = self.dgen.univ.trajectory[i]
            mean_pos += cp.array(self.dgen.atoms.positions)
            sys.stderr.write('\rFrame %d'%i)
        sys.stderr.write('\n')
        mean_pos /= (num_frames/frame_stride)

        print('Calculating displacement CCs')
        self.cov = cp.zeros((6, num_atoms, num_atoms))
        for i in range(first_frame, num_frames + first_frame, frame_stride):
            _ = self.dgen.univ.trajectory[i]
            pos = cp.array(self.dgen.atoms.positions) - mean_pos
            if self.f_weighting:
                pos = (pos.T * self.dgen.atom_f0).T # F-weighting the displacements
            self.cov[0] += cp.outer(pos[:, 0], pos[:, 0])
            self.cov[1] += cp.outer(pos[:, 1], pos[:, 1])
            self.cov[2] += cp.outer(pos[:, 2], pos[:, 2])
            self.cov[3] += cp.outer(pos[:, 0], pos[:, 1])
            self.cov[4] += cp.outer(pos[:, 1], pos[:, 2])
            self.cov[5] += cp.outer(pos[:, 2], pos[:, 0])
            sys.stderr.write('\rFrame %d'%i)
        sys.stderr.write('\n')

        #mean_pos -= mean_pos.mean(0)
        mean_pos = mean_pos.get()
        dist = np.linalg.norm(np.subtract.outer(mean_pos, mean_pos)[:,[0,1,2],:,[0,1,2]], axis=0)

        print('Saving covariance matrix to', self.cov_fname)
        with h5py.File(self.cov_fname, 'w') as fptr:
            hcov = self.cov.get() # pylint: disable=no-member
            hf0 = self.dgen.atom_f0.get()

            fptr['cov'] = hcov
            fptr['dist'] = dist
            fptr['mean_pos'] = mean_pos
            fptr['f0'] = hf0 # Atomic scattering factors

    def _load_cov(self):
        print('Loading covariance matrix from', self.cov_fname)
        with h5py.File(self.cov_fname, 'r') as fptr:
            self.cov = self._get_allcov(cp.array(fptr['cov'][:]))

    def calc_vecs(self, num_vecs=100):
        '''Calculate principal component vectors by diagonalizing covariance matrix'''
        if self.cov is None:
            self._load_cov()

        if len(self.cov.shape) == 3:
            self.cov = self._get_allcov(self.cov)

        # Diagonalizing (3N, 3N) matrix
        sys.stderr.write('Diagonalizing...')
        vals, vecs = cp.linalg.eigh(self.cov)
        sys.stderr.write('done\n')

        # Note that eigenvalues are sorted in INCREASING order with sign
        # To get sorting acc. to absolute value with max first...
        sorter = cp.abs(vals).argsort()[::-1]

        if self.f_weighting:
            # We need to remove the scattering f-weighting from the eigenvectors
            vecs = (vecs.T / cp.tile(self.dgen.atom_f0.get(), 3)).T

        # Select first N eigenvectors
        vals_n = vals[sorter[:num_vecs]].astype('f4')
        vecs_n = vecs[:, sorter[:num_vecs]].astype('f4')

        # Save to file
        print('Saving PC vectors to', self.vecs_fname)
        with h5py.File(self.vecs_fname, 'w') as fptr:
            fptr['vecs'] = vecs_n.get()
            fptr['cov_weights'] = np.diag(np.ones(num_vecs) * 10.)
            fptr['orig_vals'] = vals_n.get()

    @staticmethod
    def _get_allcov(cov):
        allcov = cp.zeros((3, cov.shape[1], 3, cov.shape[2]))

        allcov[0, :, 0] = cov[0]
        allcov[0, :, 1] = cov[3].T
        allcov[0, :, 2] = cov[5]
        allcov[1, :, 0] = cov[3]
        allcov[1, :, 1] = cov[1]
        allcov[1, :, 2] = cov[4].T
        allcov[2, :, 0] = cov[5].T
        allcov[2, :, 1] = cov[4]
        allcov[2, :, 2] = cov[2]

        return allcov.reshape(cov.shape[1]*3, cov.shape[2]*3)

    def write_mean_pos(self, pdb_fname=None):
        '''Write mean positions from covariance file to pdb'''
        if pdb_fname is None:
            pdb_fname = op.splitext(self.traj_fname)[0] + '_avg.pdb'

        with h5py.File(self.cov_fname, 'r') as fptr:
            p_avg = fptr['mean_pos'][:]

        self.dgen.atoms.positions = p_avg

        print('Writing mean_positions to', pdb_fname)
        self.dgen.atoms.write(pdb_fname)

def main():
    '''Run as console script with given config file'''
    parser = argparse.ArgumentParser(description='Process MD trajectory')
    parser.add_argument('config_file', help='Path to config file')
    parser.add_argument('-C', '--calc_cov', action='store_true',
                        help='Calculate covariance matrix from trajectory')
    parser.add_argument('-D', '--diagonalize', action='store_true',
                        help='Calculate principal vectors from covariance matrix')
    parser.add_argument('-M', '--meanpos', action='store_true',
                        help='Write mean positions of trajectory to PDB file')
    parser.add_argument('-n', '--num_vecs', type=int, default=100,
                        help='Number of PC vecs to save (default: 100)')
    parser.add_argument('-d', '--device', help='GPU device number. Default: 0', type=int, default=0)
    args = parser.parse_args()

    if not (args.calc_cov or args.diagonalize or args.meanpos):
        print('Need either -C or -D option to describe what to do')
        return

    cp.cuda.Device(args.device).use()

    mdproc = MDProcessor(args.config_file)
    if args.calc_cov:
        mdproc.calc_cov()
    if args.diagonalize:
        mdproc.calc_vecs(args.num_vecs)
    if args.meanpos:
        mdproc.write_mean_pos()

if __name__ == '__main__':
    main()
