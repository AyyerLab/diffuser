import sys
import os.path as op
import argparse
import numpy as np
import h5py
import cupy as cp

from diffuser import RBDiffuse, DensityGenerator

class TrajectoryDiffuse():
    '''Generate diffuse intensities from MD trajectory and rigid-body parameters

    Also generates displacement covariance matrix from trajectory
    '''
    def __init__(self, config_file, cov_only=False):
        grid = not cov_only
        self.dgen = DensityGenerator(config_file, vecs=False, grid=grid)
        self.num_steps = self.dgen.config.getint('parameters', 'num_steps')

        self.out_fname = self.dgen.config.get_path('files', 'out_fname', fallback=None)
        if self.out_fname is None:
            traj_fname = self.dgen.config.get_path('files', 'traj_fname')
            self.out_fname = op.splitext(traj_fname)[0] + '_traj_diffcalc.h5'

        self.rbd = RBDiffuse(self.dgen.rot_plane)

    def run(self, num_frames=-1, first_frame=0, frame_stride=1, init=True):
        '''Calculate and save diffuse scattering from trajectory and rigid-body parameters'''
        if num_frames == -1:
            num_frames = len(self.dgen.univ.trajectory) - first_frame
        print('Calculating diffuse intensities from %d frames' % num_frames)

        for i in range(first_frame, num_frames + first_frame, frame_stride):
            self.rbd.dens = cp.copy(self.dgen.gen_frame_dens(i))
            self.rbd.fdens = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.rbd.dens)))

            if init:
                do_translate = (self.dgen.cov_vox.sum() != 0.)
                self.rbd.init_diffcalc(translate=do_translate)
                init = False
            prefix = 'Frame %d/%d: ' % (i, num_frames + first_frame)
            self.rbd.run_mc(self.num_steps, self.dgen.sigma_deg, self.dgen.cov_vox, prefix=prefix)
        sys.stderr.write('\n')

        print('Saving output to', self.out_fname)
        self.rbd.save(self.out_fname)

    def run_cc(self, num_frames=-1, first_frame=0, frame_stride=1):
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
        corr = cp.zeros((6, num_atoms, num_atoms))
        for i in range(first_frame, num_frames + first_frame, frame_stride):
            _ = self.dgen.univ.trajectory[i]
            pos = cp.array(self.dgen.atoms.positions) - mean_pos
            pos = (pos.T * self.dgen.atom_f0).T # F-weighting the displacements
            corr[0] += cp.outer(pos[:, 0], pos[:, 0])
            corr[1] += cp.outer(pos[:, 1], pos[:, 1])
            corr[2] += cp.outer(pos[:, 2], pos[:, 2])
            corr[3] += cp.outer(pos[:, 0], pos[:, 1])
            corr[4] += cp.outer(pos[:, 1], pos[:, 2])
            corr[5] += cp.outer(pos[:, 2], pos[:, 0])
            sys.stderr.write('\rFrame %d'%i)
        sys.stderr.write('\n')

        cc_fname = op.splitext(self.out_fname)[0] + '_cov.h5'
        print('Saving covariance matrix to', cc_fname)

        #mean_pos -= mean_pos.mean(0)
        mean_pos = mean_pos.get()
        dist = np.linalg.norm(np.subtract.outer(mean_pos, mean_pos)[:,[0,1,2],:,[0,1,2]], axis=0)

        with h5py.File(cc_fname, 'w') as fptr:
            hcorr = corr.get() # pylint: disable=no-member
            hf0 = self.dgen.atom_f0.get()

            fptr['corr'] = hcorr
            fptr['dist'] = dist
            fptr['mean_pos'] = mean_pos
            fptr['f0'] = hf0 # Atomic scattering factors

def main():
    '''Run as console script with given config file'''
    parser = argparse.ArgumentParser(description='Generate diffuse intensities from rigid body motion')
    parser.add_argument('-c', '--config',
                        help='Config file. Default: config.ini', default='config_traj.ini')
    parser.add_argument('-n', '--num_frames',
                        help='Number of frames to process. Default: -1 (all)', type=int, default=-1)
    parser.add_argument('-f', '--first_frame',
                        help='Index of first frame. Default: 0', type=int, default=0)
    parser.add_argument('-s', '--frame_stride',
                        help='Stride length for frames. Default: 1', type=int, default=1)
    parser.add_argument('-d', '--device',
                        help='GPU device ID (if applicable). Default: 0', type=int, default=0)
    parser.add_argument('-C', '--cov',
                        help='Calculate displacement covariance instead of diffuse intensities. Default=False',
                        action='store_true')
    args = parser.parse_args()

    cp.cuda.Device(args.device).use()

    trajdiff = TrajectoryDiffuse(args.config, cov_only=args.cov)
    if args.cov:
        trajdiff.run_cc(args.num_frames, first_frame=args.first_frame, frame_stride=args.frame_stride)
    else:
        trajdiff.run(args.num_frames, first_frame=args.first_frame, frame_stride=args.frame_stride)

if __name__ == '__main__':
    main()
