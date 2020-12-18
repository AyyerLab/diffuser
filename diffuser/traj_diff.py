import sys
import os.path as op
import argparse
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
            if cov_only:
                self.out_fname = op.splitext(traj_fname)[0] + '_cov.h5'
            else:
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

    def get_frame_dens(self, ind, out_fname=None):
        '''Save electron density of trajectory frame'''
        dens = self.dgen.gen_frame_dens(ind).get()
        if out_fname is None:
            traj_fname = self.dgen.config.get_path('files', 'traj_fname')
            out_fname = op.splitext(traj_fname)[0] + '_frame_dens.h5'

        with h5py.File(out_fname, 'w') as fptr:
            fptr['dens'] = dens
            fptr['index'] = ind

def main():
    '''Run as console script with given config file'''
    parser = argparse.ArgumentParser(description='Generate diffuse intensities from rigid body motion')
    parser.add_argument('config_file', help='Path to config file')
    parser.add_argument('-n', '--num_frames',
                        help='Number of frames to process. Default: -1 (all)', type=int, default=-1)
    parser.add_argument('-f', '--first_frame',
                        help='Index of first frame. Default: 0', type=int, default=0)
    parser.add_argument('-s', '--frame_stride',
                        help='Stride length for frames. Default: 1', type=int, default=1)
    parser.add_argument('-d', '--device',
                        help='GPU device ID (if applicable). Default: 0', type=int, default=0)
    args = parser.parse_args()

    cp.cuda.Device(args.device).use()

    trajdiff = TrajectoryDiffuse(args.config_file)
    trajdiff.run(args.num_frames, first_frame=args.first_frame, frame_stride=args.frame_stride)

if __name__ == '__main__':
    main()
