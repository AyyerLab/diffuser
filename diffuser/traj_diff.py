import sys
import os.path as op
import configparser
import argparse
import numpy as np
import h5py
import cupy as cp
import MDAnalysis as md

from .rbdiff import RBDiffuse

class TrajectoryDiffuse():
    '''Generate diffuse intensities from MD trajectory and rigid-body parameters'''
    def __init__(self, config_file):
        self._parse_config(config_file)
        self.rbd = RBDiffuse(self.rot_plane)
        self._define_kernels()

    def _define_kernels(self):
        with open('kernels.cu', 'r') as fptr:
            kernels = cp.RawModule(code=fptr.read())
        self.k_gen_dens = kernels.get_function('gen_dens')

    def _parse_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        # Files
        traj_fname = config.get('files', 'traj_fname', fallback=None)
        traj_flist = config.get('files', 'traj_flist', fallback=None)
        topo_fname = config.get('files', 'topo_fname', fallback=None)
        sel_string = config.get('files', 'selection_string', fallback='all')
        pdb_fname = config.get('files', 'pdb_fname', fallback=None)
        self.out_fname = config.get('files', 'out_fname', fallback=None)

        # Parameters
        self.size = config.getint('parameters', 'size')
        self.res_edge = config.getfloat('parameters', 'res_edge')
        self.sigma_deg = config.getfloat('parameters', 'sigma_deg', fallback=0.)
        sigma_vox = config.getfloat('parameters', 'sigma_vox', fallback=0.)
        rot_axis = config.getint('parameters', 'rot_axis', fallback=0)
        self.num_steps = config.getint('parameters', 'num_steps')

        self.cov_vox = np.identity(3) * sigma_vox**2
        self.rot_plane = tuple(np.delete([0,1,2], rot_axis))
        print('Parsed config file')

        if pdb_fname is None:
            if topo_fname is None:
                raise AttributeError('Need either pdb_fname or topo_fname (if using trajectories)')

            if traj_fname is not None and traj_flist is not None:
                raise AttributeError('Cannot specify both traj_fname and traj_flist. Pick one.')
            if traj_fname is not None:
                self.univ = md.Universe(topo_fname, traj_fname)
            elif traj_flist is not None:
                with open(traj_flist, 'r') as fptr:
                    flist = [l.strip() for l in fptr.readlines()]
                self.univ = md.Universe(topo_fname, flist)
            else:
                raise AttributeError('Need one of traj_fname or traj_flist with topology file')

            self._initialize_md(sel_string)
            if self.out_fname is None:
                self.out_fname = op.splitext(traj_fname)[0] + '_rbt'+str(sigma_vox) +'_diffcalc.h5'#'_diffcalc.ccp4'
        else:
            if topo_fname is not None:
                raise AttributeError('Cannot specify both pdb and topology/trajectory. Pick one.')
            self.univ = md.Universe(pdb_fname)
            self._initialize_md(sel_string)
            if self.out_fname is None:
                self.out_fname = op.splitext(pdb_fname)[0] + '_diffcalc.h5' #ccp4'
        print('Initialized MD universe')

    def _initialize_md(self, sel_string):
        self.atoms = self.univ.select_atoms(sel_string)
        #self.elem = cp.array([a.name[0] for a in self.atoms])
        #atom_types = cp.unique(self.elem)

        # This hack works for low-Z atoms (mass = 2Z)
        self.atom_f0 = (cp.array([np.around(a.mass) for a in self.atoms]) / 2.).astype('f4')
        self.atom_f0[self.atom_f0 == 0.5] = 1. # Hydrogens

        # B_sol filter for support mask
        cen = self.size//2
        ind = cp.linspace(-cen, cen, self.size)
        x, y, z = cp.meshgrid(ind, ind, ind, indexing='ij')
        q = cp.sqrt(x*x + y*y + z*z) / cen / self.res_edge
        # 30 A^2 B_sol
        self.b_sol_filt = cp.fft.ifftshift(cp.exp(-30 * q * q))

    def gen_dens(self, ind):
        dens = cp.zeros(3*(self.size,), dtype='f4')

        # Get positions of atoms in this frame in centered voxels
        _ = self.univ.trajectory[ind]
        pos = cp.array(self.atoms.positions)
        pos -= cp.array(self.atoms.center_of_mass().astype('f4'))
        pos *= 2. / self.res_edge
        pos += self.size // 2

        # Interpolate into 3D array
        self.k_gen_dens((self.atoms.n_atoms//32+1,), (32,),
            (pos, self.atom_f0, self.atoms.n_atoms, self.size, dens))

        # Solvent mask filtering
        mask = (dens<0.2).astype('f4')
        mask = cp.real(cp.fft.ifftn(cp.fft.fftn(mask)*self.b_sol_filt)) - 1
        dens += mask

        return dens

    def run(self, num_frames=-1, first_frame=0, frame_stride=1, init=True):
        self.rbd.cella = self.size * self.res_edge / 2.
        if num_frames == -1:
            num_frames = len(self.univ.trajectory) - first_frame
        print('Calculating diffuse intensities from %d frames' % num_frames)

        for i in range(first_frame, num_frames + first_frame, frame_stride):
            self.rbd.dens = cp.copy(self.gen_dens(i))
            self.rbd.fdens = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.rbd.dens)))

            if init:
                do_translate = (self.cov_vox.sum() != 0.)
                self.rbd.initialize(translate=do_translate)
                init = False
            prefix = 'Frame %d/%d: ' % (i, num_frames + first_frame)
            self.rbd.run_mc(self.num_steps, self.sigma_deg, self.cov_vox, prefix=prefix)
        sys.stderr.write('\n')

        print('Saving output to', self.out_fname)
        self.rbd.save(self.out_fname)

    def run_cc(self, num_frames=-1, first_frame=0, frame_stride=1):
        if num_frames + first_frame > len(self.univ.trajectory) or num_frames == -1:
            num_frames = len(self.univ.trajectory) - first_frame
        print('Calculating displacement CC for %d frames' % num_frames)

        #ca_atoms = self.univ.select_atoms('name CA')
        #ca_atoms = self.univ.select_atoms('protein and not (name H*)')
        num_atoms = self.atoms.n_atoms
        print('Position of %d C-alpha atoms will be used' % num_atoms)

        corr = cp.zeros((6, num_atoms, num_atoms))
        mean_pos = cp.zeros((num_atoms, 3))

        print('Calculating mean position over selected frames')
        for i in range(first_frame, num_frames + first_frame, frame_stride):
            _ = self.univ.trajectory[i]
            mean_pos += cp.array(self.atoms.positions)
            sys.stderr.write('\rFrame %d'%i)
        sys.stderr.write('\n')
        mean_pos /= (num_frames/frame_stride)

        print('Calculating displacement CCs')
        for i in range(first_frame, num_frames + first_frame, frame_stride):
            _ = self.univ.trajectory[i]
            pos = cp.array(self.atoms.positions) - mean_pos
            pos = (pos.T * self.atom_f0).T # F-weighting the displacements
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
            hcorr = corr.get()
            hf0 = self.atom_f0.get()

            fptr['corr'] = hcorr
            fptr['dist'] = dist
            fptr['mean_pos'] = mean_pos
            fptr['f0'] = hf0 # Atomic scattering factors

def main():
    parser = argparse.ArgumentParser(description='Generate diffuse intensities from rigid body motion')
    parser.add_argument('-c', '--config', help='Config file. Default: config.ini', default='config_traj.ini')
    parser.add_argument('-n', '--num_frames', help='Number of frames to process. Default: -1 (all)', type=int, default=-1)
    parser.add_argument('-f', '--first_frame', help='Index of first frame. Default: 0', type=int, default=0)
    parser.add_argument('-s', '--frame_stride', help='Stride length for frames. Default: 1', type=int, default=1)
    parser.add_argument('-d', '--device', help='GPU device ID (if applicable). Default: 0', type=int, default=0)
    parser.add_argument('-C', '--cov', help='Calculate displacement covariance instead of diffuse intensities. Default=False', action='store_true')
    args = parser.parse_args()

    cp.cuda.Device(args.device).use()

    trajdiff = TrajectoryDiffuse(args.config)
    if args.cov:
        trajdiff.run_cc(args.num_frames, first_frame=args.first_frame, frame_stride=args.frame_stride)
    else:
        trajdiff.run(args.num_frames, first_frame=args.first_frame, frame_stride=args.frame_stride)

if __name__ == '__main__':
    main()
