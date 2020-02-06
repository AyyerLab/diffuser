import os
import sys
import os.path as op
import configparser
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
import h5py
import MDAnalysis as md

import rbdiff

class TrajectoryDiffuse():
    '''Generate diffuse intensities from set of coordinates and rigid-body parameters'''
    def __init__(self, config_file):
        self._parse_config(config_file)
        self.rbd = rbdiff.RBDiffuse(self.rot_plane)
        if CUPY:
            self._define_kernels()

    def _define_kernels(self):
        self.k_gen_dens = np.RawKernel(r'''
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

        self.cov_vox = numpy.identity(3) * sigma_vox**2
        self.rot_plane = tuple(numpy.delete([0,1,2], rot_axis))
        print('Parsed config file')

        if pdb_fname is None:
            if topo_fname is None:
                raise AttributeError('Need either pdb_fname or topo_fname (if using trajectories)')

            if traj_fname is not None and traj_flist is not None:
                raise AttributeError('Cannot specify both traj_fname and traj_flist. Pick one.')
            elif traj_fname is not None:
                self.univ = md.Universe(topo_fname, traj_fname)
            elif traj_flist is not None:
                with open(traj_flist, 'r') as f:
                    flist = [l.strip() for l in f.readlines()]
                self.univ = md.Universe(topo_fname, flist)
            else:
                raise AttributeError('Need one of traj_fname or traj_flist with topology file')

            self._initialize_md(sel_string)
            if self.out_fname is None:
                self.out_fname = op.splitext(traj_fname)[0] + '_diffcalc.ccp4'
        else:
            if topo_fname is not None:
                raise AttributeError('Cannot specify both pdb and topology/trajectory. Pick one.')
            self.univ = md.Universe(pdb_fname)
            self._initialize_md(sel_string)
            if self.out_fname is None:
                self.out_fname = op.splitext(pdb_fname)[0] + '_diffcalc.ccp4'
        print('Initialized MD universe')

    def _initialize_md(self, sel_string):
        self.atoms = self.univ.select_atoms(sel_string)
        #self.elem = np.array([a.name[0] for a in self.atoms])
        #atom_types = np.unique(self.elem)

        # This hack works for low-Z atoms (mass = 2Z)
        self.atom_f0 = (np.array([numpy.around(a.mass) for a in self.atoms]) / 2.).astype('f4')
        self.atom_f0[self.atom_f0 == 0.5] = 1. # Hydrogens

        # B_sol filter for support mask
        cen = self.size//2
        ind = np.linspace(-cen, cen, self.size)
        x, y, z = np.meshgrid(ind, ind, ind, indexing='ij')
        q = np.sqrt(x*x + y*y + z*z) / cen / self.res_edge
        # 30 A^2 B_sol
        self.b_sol_filt = np.fft.ifftshift(np.exp(-30 * q * q))

    def gen_dens(self, ind):
        dens = np.zeros(3*(self.size,), dtype='f4')

        # Get positions of atoms in this frame in centered voxels
        self.univ.trajectory[ind]
        pos = np.array(self.atoms.positions)
        pos -= np.array(self.atoms.center_of_mass().astype('f4'))
        pos *= 2. / self.res_edge
        pos += self.size // 2

        # Interpolate into 3D array
        if CUPY:
            self.k_gen_dens((self.atoms.n_atoms//32+1,), (32,),
                (pos, self.atom_f0, self.atoms.n_atoms, self.size, dens))
        else:
            ipos = pos.astype('i4')
            fpos = pos - ipos
            cpos = 1 - fpos

            curr_pos = ipos
            numpy.add.at(dens, tuple(curr_pos.T), cpos[:,0]*cpos[:,1]*cpos[:,2]*self.atom_f0)
            curr_pos = ipos + numpy.array([0,0,1])
            numpy.add.at(dens, tuple(curr_pos.T), cpos[:,0]*cpos[:,1]*fpos[:,2]*self.atom_f0)
            curr_pos = ipos + numpy.array([0,1,0])
            numpy.add.at(dens, tuple(curr_pos.T), cpos[:,0]*fpos[:,1]*cpos[:,2]*self.atom_f0)
            curr_pos = ipos + numpy.array([0,1,1])
            numpy.add.at(dens, tuple(curr_pos.T), cpos[:,0]*fpos[:,1]*fpos[:,2]*self.atom_f0)
            curr_pos = ipos + numpy.array([1,0,0])
            numpy.add.at(dens, tuple(curr_pos.T), fpos[:,0]*cpos[:,1]*cpos[:,2]*self.atom_f0)
            curr_pos = ipos + numpy.array([1,0,1])
            numpy.add.at(dens, tuple(curr_pos.T), fpos[:,0]*cpos[:,1]*fpos[:,2]*self.atom_f0)
            curr_pos = ipos + numpy.array([1,1,0])
            numpy.add.at(dens, tuple(curr_pos.T), fpos[:,0]*fpos[:,1]*cpos[:,2]*self.atom_f0)
            curr_pos = ipos + numpy.array([1,1,1])
            numpy.add.at(dens, tuple(curr_pos.T), fpos[:,0]*fpos[:,1]*fpos[:,2]*self.atom_f0)

        # Solvent mask filtering
        mask = (dens<0.2).astype('f4')
        mask = np.real(np.fft.ifftn(np.fft.fftn(mask)*self.b_sol_filt)) - 1
        dens += mask

        '''
        # Set up Gaussian windowing
        sigma = np.sqrt(30 / 4 / np.pi**2) / 1.5 # sigma for 30 A^2 B-factor TODO: Generalize
        winr = int(np.ceil(3*sigma)) # Window radius
        window = np.indices(3*(2*winr+1,), dtype='f4')
        window[0] -= winr
        window[1] -= winr
        window[2] -= winr

        # Place Gaussian for each atom
        for i, p in enumerate(pos):
            sx = window[0] + fpos[i,0]
            sy = window[1] + fpos[i,1]
            sz = window[2] + fpos[i,2]
            dens[ipos[i,0]-winr:ipos[i,0]+winr+1,
                 ipos[i,1]-winr:ipos[i,1]+winr+1,
                 ipos[i,2]-winr:ipos[i,2]+winr+1] += self.atom_f0[i] * np.exp(-(sx**2 + sy**2 + sz**2) / sigma**2 / 2.)
            sys.stderr.write('\rAtom %.6d'%i)
        sys.stderr.write('\n')
        '''
        return dens

    def run(self, num_frames=-1, first_frame=0, frame_stride=1, init=True):
        self.rbd.cella = self.size * self.res_edge / 2.
        if num_frames == -1:
            num_frames = len(self.univ.trajectory) - first_frame
        print('Calculating diffuse intensities from %d frames' % num_frames)

        for i in range(first_frame, num_frames + first_frame, frame_stride):
            self.rbd.dens = np.copy(self.gen_dens(i))
            self.rbd.fdens = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.rbd.dens)))

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

        corr = np.zeros((6, num_atoms, num_atoms))
        mean_pos = np.zeros((num_atoms, 3))

        print('Calculating mean position over selected frames')
        for i in range(first_frame, num_frames + first_frame, frame_stride):
            self.univ.trajectory[i]
            mean_pos += np.array(self.atoms.positions)
            sys.stderr.write('\rFrame %d'%i)
        sys.stderr.write('\n')
        mean_pos /= num_frames

        print('Calculating displacement CCs')
        for i in range(first_frame, num_frames + first_frame, frame_stride):
            self.univ.trajectory[i]
            pos = np.array(self.atoms.positions) - mean_pos
            pos *= self.atom_f0 # F-weighting the displacements
            corr[0] += np.outer(pos[:,0], pos[:,0])
            corr[1] += np.outer(pos[:,1], pos[:,1])
            corr[2] += np.outer(pos[:,2], pos[:,2])
            corr[3] += np.outer(pos[:,0], pos[:,1])
            corr[4] += np.outer(pos[:,1], pos[:,2])
            corr[5] += np.outer(pos[:,2], pos[:,0])
            sys.stderr.write('\rFrame %d'%i)
        sys.stderr.write('\n')

        cc_fname = os.path.splitext(self.out_fname)[0] + '_cov.h5'
        print('Saving covariance matrix to', cc_fname)

        mean_pos -= mean_pos.mean(0)
        if CUPY:
            mean_pos = mean_pos.get()
        dist = numpy.linalg.norm(numpy.subtract.outer(mean_pos, mean_pos)[:,[0,1,2],:,[0,1,2]], axis=0)

        with h5py.File(cc_fname, 'w') as f:
            if CUPY:
                hcorr = corr.get()
                hf0 = self.atom_f0.get()
            else:
                hcorr = corr
                hf0 = self.atom_f0

            f['corr'] = hcorr
            f['dist'] = dist
            f['mean_pos'] = mean_pos
            f['f0'] = hf0 # Atomic scattering factors

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate diffuse intensities from rigid body motion')
    parser.add_argument('-c', '--config', help='Config file. Default: config.ini', default='config.ini')
    parser.add_argument('-n', '--num_frames', help='Number of frames to process. Default: -1 (all)', type=int, default=-1)
    parser.add_argument('-f', '--first_frame', help='Index of first frame. Default: 0', type=int, default=0)
    parser.add_argument('-s', '--frame_stride', help='Stride length for frames. Default: 1', type=int, default=1)
    parser.add_argument('-d', '--device', help='GPU device ID (if applicable). Default: 0', type=int, default=0)
    parser.add_argument('-C', '--cov', help='Calculate displacement covariance instead of diffuse intensities. Default=False', action='store_true')
    args = parser.parse_args()

    if CUPY:
        np.cuda.Device(args.device).use()

    trajdiff = TrajectoryDiffuse(args.config)
    if args.cov:
        trajdiff.run_cc(args.num_frames, first_frame=args.first_frame, frame_stride=args.frame_stride)
    else:
        trajdiff.run(args.num_frames, first_frame=args.first_frame, frame_stride=args.frame_stride)

if __name__ == '__main__':
    main()
