
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
import MDAnalysis as md

import rbdiff

class TrajectoryDiffuse():
    '''Generate diffuse intensities from set of coordinates and rigid-body parameters'''
    def __init__(self, config_file):
        self._parse_config(config_file)
        self.rbd = rbdiff.RBDiffuse(self.rot_plane)

    def _parse_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        # Files
        traj_fname = config.get('files', 'traj_fname', fallback=None)
        topo_fname = config.get('files', 'topo_fname', fallback=None)
        sel_string = config.get('files', 'selection_string', fallback='all')
        pdb_fname = config.get('files', 'pdb_fname', fallback=None)
        self.out_fname = config.get('files', 'out_fname', fallback=None)

        # Parameters
        self.sigma_deg = config.getfloat('parameters', 'sigma_deg', fallback=0.)
        sigma_vox = config.getfloat('parameters', 'sigma_vox', fallback=0.)
        rot_axis = config.getint('parameters', 'rot_axis', fallback=0)
        self.num_steps = config.getint('parameters', 'num_steps')

        self.cov_vox = numpy.identity(3) * sigma_vox**2
        self.rot_plane = tuple(numpy.delete([0,1,2], rot_axis))
        print('Parsed config file')
        if pdb_fname is None:
            if topo_fname is None or traj_fname is None:
                raise AttributeError('Need either pdb_fname or both traj_fname and topo_fname')
            self.univ = md.Universe(topo_fname, traj_fname)
            self._initialize_md(sel_string)
            if self.out_fname is None:
                self.out_fname = op.splitext(traj_fname)[0] + '_diffcalc.ccp4'
        else:
            if topo_fname is not None and traj_fname is not None:
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

        # This hack works for low-Z atoms
        self.atom_f0 = numpy.array([numpy.around(a.mass) for a in self.atoms]) / 2.
        self.atom_f0[self.atom_f0 == 0.5] = 1.
        
    def gen_dens(self, ind):
        dens = numpy.zeros(3*(301,), dtype='f4') # TODO: Generalize

        # Get positions of atoms in this frame
        self.univ.trajectory[ind]
        pos = numpy.array(self.atoms.positions)

        # Convert coordinates to voxels (centered)
        pos -= self.atoms.center_of_mass()
        pos /= 1.5 # 1.5 A voxels TODO: Generalize
        pos += 301 // 2 # 301^3 volume TODO: Generalize

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
        return np.array(dens).astype('f4')

    def run(self, num_frames=-1, init=True):
        self.rbd.cella = 301*1.5 # TODO: Generalize
        if num_frames == -1:
            num_frames = len(self.univ.trajectory)
        print('Calculating diffuse intensities from %d frames' % num_frames)
        
        for i in range(num_frames):
            sys.stderr.write('Frame %d\n'%i)
            self.rbd.dens = np.copy(self.gen_dens(i))
            self.rbd.fdens = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.rbd.dens)))

            if init:
                do_translate = (self.cov_vox.sum() != 0.)
                self.rbd.initialize(translate=do_translate)
                init = False
            self.rbd.run_mc(self.num_steps, self.sigma_deg, self.cov_vox)

        print('Saving output to', self.out_fname)
        self.rbd.save(self.out_fname)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate diffuse intensities from rigid body motion')
    parser.add_argument('-c', '--config', help='Config file. Default: config.ini', default='config.ini')
    parser.add_argument('-n', '--num_frames', help='Number of frames to process. Default: -1 (all)', type=int, default=-1)
    parser.add_argument('-d', '--device', help='GPU device ID (if applicable). Default: 0', type=int, default=0)
    args = parser.parse_args()

    if CUPY:
        np.cuda.Device(args.device).use()

    trajdiff = TrajectoryDiffuse(args.config)
    trajdiff.run(args.num_frames)

if __name__ == '__main__':
    main()
