import os.path as op
import numpy as np
import h5py
import cupy as cp
import MDAnalysis as md

from diffuser import DiffuserConfig

class DensityGenerator():
    '''Generate density grid from coordinates'''
    def __init__(self, config_file, vecs=False):
        if isinstance(config_file, DiffuserConfig):
            self.config = config_file
        else:
            self.config = DiffuserConfig(config_file)

        with open(op.join(op.dirname(__file__), 'kernels.cu'), 'r') as fptr:
            self.k_gen_dens = cp.RawModule(code=fptr.read()).get_function('gen_dens')
        self.cov_weights = cp.array([0.])

        self.size = self.config.get_size()
        self.qvox = self.config.get_qvox(self.size)
        print('q-space voxel size:\n%s' % self.qvox)
        self.a2vox = cp.array((self.qvox * self.size)).astype('f4')

        self._get_rbparams()
        self._gen_grid()
        self._get_univ()
        if vecs:
            self._get_vecs()

    def _get_rbparams(self):
        self.sigma_deg = self.config.getfloat('parameters', 'sigma_deg', fallback=0.)
        self.sigma_uncorr = self.config.getfloat('parameters', 'sigma_uncorr_A', fallback=0.)

        sigma_vox = self.config.getfloat('parameters', 'sigma_vox', fallback=0.)
        self.cov_vox = np.identity(3) * sigma_vox**2

        rot_axis = self.config.getint('parameters', 'rot_axis', fallback=0)
        self.rot_plane = tuple(np.delete([0,1,2], rot_axis))

    def _gen_grid(self):
        '''Generate qrad array and solvent B-factor filter'''
        cen = self.size // 2

        # Grid
        x, y, z = np.meshgrid(np.arange(self.size[0], dtype='f4') - cen[0],
                              np.arange(self.size[1], dtype='f4') - cen[1],
                              np.arange(self.size[2], dtype='f4') - cen[2],
                              indexing='ij')
        x, y, z = tuple(np.dot(self.qvox, np.array([x, y, z]).reshape(3,-1)).reshape((3,) + x.shape))
        self.qrad = cp.array(np.linalg.norm(np.array([x, y, z]), axis=0))
        self.x = x
        self.y = y
        self.z = z

        # B_sol filter
        b_sol = self.config.getfloat('parameters', 'b_sol_A2', fallback=30.)
        self.b_sol_filt = cp.fft.ifftshift(cp.exp(-b_sol * self.qrad**2))

    def _get_vecs(self):
        vecs_fname = self.config.get_path('files', 'vecs_fname', fallback=None)

        # Get PC vectors
        with h5py.File(vecs_fname, 'r') as fptr:
            self.vecs = cp.array(fptr['vecs'][:].astype('f4'))
            self.cov_weights = cp.array(fptr['cov_weights'][:])

        # Check number of components = 3N
        assert self.vecs.shape[0] == self.avg_pos.size

        # Check number of vectors matches number of weights
        assert self.vecs.shape[1] == self.cov_weights.shape[0]
        self.num_vecs = self.cov_weights.shape[0]
        print('Using %d pricipal-component vectors on %d atoms' % (self.vecs.shape[1], self.vecs.shape[0]//3))

    def _get_univ(self):
        traj_list = self.config.get_traj_list()
        topo_fname = self.config.get_path('files', 'topo_fname', fallback=None)
        pdb_fname = self.config.get_path('files', 'pdb_fname', fallback=None)
        sel_string = self.config.get('files', 'selection_string', fallback='all')

        if pdb_fname is None:
            if topo_fname is None:
                raise AttributeError('Need either pdb_fname or topo_fname (if using trajectories)')
            if traj_list is None:
                raise AttributeError('Need one of traj_fname or traj_flist with topology file')
            self.univ = md.Universe(topo_fname, traj_list)
        else:
            if topo_fname is not None:
                raise AttributeError('Cannot specify both pdb and topology/trajectory. Pick one.')
            self.univ = md.Universe(pdb_fname)

        self.atoms = self.univ.select_atoms(sel_string)
        self.avg_pos = cp.array(self.atoms.positions)
        self.avg_pos -= self.avg_pos.mean(0)

        # This hack works for low-Z atoms (mass = 2Z)
        self.atom_f0 = (cp.array([np.around(a.mass) for a in self.atoms]) / 2.).astype('f4')
        self.atom_f0[self.atom_f0 == 0.5] = 1. # Hydrogens
        print('Initialized MD universe')

    @staticmethod
    def _gen_rotz(angle):
        cang = np.cos(angle)
        sang = np.sin(angle)
        arr = np.array([[cang, -sang, 0.], [sang, cang, 0.], [0., 0., 1.]])
        return cp.array(arr).astype('f4')

    @staticmethod
    def _random_rot():
        qnorm = 2.
        while True:
            quat = np.random.normal(0, 0.1, 4)
            qnorm = np.linalg.norm(quat)
            if qnorm < 1:
                quat /= qnorm
                break
        q = quat
        rotmatrix = np.array([
            [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
            [2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
            [2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]])
        return cp.array(rotmatrix.astype('f4'))

    @staticmethod
    def _random_small_rot(sigma_deg):
        angle = np.random.normal(0, sigma_deg*np.pi/180)
        while True:
            vec = np.random.random(3)
            norm = np.linalg.norm(vec)
            if norm < 1:
                vec /= norm
                break
        tilde = np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
        rotmatrix = (np.cos(angle)*np.identity(3) +
                    np.sin(angle)*tilde +
                    (1. - np.cos(angle))*(np.dot(tilde, tilde) + np.identity(3)))

        return cp.array(rotmatrix.astype('f4'))

    def gen_random_dens(self):
        '''Generate electron density by randomly distorting average molecule

        Applies distortions along principal component vectors followed by
        rigid body translations and rotations
        '''
        # Generate distorted molecule
        if np.linalg.norm(self.cov_weights.get()) > 0.:
            projs = cp.array(np.random.multivariate_normal(np.zeros(self.num_vecs),
                                                           self.cov_weights.get())).astype('f4')
            curr_pos = self.avg_pos + cp.dot(self.vecs, projs).reshape(3, -1).T
        else:
            curr_pos = cp.copy(self.avg_pos)

        # Apply rigid body motions
        curr_pos += cp.array(np.random.multivariate_normal(np.zeros(3), self.cov_vox)).astype('f4')
        curr_pos = cp.dot(curr_pos, self._random_small_rot(self.sigma_deg))

        # Apply uncorrelated displacements
        if self.sigma_uncorr > 0.:
            curr_pos += cp.random.randn(*curr_pos.shape) * self.sigma_uncorr

        return self.calc_dens_pos(curr_pos)

    def gen_proj_dens(self, mode, weight):
        '''Generate electron density by distorting average molecule by given mode and weight

        Applies distortion along specific principal component vector
        No rigid body translations and rotations

        Arguments:
            mode - Mode number
            sigma - Standard deviation of mode weight
        '''
        if self.vecs is None:
            raise ValueError('Need to parse PC-vectors to project along them')

        # Generate distorted molecule
        curr_pos = self.avg_pos + self.vecs[:,mode].reshape(3, -1).T * weight

        return self.calc_dens_pos(curr_pos)

    def gen_frame_dens(self, ind):
        '''Generate electron density of given frame of MD trajectory'''
        _ = self.univ.trajectory[ind]
        pos = cp.array(self.atoms.positions)
        pos -= cp.array(self.atoms.center_of_mass().astype('f4'))

        return self.calc_dens_pos(pos)

    def get_intens(self):
        '''Generate intensity from average structure'''
        dens = self.calc_dens_pos(self.avg_pos)
        fdens = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(dens)))
        return cp.abs(fdens)**2

    def calc_dens_pos(self, curr_pos):
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
