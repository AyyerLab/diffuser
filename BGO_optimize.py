import sys
import configparser
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as cndimage
import h5py
import skopt
from skopt import forest_minimize
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
import pcdiff

class CovarianceOptimizer():
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        conf.read(config_file)
        with h5py.File(conf.get('optimizer', 'itarget_fname'), 'r') as f:
            #self.Itarget = cp.array(f['diff_intens'][:])
            self.Itarget = f['diff_intens'][:]

        # Setup PCDiff...
        self.num_steps = conf.getint('optimizer', 'num_steps')
        self.num_vecs = conf.getint('optimizer', 'num_vecs')
        self.pcd = pcdiff.PCDiffuse(config_file)
        self.pcd.num_steps = self.num_steps
        self.pcd.num_vecs = self.num_vecs
        self.pcd.cov_weights = cp.identity(self.num_vecs)
        self.pcd.vecs = self.pcd.vecs[:,:self.num_vecs]
        self.size = self.pcd.size

        self.output_fname = conf.get('optimizer', 'output_fname')
        self.diag_bounds = tuple([float(s) for s in conf.get('optimizer', 'diag_bounds', fallback='0 0').split()])
        self.offdiag_bounds = tuple([float(s) for s in conf.get('optimizer', 'offdiag_bounds', fallback='0 0').split()])
        self.llm_sigma_bounds = tuple([float(s) for s in conf.get('optimizer', 'LLM_sigma_A_bounds', fallback = '0 0').split()])
        self.llm_gamma_bounds = tuple([float(s) for s in conf.get('optimizer', 'LLM_gamma_A_bounds', fallback = '0 0').split()])
        self.uncorr_bounds = tuple([float(s) for s in conf.get('optimizer', 'uncorr_sigma_A_bounds', fallback = '0 0').split()])
        self.rbd_bounds = tuple([float(s) for s in conf.get('optimizer', 'RBD_sigma_A_bounds', fallback = '0 0').split()])
        self.rbr_bounds = tuple([float(s) for s in conf.get('optimizer', 'RBR_sigma_deg_bounds', fallback = '0 0').split()])
        self.do_aniso = conf.getboolean('optimizer', 'calc_anisotropic_cc', fallback=False)
        self.do_weighting = conf.getboolean('optimizer', 'apply_voxel_weighting', fallback=False)
        self.lattice_llm = conf.getboolean('optimizer', 'lattice_llm', fallback=False)
        qrange = tuple([float(s) for s in conf.get('optimizer', 'q_range', fallback = '0.05 1').split()])
        self.point_group = conf.get('optimizer', 'point_group', fallback='1')

        if self.point_group not in ['1', '222']:
            raise ValueError('%s point group not implemented' % self.point_group)
        self.dims = []
        self.get_dims()

        self.intrad, self.voxmask = self.get_qsel(*qrange)
        self.voxmask &= ~np.isnan(self.Itarget)
        self.radcount = None
        if self.do_aniso:
            radavg = self.get_radavg(self.Itarget)
            self.Itarget -= radavg[self.intrad]

    def optimize(self, num_iter, resume=False, n_initial_points=10, **kwargs):
        if resume:
            cres = skopt.load(self.output_fname)
            x0 = cres.x_iters
            y0 = cres.func_vals
        else:
            x0 = None
            y0 = None

        checkpoint_saver = CheckpointSaver(self.output_fname, store_objective=False)
        res = skopt.gp_minimize(self.obj_fun,
            self.dims,
            n_calls = num_iter,
            n_random_starts = n_initial_points,
            callback = [checkpoint_saver],
            noise = 1e-7,
            verbose = True,
            x0 = x0,
            y0 = y0,
            **kwargs
            )

    def get_dims(self):
        db = self.diag_bounds
        odb = self.offdiag_bounds
        llmsb = self.llm_sigma_bounds
        llmgb = self.llm_gamma_bounds
        ucb = self.uncorr_bounds
        rbd = self.rbd_bounds
        rbr = self.rbr_bounds

        self.dims_code = 0 # No optimization
        if  db[1] - db[0] != 0:
            self.dims_code += 1
            print('Optimizing diagonal components')
        if odb[1] - odb[0] != 0:
            self.dims_code += 2
            print('Optimizing off-diagonal components')
        if ucb[1] - ucb[0] != 0:
            self.dims_code += 4
            print('Optimizing uncorrelated variance')
        if rbd[1] - rbd[0] != 0:
            self.dims_code += 8
            print('Optimizing rigid body translations')
        if rbr[1] - rbr[0] != 0:
            self.dims_code += 16
            print('Optimizing rigid body rotations')
        if llmsb[1] - llmsb[0] != 0 and llmgb[1] - llmgb[0] !=0:
            self.dims_code += 32
            print('Optimizing LLM parameters')

        for i in range(self.num_vecs):
            for j in range(i+1):
                if i == j and self.dims_code & 1 != 0:
                    self.dims += [skopt.space.Real(*db, name='X_%d_%d'%(i,j))]
                elif self.dims_code & 2 != 0:
                    self.dims += [skopt.space.Real(*odb, name='X_%d_%d'%(i,j))]

        if self.dims_code & 4 != 0:
            self.dims += [skopt.space.Real(*ucb, name='Uncorr_A')]

        if self.dims_code & 8 != 0:
            self.dims += [skopt.space.Real(*rbd, name='RBD_A')]

        if self.dims_code & 16 != 0:
            self.dims += [skopt.space.Real(*rbr, name='RBR_deg')]

        if self.dims_code & 32 != 0:
            self.dims += [skopt.space.Real(*llmsb, name='LLM_sigma_A')]
            self.dims += [skopt.space.Real(*llmgb, name='LLM_gamma_A')]

        print(len(self.dims), 'dimensional optimization')

    def get_mc_intens(self, s):
        '''Get MC diffuse intensities for given 's' parameter'''
        #get s values from 5vecs_diag_optimization
        ''' res_5vecs_diag = skopt.load('../CypA/xtc/md295_forest_md_5vecs_diag.pkl')
        sigmas_diag = cp.array(res_5vecs_diag.x)
        self.pcd.cov_weights = self.pcd.cov_weights * sigmas_diag**2
        '''

        self.pcd.cov_weights[:] = 0.
        n = 0
        for i in range(self.num_vecs):
            for j in range(i+1):
                if i == j and self.dims_code & 1 != 0:
                    self.pcd.cov_weights[i, j] = s[n]**2
                    n += 1
                elif self.dims_code & 2 != 0:
                    self.pcd.cov_weights[i, j] = s[n]
                    self.pcd.cov_weights[j, i] = s[n]
                    n += 1

        if self.dims_code & 4 != 0:
            self.pcd.sigma_uncorr_A = s[n]
            n += 1
        if self.dims_code & 8 != 0:
            self.pcd.cov_vox = s[n]**2 * np.identity(3)
            n += 1
        if self.dims_code & 16 != 0:
            self.pcd.sigma_deg = s[n]
            n += 1

        if self.dims_code % 32 != 0:
            self.pcd.run_mc()
            Imc = self.pcd.diff_intens
        else:
            Imc = self.pcd.get_intens()
        if self.dims_code & 32 != 0:
            if self.lattice_llm:
                Icalc = self.pcd.liqlatt(s[-2], s[1]) * cp.array(Imc)
            else:
                Icalc = self.pcd.liquidize(cp.array(Imc), s[-2], s[-1])
        else:
            Icalc = Imc

        if self.point_group == '222':
            Icalc = 0.25 * (Icalc + Icalc[::-1] + Icalc[:,::-1] + Icalc[:,:,::-1])

        return Icalc.get()

    def obj_fun (self, s):
        '''Calcuates L2-norm between MC diffuse with given 's' and target diffuse'''
        Icalc = self.get_mc_intens(s)

        if self.do_aniso:
            radavg = self.get_radavg(Icalc)
            Icalc -= radavg[self.intrad]

        if self.do_weighting:
            cov = np.cov(Icalc[self.voxmask], self.Itarget[self.voxmask], aweights=1./self.intrad[self.voxmask]**2)
            retval = 1. - cov[0, 1] / np.sqrt(cov[0,0] * cov[1,1])
        else:
            retval = 1. - np.corrcoef(Icalc[self.voxmask], self.Itarget[self.voxmask])[0,1]

        return float(retval)

    def get_qsel(self, qmin, qmax):
        cen = self.size // 2
        num_bins = self.pcd
        binsize = self.pcd.qrad[0,0,0] - self.pcd.qrad[0,0,1]
        binrad = (self.pcd.qrad / binsize).get().astype('i4')
        return binrad, ((self.pcd.qrad >= qmin) & (self.pcd.qrad <= qmax)).get()

    def get_radavg(self, intens):
        if self.radcount is None:
            self.radcount = np.zeros(self.intrad.max() + 1)
            np.add.at(self.radcount, self.intrad[self.voxmask], 1)
            self.radcount[self.radcount == 0] = 1
        radavg = np.zeros_like(self.radcount)
        np.add.at(radavg, self.intrad[self.voxmask], intens[self.voxmask])
        return radavg / self.radcount

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Bayesian optimization of modes covariance matrix')
    parser.add_argument('config_file', help='Path to config file')
    parser.add_argument('num_iter', help='Number of iterations', type=int)
    parser.add_argument('-d', '--device', help='GPU device number', type=int, default=0)
    parser.add_argument('-R', '--resume', help='Start from output file', action='store_true')
    parser.add_argument('-i', '--initial_points', help='Number of initial random points', type=int, default=10)
    args = parser.parse_args()

    cp.cuda.Device(args.device).use()

    opt = CovarianceOptimizer(args.config_file)
    opt.optimize(args.num_iter, resume=args.resume, n_initial_points=args.initial_points)

