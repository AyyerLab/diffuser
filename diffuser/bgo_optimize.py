# pylint: disable=too-many-instance-attributes

import argparse
import numpy as np
import cupy as cp
import h5py
import skopt
from skopt.callbacks import CheckpointSaver

from diffuser import PCDiffuse, DiffuserConfig, Liquidizer

class CovarianceOptimizer():
    '''Class to run Bayesian optimization to get best covariance parameters
    matching the calculated and target diffuse scattering
    '''
    def __init__(self, config_file):
        conf = DiffuserConfig(config_file)
        with h5py.File(conf.get_path('optimizer', 'itarget_fname'), 'r') as fptr:
            self.i_target = fptr['diff_intens'][:]

        # Setup PCDiff...
        self.num_steps = conf.getint('optimizer', 'num_steps')
        self.num_vecs = conf.getint('optimizer', 'num_vecs')
        self.output_fname = conf.get_path('optimizer', 'output_fname')
        intern_fname = conf.get_path('optimizer', 'intern_intens_fname', fallback=None)

        self.diag_bounds = conf.get_bounds('optimizer', 'diag_bounds')
        self.offdiag_bounds = conf.get_bounds('optimizer', 'offdiag_bounds')
        self.llm_sigma_bounds = conf.get_bounds('optimizer', 'LLM_sigma_A_bounds')
        self.llm_gamma_bounds = conf.get_bounds('optimizer', 'LLM_gamma_A_bounds')
        self.uncorr_bounds = conf.get_bounds('optimizer', 'uncorr_sigma_A_bounds')
        self.rbd_bounds = conf.get_bounds('optimizer', 'RBD_sigma_A_bounds')
        self.rbr_bounds = conf.get_bounds('optimizer', 'RBR_sigma_deg_bounds')
        self.do_aniso = conf.getboolean('optimizer', 'calc_anisotropic_cc', fallback=False)
        self.do_weighting = conf.getboolean('optimizer', 'apply_voxel_weighting', fallback=False)
        self.lattice_llm = conf.getboolean('optimizer', 'lattice_llm', fallback=False)

        qrange = conf.get_bounds('optimizer', 'q_range', '0.05 1.')
        self.point_group = conf.get('optimizer', 'point_group', fallback='1')

        self.pcd = PCDiffuse(config_file)
        self.pcd.num_steps = self.num_steps
        self.pcd.dgen.num_vecs = self.num_vecs
        self.pcd.dgen.cov_weights = cp.identity(self.num_vecs)
        self.pcd.dgen.vecs = self.pcd.dgen.vecs[:,:self.num_vecs]
        self.size = self.pcd.dgen.size

        if self.point_group not in ['1', '222']:
            raise ValueError('%s point group not implemented' % self.point_group)
        self.dims = []
        self._get_dims()

        if intern_fname is not None:
            with h5py.File(intern_fname, 'r') as fptr:
                self.i_intern = cp.array(fptr['diff_intens'][:])
        else:
            self.i_intern = None

        if self.dims_code & 32 != 0:
            self.liq = Liquidizer(self.pcd.dgen)

        self.intrad, self.voxmask = self._get_qsel(*qrange)
        self.voxmask &= ~np.isnan(self.i_target)
        self.radcount = None
        if self.do_aniso:
            radavg = self._get_radavg(self.i_target)
            self.i_target -= radavg[self.intrad]

    def optimize(self, num_iter, resume=False, n_initial_points=10, **kwargs):
        '''Run BGO optimizer to find optimal parameters to fit against target diffuse'''
        if resume:
            cres = skopt.load(self.output_fname)
            x_init = cres.x_iters
            y_init = cres.func_vals
        else:
            x_init = None
            y_init = None

        checkpoint_saver = CheckpointSaver(self.output_fname, store_objective=False)
        skopt.gp_minimize(self.obj_fun,
            self.dims,
            n_calls = num_iter,
            n_random_starts = n_initial_points,
            callback = [checkpoint_saver],
            noise = 1e-7,
            verbose = True,
            x0 = x_init,
            y0 = y_init,
            **kwargs
            )

    def _get_dims(self):
        # pylint: disable=too-many-branches
        ddb = self.diag_bounds
        odb = self.offdiag_bounds
        llmsb = self.llm_sigma_bounds
        llmgb = self.llm_gamma_bounds
        ucb = self.uncorr_bounds
        rbd = self.rbd_bounds
        rbr = self.rbr_bounds

        self.dims_code = 0 # No optimization
        if ddb[1] - ddb[0] != 0:
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
                    self.dims += [skopt.space.Real(*ddb, name='X_%d_%d'%(i,j))]
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

    def get_mc_intens(self, svec):
        '''Get MC diffuse intensities for given s-vector'''
        # pylint: disable=too-many-branches
        # Set up MC parameters
        self.pcd.dgen.cov_weights[:] = 0.
        n = 0
        for i in range(self.num_vecs):
            for j in range(i+1):
                if i == j and self.dims_code & 1 != 0:
                    self.pcd.dgen.cov_weights[i, j] = svec[n]**2
                    n += 1
                elif self.dims_code & 2 != 0:
                    self.pcd.dgen.cov_weights[i, j] = svec[n]
                    self.pcd.dgen.cov_weights[j, i] = svec[n]
                    n += 1

        if self.dims_code & 4 != 0:
            self.pcd.dgen.sigma_uncorr = svec[n]
            n += 1
        if self.dims_code & 8 != 0:
            self.pcd.dgen.cov_vox = svec[n]**2 * np.identity(3)
            n += 1
        if self.dims_code & 16 != 0:
            self.pcd.dgen.sigma_deg = svec[n]
            n += 1

        # Calculate MC intensity if needed
        if self.dims_code % 32 != 0:
            self.pcd.run_mc()
            i_mc = self.pcd.diff_intens
        elif self.i_intern is not None:
            i_mc = self.i_intern
        else:
            i_mc = self.pcd.dgen.get_intens()

        # Apply LLM transforms
        if self.dims_code & 32 != 0:
            if self.lattice_llm:
                i_calc = self.liq.liqlatt(svec[-2], svec[1]) * i_mc
            else:
                i_calc = self.liq.liquidize(i_mc, svec[-2], svec[-1])
        else:
            i_calc = i_mc

        if self.point_group == '222':
            i_calc = 0.25 * (i_calc + i_calc[::-1] + i_calc[:,::-1] + i_calc[:,:,::-1])

        return i_calc.get()

    def obj_fun(self, svec):
        '''Calcuates (1 - CC) between calculated diffuse with given s-vector and target diffuse'''
        i_calc = self.get_mc_intens(svec)

        if self.do_aniso:
            radavg = self._get_radavg(i_calc)
            i_calc -= radavg[self.intrad]

        if self.do_weighting:
            cov = np.cov(i_calc[self.voxmask], self.i_target[self.voxmask], aweights=1./self.intrad[self.voxmask]**2)
            retval = 1. - cov[0, 1] / np.sqrt(cov[0,0] * cov[1,1])
        else:
            retval = 1. - np.corrcoef(i_calc[self.voxmask], self.i_target[self.voxmask])[0,1]

        return float(retval)

    def _get_qsel(self, qmin, qmax, binsize=None):
        qrad = self.pcd.dgen.qrad
        if binsize is None:
            binsize = qrad[0,0,0] - qrad[0,0,1]
        binrad = (qrad / binsize).get().astype('i4')
        return binrad, ((qrad >= qmin) & (qrad <= qmax)).get()

    def _get_radavg(self, intens):
        if self.radcount is None:
            self.radcount = np.zeros(self.intrad.max() + 1)
            np.add.at(self.radcount, self.intrad[self.voxmask], 1)
            self.radcount[self.radcount == 0] = 1
        radavg = np.zeros_like(self.radcount)
        np.add.at(radavg, self.intrad[self.voxmask], intens[self.voxmask])
        return radavg / self.radcount

def main():
    '''Run as console script with given config file and number of iterations'''
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

if __name__ == '__main__':
    main()
