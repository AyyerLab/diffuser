# pylint: disable=too-many-instance-attributes

import argparse
import numpy as np
import cupy as cp
import h5py
import skopt
from skopt.callbacks import CheckpointSaver

from diffuser import TLSDiffuse, DiffuserConfig

class TLSOptimizer():
    '''Class to run Bayesian optimization to get best tlsw parameters
    matching the calculated and target diffuse scattering
    '''
    def __init__(self, config_file, verbose=False):
        self.verbose = verbose
        conf = DiffuserConfig(config_file)
        with h5py.File(conf.get_path('optimizer', 'itarget_fname'), 'r') as fptr:
            self.i_target = fptr['diff_intens'][:]

        # Parse Config
        self.num_steps = conf.getint('optimizer', 'num_steps')
        self.output_fname = conf.get_path('optimizer', 'output_fname')

        self.vib_std_bounds = conf.get_bounds('optimizer', 'vib_std_bounds') # Angstrom
        self.lib_std_bounds = conf.get_bounds('optimizer', 'lib_std_bounds') # Radians
        self.lib_axis_pos_bounds = conf.get_bounds('optimizer', 'lib_axis_pos_bounds') # Angstroms
        self.screw_bounds = conf.get_bounds('optimizer', 'screw_bounds') # Angstroms/radian
        self.do_aniso = conf.getboolean('optimizer', 'calc_anisotropic_cc', fallback=False)
        self.do_weighting = conf.getboolean('optimizer', 'apply_voxel_weighting', fallback=False)

        qrange = conf.get_bounds('optimizer', 'q_range', '0.05 1.')
        self.point_group = conf.get('optimizer', 'point_group', fallback='1')

        # Setup TLSDiff
        self.tlsd = TLSDiffuse(config_file)
        self.tlsd.num_steps = self.num_steps
        self.size = self.tlsd.dgen.size

        if self.point_group not in ['1', '4', '222']:
            raise NotImplementedError('%s point group not implemented' % self.point_group)
        self.dims = self._get_dims()

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
            n_initial_points = 0
            num_iter += len(cres.x_iters)
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

    def _get_dims(self): # pylint: disable=too-many-branches
        '''Get dimensions of TLS optimization
        21 dimensions in total:
            3 = vib_std
            3 = vib_rvec
            3 = lib_std
            3 = lib_rvec
            6 = w_vecs
            3 = screw
        '''
        dims = []
        for i in range(3):
            dims += [skopt.space.Real(*self.vib_std_bounds, name='vib_std_%d'%(i+1))]
        for i in range(3):
            dims += [skopt.space.Real(-np.pi, np.pi, name='vib_rvec_%d'%(i+1))]
        for i in range(3):
            dims += [skopt.space.Real(*self.lib_std_bounds, name='lib_std_%d'%(i+1))]
        for i in range(3):
            dims += [skopt.space.Real(-np.pi, np.pi, name='lib_rvec_%d'%(i+1))]
        for i in range(6):
            dims += [skopt.space.Real(*self.lib_axis_pos_bounds, name='lib_axis_pos_%d'%(i+1))]
        for i in range(3):
            dims += [skopt.space.Real(*self.screw_bounds, name='lib_std_%d'%(i+1))]
        return dims

    def get_mc_intens(self, svec): # pylint: disable=too-many-branches
        '''Get MC diffuse intensities for given s-vector'''
        if self.verbose:
            print('Generating intensity for s =', svec)

        self.tlsd.dgen.tls_vib_std = cp.array(svec[0:3])
        self.tlsd.dgen.tls_vib_rvec = cp.array(svec[3:6])
        self.tlsd.dgen.tls_lib_std = cp.array(svec[6:9])
        self.tlsd.dgen.tls_lib_rvec = cp.array(svec[9:12])
        self.tlsd.dgen.tls_axis_positions = cp.array(svec[12:18])
        self.tlsd.dgen.tls_screws = cp.array(svec[18:21])

        # Calculate MC intensity
        self.tlsd.run_mc()
        i_mc = self.tlsd.diff_intens

        if self.point_group == '222':
            i_mc = 0.25 * (i_mc + i_mc[::-1] + i_mc[:,::-1] + i_mc[:,:,::-1])
        elif self.point_group == '4': # Assuming 4-fold axis is Z (fastest changing)
            i_mc = 0.25 * (i_mc + i_mc[::-1,::-1] + i_mc.transpose(1,0,2)[:,::-1] + i_mc.transpose(1,0,2)[::-1,:])

        return i_mc.get()

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
        qrad = self.tlsd.dgen.qrad
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
    parser.add_argument('-v', '--verbose', help='Verbose output of s-vector for each iterations', action='store_true')
    args = parser.parse_args()

    cp.cuda.Device(args.device).use()

    opt = TLSOptimizer(args.config_file, verbose=args.verbose)
    opt.optimize(args.num_iter, resume=args.resume, n_initial_points=args.initial_points)

if __name__ == '__main__':
    main()
