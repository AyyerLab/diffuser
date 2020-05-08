import configparser
import numpy as np
import cupy as cp
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
        #noise = 51209.805 / 1.e5 / 3
        self.num_vecs = conf.getint('optimizer', 'num_vecs')
        self.pcd = pcdiff.PCDiffuse(config_file)
        self.pcd.num_steps = self.num_steps
        self.pcd.num_vecs = self.num_vecs
        self.pcd.cov_weights = cp.identity(self.num_vecs)
        self.pcd.vecs = self.pcd.vecs[:,:self.num_vecs]
        self.radsel = self.get_radsel(401, 10, 200)

        self.output_fname = conf.get('optimizer', 'output_fname')
        diag_bounds = tuple([float(s) for s in conf.get('optimizer', 'diag_bounds', fallback='0 0').split()])
        offdiag_bounds = tuple([float(s) for s in conf.get('optimizer', 'offdiag_bounds', fallback='0 0').split()])
        self.dims = []
        self.dims_code = 3 # Both diagonal and off-diagonal
        self.get_dims(diag_bounds, offdiag_bounds)

    def optimize(self, num_iter, resume=False, n_initial_points=10, **kwargs):
        if resume:
            cres = skopt.load(self.output_fname)
            x0 = cres.x_iters
            y0 = cres.func_vals
        else:
            x0 = None
            y0 = None

        checkpoint_saver = CheckpointSaver(self.output_fname, store_objective=False)
        res = forest_minimize(self.obj_fun,
            self.dims,
            n_calls = num_iter,
            n_random_starts = n_initial_points,
            callback = [checkpoint_saver],
            verbose = True,
            x0 = x0,
            y0 = y0,
            **kwargs
            )

    def get_dims(self, db, odb):
        if db[1] - db[0] == 0 and odb[1] - odb[0] == 0:
            raise ValueError('Need either diag_bounds or offdiag_bounds')
        elif db[1] - db[0] == 0:
            self.dims_code -= 1
        elif odb[1] - odb[0] == 0:
            self.dims_code -= 2

        for i in range(self.num_vecs):
            for j in range(i+1):
                if i == j and self.dims_code & 1 != 0:
                    self.dims += [skopt.space.Real(*db)]
                elif self.dims_code & 2 != 0:
                    self.dims += [skopt.space.Real(*odb)]
        print(len(self.dims), 'dimensional optimization')

    def get_mc_intens(self, s):
        '''Get MC diffuse intensities for given 's' parameter'''
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
        self.pcd.run_mc()
        return self.pcd.diff_intens

    def obj_fun(self, s):
        '''Calcuates L2-norm between MC diffuse with given 's' and target diffuse'''
        Imc = self.get_mc_intens(s)
        #return (cp.linalg.norm(Imc.ravel() - Itarget.ravel()).get()).item() / 1.e8
        #return 1. - cp.corrcoef(Imc.ravel()[self.radsel], self.Itarget.ravel()[self.radsel])[0,1].get()
        return 1. - np.corrcoef(Imc.ravel()[self.radsel], self.Itarget.ravel()[self.radsel])[0,1]

    @staticmethod
    def get_radsel(size, rmin, rmax):
        ind = np.arange(size).astype('f8') - size//2
        x, y, z = np.meshgrid(ind, ind, ind, indexing='ij')
        rad = np.sqrt(x*x + y*y + z*z)
        return ((rad>=rmin) & (rad<=rmax)).ravel()

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

