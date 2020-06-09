import sys
import configparser
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as cndimage
from scipy import special
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

        self.size = conf.getint('parameters', 'size')
        self.res_edge_A = conf.getfloat('parameters', 'res_edge')
        self.output_fname = conf.get('optimizer', 'output_fname')
        self.diag_bounds = tuple([float(s) for s in conf.get('optimizer', 'diag_bounds', fallback='0 0').split()])
        self.offdiag_bounds = tuple([float(s) for s in conf.get('optimizer', 'offdiag_bounds', fallback='0 0').split()])
        self.llm_sigma_bounds = tuple([float(s) for s in conf.get('optimizer', 'LLM_sigma_A_bounds', fallback = '0 0').split()])
        self.llm_gamma_bounds = tuple([float(s) for s in conf.get('optimizer', 'LLM_gamma_A_bounds', fallback = '0 0').split()])
        self.uncorr_bounds = tuple([float(s) for s in conf.get('optimizer', 'uncorr_sigma_A_bounds', fallback = '0 0').split()])
        self.do_aniso = conf.getboolean('optimizer', 'calc_anisotropic_cc', fallback=False)
        self.do_weighting = conf.getboolean('optimizer', 'apply_voxel_weighting', fallback=False)

        self.dims = []
        self.get_dims()

        self.intrad, self.radsel = self.get_radsel(self.size, 10, self.size // 4)
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
        if llmsb[1] - llmsb[0] != 0 and llmgb[1] - llmgb[0] !=0: 
            self.dims_code += 8
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
            self.dims += [skopt.space.Real(*llmsb, name ='LLM_sigma_A')]
            self.dims += [skopt.space.Real(*llmgb, name ='LLM_gamma_A')]
        
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

        self.pcd.run_mc()
        return cp.array(self.pcd.diff_intens)

    def liquidize(self, intens, sigma_A, gamma_A):
        cen = self.size // 2
        ind = cp.arange(self.size).astype('f8') - cen
        x, y, z = cp.meshgrid(ind, ind, ind, indexing='ij')
        nrad = cp.sqrt(x*x + y*y + z*z) / cen

        q_Ainv = cp.array(nrad / self.res_edge_A)
        s_sq = (2. * cp.pi * sigma_A * q_Ainv)**2
        patt = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(intens)))

        liq = cp.zeros_like(intens)
        slimits = np.array([np.real(np.sqrt(special.lambertw(-(1.e-3 * special.factorial(n))**(1./n) / n, k=0)) * np.sqrt(n) * -1j) for n in range(1,150)])
        n_max = np.where(slimits > 2. * np.pi * sigma_A / self.res_edge_A)[0][0] + 1
        for n in range(n_max):
            kernel = cp.exp(-n * self.res_edge_A * cen * nrad / gamma_A)
            liq += cp.exp(-s_sq) * s_sq**n / float(special.factorial(n)) * cp.abs(cp.fft.fftshift(cp.fft.ifftn(patt * kernel)))
            sys.stderr.write('\rLiquidizing: %d/%d' % (n+1, n_max))
        sys.stderr.write('\n')

        return liq

    def obj_fun (self, s):
        '''Calcuates L2-norm between MC diffuse with given 's' and target diffuse'''
        #Imc = self.get_mc_intens(s).get()

        if self.dims_code & 8 != 0:
            Imc = self.get_mc_intens(s[:-2])
            Icalc = self.liquidize(Imc, s[-2], s[-1]).get()
        else:
            Icalc = self.get_mc_intens(s) 
            
        if self.do_aniso:
            radavg = self.get_radavg(Icalc)
            Icalc -= radavg[self.intrad]

        if self.do_weighting:
            cov = np.cov(Icalc[self.radsel], self.Itarget[self.radsel], aweights=1./self.intrad[self.radsel]**2)
            retval = 1. - cov[0, 1] / np.sqrt(cov[0,0] * cov[1,1])
        else:
            retval = 1. - np.corrcoef(Icalc[self.radsel], self.Itarget[self.radsel])[0,1]

        return float(retval)

    @staticmethod
    def get_radsel(size, rmin, rmax):
        ind = np.arange(size).astype('f8') - size//2
        x, y, z = np.meshgrid(ind, ind, ind, indexing='ij')
        rad = np.sqrt(x*x + y*y + z*z).astype('i4')
        return rad, ((rad>=rmin) & (rad<=rmax))

    def get_radavg(self, intens):
        if self.radcount is None:
            self.radcount = np.zeros(self.intrad.max() + 1)
            np.add.at(self.radcount, self.intrad, 1)
        radavg = np.zeros_like(self.radcount)
        np.add.at(radavg, self.intrad, intens)
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

