import sys
import numpy as np
import cupy as cp
import h5py
from scipy import special

class Liquidizer():
    '''Apply Liquid-like Motion (LLM) distortion to intensity volume'''
    def __init__(self, dens_gen):
        self.dgen = dens_gen
        cen = self.dgen.size // 2

        # Maximum res_edge for LLM limits
        self.res_max = float(1. / max(self.dgen.qrad[0, cen[1], cen[2]],
                                      self.dgen.qrad[cen[0], 0, cen[2]],
                                      self.dgen.qrad[cen[0], cen[1], 0]))

        # u-vectors for LLM
        uvox = np.linalg.inv(self.dgen.qvox.T) / self.dgen.size
        x, y, z = self.dgen.x, self.dgen.y, self.dgen.z
        self.urad = cp.array(np.linalg.norm(np.dot(uvox, np.array([x, y, z]).reshape(3,-1)),
                                            axis=0).reshape(x.shape))

        rlatt_fname = self.dgen.config.get_path('files', 'rlatt_fname', fallback=None)

        # Get reciprocal lattice if defined
        if rlatt_fname is None:
            self.rlatt = None
            self.lpatt = None
        else:
            with h5py.File(rlatt_fname, 'r') as fptr:
                self.rlatt = cp.array(fptr['diff_intens'][:])
            self.lpatt = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.rlatt)))

        self.slimits = [np.real(np.sqrt(special.lambertw(-(1.e-3 * special.factorial(n))**(1./n) / n,
                                                         k=0)) *
                                        np.sqrt(n) * -1j)
                        for n in range(1, 150)]
        self.slimits = np.array(self.slimits)

    def liquidize(self, intens, sigma_A, gamma_A):
        '''Apply liquidization transform on given intensity'''
        s_sq = (2. * cp.pi * sigma_A * self.dgen.qrad)**2
        patt = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(intens)))

        if self.slimits.max() > 2. * np.pi * sigma_A / self.res_max:
            n_max = np.where(self.slimits > 2. * np.pi * sigma_A / self.res_max)[0][0] + 1
        else:
            print('No effect of liquid-like motions with these parameters')
            return intens

        liq = cp.zeros_like(intens)
        for n in range(n_max):
            kernel = cp.exp(-n * self.urad / gamma_A)
            weight = cp.exp(-s_sq + n*cp.log(s_sq) - float(special.loggamma(n+1)))
            liq += weight * cp.abs(cp.fft.fftshift(cp.fft.ifftn(patt * kernel)))
            sys.stderr.write('\rLiquidizing: %d/%d' % (n+1, n_max))
        sys.stderr.write('\n')

        return liq

    def liqlatt(self, sigma_A, gamma_A):
        '''Apply liquidization transform to reciprocal lattice'''
        if self.rlatt is None:
            raise AttributeError('Provide rlatt to apply liqlatt')
        s_sq = (2 * cp.pi * sigma_A * self.dgen.qrad)**2

        if self.slimits.max() > 2 * np.pi * sigma_A / self.res_max:
            n_max = np.where(self.slimits > 2. * np.pi * sigma_A / self.res_max)[0][0] + 1
        else:
            return self.rlatt

        if n_max == 0:
            return cp.ones_like(self.rlatt)

        liq = cp.zeros_like(self.rlatt)
        for n in range(1, n_max):
            weight = cp.exp(-s_sq + n * cp.log(s_sq) - float(special.loggamma(n+1)))
            kernel = cp.exp(-n * self.urad / gamma_A)
            liq += weight * cp.abs(cp.fft.fftshift(cp.fft.ifftn(self.lpatt * kernel)))
            sys.stderr.write('\rLiquidizing: %d/%d' % (n, n_max-1))
        sys.stderr.write('\n')

        return liq
