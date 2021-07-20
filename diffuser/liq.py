import sys
import numpy as np
import cupy as cp
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

        self.abc = tuple([int(_) for _ in self.dgen.config.get('parameters', 'rlatt_vox', fallback='0 0 0').split()])

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
        if self.abc[0] == 0:
            msg = 'Provide rlatt_vox to apply liqlatt (recip. lattice voxel dimensions)'
            raise AttributeError(msg)

        s_sq = (2 * cp.pi * sigma_A * self.dgen.qrad)**2
        shape = self.dgen.qrad.shape
        ncells = (shape[0]//self.abc[0], shape[1]//self.abc[1], shape[2]//self.abc[2])

        n_max = 0
        if self.slimits.max() > 2 * np.pi * sigma_A / self.res_max:
            n_max = np.where(self.slimits > 2. * np.pi * sigma_A / self.res_max)[0][0] + 1

        if n_max == 0:
            #bzone = cp.zeros(self.abc)
            #bzone[self.abc[0]//2, self.abc[1]//2, self.abc[2]//2] = 1
            #return cp.tile(bzone, ncells)
            return cp.ones_like(s_sq)

        liq = cp.zeros_like(s_sq)
        for n in range(1, n_max):
            weight = cp.exp(-s_sq + n * cp.log(s_sq) - float(special.loggamma(n+1)))

            curr_gamma = gamma_A / n
            kernel = 8 * np.pi * curr_gamma**3 / (1 + (2 * np.pi * curr_gamma * self.dgen.qrad)**2)**2
            bzone = kernel.reshape(ncells[0], self.abc[0], ncells[1], self.abc[1], ncells[2], self.abc[2])
            bzone = bzone.sum((0, 2, 4))

            liq += weight * cp.tile(bzone, ncells)
            sys.stderr.write('\rLiquidizing: %d/%d' % (n, n_max-1))
        sys.stderr.write('\n')

        return liq
