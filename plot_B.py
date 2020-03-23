import numpy as np 
import cupy as cp
import h5py
import matplotlib.pylab as P


# Dataset for plotting

with h5py.File('../CypA/xtc/md295_cov_weight_linear_diff_intens3_2_22.h5', 'r') as f: 
         B = f['B'][:] 
         sigma =f['vec_weights'][:] 
#         corr = f['corr'][:]
cc = np.loadtxt('/home/mazumdep/diffuser/cc.dat', skiprows =1)

# plotting

P.figure(figsize=(15, 6))

P.subplot(131)
P.plot(B[:, 0], 'r-', label = 'B_1')
P.plot(B[:, 1], 'b-', label = 'B_2')

P.xlabel('Iteration')
P.ylabel('B')
P.legend()

#P.title('sigma[4, 3], 5 iteration lls fit')

P.subplot(132)
P.plot(sigma[:, 0], 'r-', label = 'sigma_1')
P.plot(sigma[:, 1], 'b-', label = 'sigma_2')
P.xlabel('Iteration')
P.ylabel('sigma')
P.legend()

P.title('sigma[3, 2], cov_weights(16, 9, 1.2), 10 iteration lls fit')

P.subplot(133)
P.plot(cc[:, 1], 'k-')
P.xlabel('Radius $\AA$')
P.ylabel('CC')
P.title('CC between I_mc I_linear')

#P.subplot(224)
#P.plot(corr[:, [1][0]][0:, 0], 'b-')
#P.xlabel('Iteration')
#P.ylabel('CCof')
#P.title('CCof')



P.show()



