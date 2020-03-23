import pcdiff
import h5py
import cupy as cp
import numpy as np

cp.cuda.Device(0).use()
pcd = pcdiff.PCDiffuse('config.ini')
sigma0 = [4, 3]
sigma =cp.array(sigma0)
print(sigma)
D = list()
X = []
for mode in range(2):   
    pcd.run_linear(mode, sigma[mode])
    diff_intens= pcd.diff_intens
    D.append(diff_intens)
    fD = np.ravel(diff_intens)
    X.append(fD)
diff_total = sum(D)
TD = np.array(diff_total)
#print(diff_total)
DI = np.array(D)
fDiff = np.array(X)



out_fname = 'md295_cov_weights_linear_diff_intens.h5'

with h5py.File('../CypA/xtc/'+out_fname, 'w') as f:
              f.create_dataset('diff_intens', data=TD)
              f.create_dataset('diff_intens_mode', data = DI)
              f.create_dataset('fdiff_intens', data = fDiff)
             

