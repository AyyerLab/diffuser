import pcdiff
import h5py
import cupy as cp
import numpy as np

cp.cuda.Device(3).use()
my_class = pcdiff.PCDiffuse('config.ini')

D = list()
X = []
for mode in range(2):   
    my_class.run_mc()
    diff_intens= my_class.diff_intens
    D.append(diff_intens)
    fD = cp.ravel(diff_intens)
    X.append(fD)
diff_total = sum(D)
TD = cp.array(diff_total)
#print(diff_total)
DI = cp.array(D)
fDiff = cp.array(X)



out_fname = 'md295_cov_weights_mc_diff_intens.h5'

with h5py.File('../CypA/xtc/'+out_fname, 'w') as f:
              f.create_dataset('diff_intens', data=TD.get())
              f.create_dataset('diff_intens_mode', data = DI.get())
              f.create_dataset('fdiff_intens', data = fDiff.get())             

