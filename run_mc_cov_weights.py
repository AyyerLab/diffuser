import pcdiff
import h5py
import cupy as cp
import numpy as np

cp.cuda.Device(2).use()
pcd = pcdiff.PCDiffuse('config.ini')

print('Calculating I_mc')

print('cov_weights_mc' + str(pcd.cov_weights))
Dmc = []
Y= []
for mode in range(2):   
    pcd.run_mc()
    diff_intens= pcd.diff_intens
    Dmc.append(diff_intens)
    fD = cp.ravel(diff_intens)
    Y.append(fD)

diff_mc = sum(Dmc)
diff_mc1 = cp.array(diff_mc)
Y_mc= cp.array(cp.ravel(diff_mc))


print('diff_mc'+ str(Y_mc))
DI_mc = cp.array(Dmc)
fDiff_mc = cp.array(Y)


out_fname = 'md295_cov_weight_mc_diff_intens_16_9_0.012.h5'

with h5py.File('../CypA/xtc/'+out_fname, 'w') as f:
              f['diff_intens'] = diff_mc1.get() 
              f['Y_mc'] = Y_mc.get()
              f.create_dataset('diff_intens_mode', data = DI_mc.get())
              f.create_dataset('fdiff_intens', data = fDiff_mc.get())
print('Done I_mc')             
