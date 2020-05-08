import pcdiff
import h5py
import cupy as cp
import numpy as np

cp.cuda.Device(3).use()
pcd = pcdiff.PCDiffuse('config.ini')

print('Calculating I_mc(Itarget)')

cov_weights = cp.array([[14.77059, 0.01628], [0.01628, 7.47768]])
                  
pcd.cov_weights = cov_weights

print('cov_weights_mc' + str(pcd.cov_weights))
   
pcd.run_mc()
diff_intens= cp.array(pcd.diff_intens)
Y_mc= diff_intens.ravel()

print('diff_mc'+ str(Y_mc))


out_fname = 'md295_cov_w_mc_fit_obj_p4_1000step.h5'

with h5py.File('../CypA/xtc/'+out_fname, 'w') as f:
              f['diff_intens'] = diff_intens.get() 
              f['Y_mc'] = Y_mc.get()
              
print('Done I_mc')             
