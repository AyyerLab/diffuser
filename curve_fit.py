import numpy as np 
import cupy as cp
import scipy
from scipy.optimize import curve_fit
import h5py


with h5py.File('../CypA/xtc/md295_cov_weight_mc_diff_intens_8_4_0035.h5', 'r') as f:
     Y_mc = f['Y_mc'][:]
#Y_mc = cp.array(Y_mc)



#with h5py.File('../CypA/xtc/md295_cov_weights_linear_diff_intens.h5', 'r') as f:
#    X = f['diff_intens'][:].ravel()
#X = cp.asarray(X)


#def line(X, m, n):
#    return X*m+n

#pcov = np.polyfit(X, Y_mc, 3, cov = True)
#print(pcov)
#print(popt)






