import cupy as cp
import numpy as np
import h5py

'''STD = cp.identity(2)
STD[0:1, :] = cp.array([4, 0])
STD[1, :] = cp.array([0, 3])
print(STD)

Cor = cp.identity(2)
Cor[0:1, :] = cp.array([1, 0.1])
Cor[1, :] = cp.array([0.1, 1])
print(Cor)

cov_weights = cp.dot(STD, cp.dot(Cor, STD))   
print(cov_weights)'''




'''matI = cp.identity(2)
matI[0:1, :] =cp.array([4, 1])
matI[1, :] = cp.array([1, 3])
print(matI)

#cov_weights = matI
cov_weights = cp.dot(matI.T, matI) /2
#x =cp.array([1, 0.01])
#cov = cp.outer(x, x)


print(cov_weights)
#print(cov)'''


cov_weights = cp.array([[8.5, 3.5], [3.5, 5]])

with h5py.File('../CypA/xtc/md295_vecs_2B.h5', 'a') as f:
    del f['cov_weights']
    f['cov_weights']=cov_weights.get()

