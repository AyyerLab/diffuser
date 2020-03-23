import pcdiff
import h5py
import cupy as cp
import numpy as np

cp.cuda.Device(2).use()
pcd = pcdiff.PCDiffuse('config.ini')

#Target diffuse map

with h5py.File('../CypA/xtc/md295_cov_weight_mc_diff_intens_16_9_1.2.h5', 'r') as f:
    Y_mc  = f['Y_mc'][:]
Y_mc = cp.array(Y_mc)


#Calculated diffuse map

print('Calculate diff_intens using PCs_run_linear')
sigma = cp.array([3, 2])
print('sigma' + str(sigma))

D = []
X= []
for mode in range(2):   
    pcd.run_linear(mode, sigma[mode])
    diff_intens= pcd.diff_intens
    D.append(diff_intens)
    fD = np.ravel(diff_intens)
    X.append(fD)
diff_total = sum(D)
TD = cp.array(diff_total)
DI = cp.array(D)
X = cp.array(X)

print('pcdiff' + str(X))
print('done I_linear')    


# Liner least square fit between the target and calculated map

print(' LLS fit: mc_diff vs linear_diff')

modemat = cp.dot(cp.linalg.inv(cp.dot(X, X.T)), X)
B = cp.dot(modemat, Y_mc)
print('B' +str(B))

sigma_new = cp.array(cp.sqrt(B) * sigma ) 
vec_weights = sigma_new
print('sigma_new' +str(vec_weights))

b=[]
w=[]


for i in range (10):
    D = []
    X= []
    for mode in range(2):   
        pcd.run_linear(mode, vec_weights[mode])
        diff_intens= pcd.diff_intens
        D.append(diff_intens)
        fD = cp.ravel(diff_intens)
        X.append(fD)
    diff_total = sum(D)
    TD = cp.array(diff_total)
    #print(diff_total)
    DI = cp.array(D)
    X = cp.array(X)
    print('pcdiff' + str(X))

    modemat = cp.dot(cp.linalg.inv(cp.dot(X, X.T)), X)
    B = cp.dot(modemat, Y_mc)
    b.append(B)
    print('B' +str(B))
    vec_weights = cp.array(cp.sqrt(B) * vec_weights) 
 
                                                     
    print('new_sigma' + str(vec_weights))
    w.append(vec_weights)


out_fname = 'md295_cov_weight_linear_diff_intens3_2_22.h5'

with h5py.File('../CypA/xtc/'+out_fname, 'w') as f:
              f.create_dataset('diff_intens', data=TD.get())
              f['diff_intens_mode'] =DI.get()
              f['B'] = (cp.array(b)).get()
              f['vec_weights'] = (cp.array(w)).get()
