import pcdiff
import h5py
import cupy as cp
import numpy as np

cp.cuda.Device(3).use()

num_vecs = 5
target_fname = '/home/mazumdep/CypA/xtc/md295_fit_diffcalc.h5'

#Target diffuse map
with h5py.File(target_fname, 'r') as f:
    Y_mc = cp.array(f['diff_intens'][:]).ravel()

pcd = pcdiff.PCDiffuse('md295_cov.ini')
pcd.num_vecs = num_vecs
pcd.cov_weights = pcd.cov_weights[:,:num_vecs]

#Calculated diffuse map
print('Calculate diff_intens using PCs_run_linear')
sigma = cp.ones(num_vecs)

D = []
X= []
for mode in range(num_vecs):
    pcd.run_linear(mode, sigma[mode])
    diff_intens= pcd.diff_intens
    D.append(diff_intens)
    fD = np.ravel(diff_intens)
    X.append(fD)
diff_total = sum(D)
TD = cp.array(diff_total)
DI = cp.array(D)
X = cp.array(X)
print(X.shape)
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
    for mode in range(num_vecs):
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
    #print('pcdiff' + str(X))

    modemat = cp.dot(cp.linalg.inv(cp.dot(X, X.T)), X)
    B = cp.dot(modemat, Y_mc)
    b.append(B)
    print('B' +str(B))
    vec_weights = cp.array(cp.sqrt(B) * vec_weights)


    print('new_sigma' + str(vec_weights))
    w.append(vec_weights)

out_fname = 'data/md295_linear_%d.h5'%(num_vecs)
with h5py.File(out_fname, 'w') as f:
    f['diff_intens'] = TD.get()
    f['diff_intens_mode'] = DI.get()
    f['B'] = cp.array(b).get()
    f['vec_weights'] = cp.array(w).get()
