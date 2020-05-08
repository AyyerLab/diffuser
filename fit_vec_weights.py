import pc
import h5py
import cupy as cp
import numpy as np

cp.cuda.Device(1).use()
pcd = pc.PCDiffuse('config.ini')

'''print('Calculating I_mc')
vec_weights_mc = cp.array([4, 3])

pcd.vec_weights = vec_weights_mc
print('vec_weights_mc' + str(pcd.vec_weights))

pcd.run_mc()
diff_intens= pcd.diff_intens
Y_mc= cp.array(diff_intens).ravel()


print('diff_mc'+ str(Y_mc))

out_fname = 'md295_weight_mc_diff_intens_4_3.h5'

with h5py.File('../CypA/xtc/'+out_fname, 'w') as f:
              f['diff_intens'] = diff_intens 
              f['Y_mc'] = Y_mc.get()
print('Done I_mc')'''             




print('Calculate diff_intens using PCs_run_linear')



sigma  = cp.array([3, 4])
print('vec_weights' + str(sigma))

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
#print(diff_total)
DI = cp.array(D)
X = cp.array(X)

print('pc' + str(X))
print('done I_linear')


print(' LLS fit: mc_diff vs linear_diff')

with h5py.File('../CypA/xtc/md295_weight_mc_diff_intens_4_3.h5', 'r') as f:
            Y_mc = cp.array(f['Y_mc'][:])

modemat = cp.dot(cp.linalg.inv(cp.dot(X, X.T)), X)
B = cp.dot(modemat, Y_mc)
print('B' +str(B))

# Getting new vec_weights
vec_weights_new = cp.array(cp.sqrt(B) * sigma)
vec_weights = vec_weights_new
print('vec_weights' +str(vec_weights))

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
    print('pc' + str(X))

    modemat = cp.dot(cp.linalg.inv(cp.dot(X, X.T)), X)
    B = cp.dot(modemat, Y_mc)
    b.append(B)
    print('B' +str(B))
    vec_weights_new = cp.sqrt(B) * vec_weights

    vec_weights = vec_weights_new
    print('new_sigma' + str(vec_weights))
    w.append(vec_weights)

out_fname = 'md295_weight_linear_diff_intens_3_4.h5'

with h5py.File('../CypA/xtc/'+out_fname, 'w') as f:
              f.create_dataset('diff_intens', data=TD.get())
              f['diff_intens_mode'] =DI.get()
              f['B'] = (cp.array(b)).get()
              f['vec_weights'] = (cp.array(w)).get()
