import pcdiff
import h5py
import cupy as cp
import numpy as np

cp.cuda.Device(2).use()
pcd = pcdiff.PCDiffuse('config.ini')

'''print('Calculating I_mc')
#vec_weights_mc = cp.random.randint(7, size =2)

#vecs_weights_mc= pcd.vec_weights
#print('weights_mc' + str(pcd.vec_weights))

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


out_fname = 'md295_cov_weight_mc_diff_intens.h5'

with h5py.File('../CypA/xtc/'+out_fname, 'w') as f:
              f['diff_intens'] = diff_mc1.get() 
              f['Y_mc'] = Y_mc.get()
              f.create_dataset('diff_intens_mode', data = DI_mc.get())
              f.create_dataset('fdiff_intens', data = fDiff_mc.get())
print('Done I_mc')'''             


#pcd.cov_weights = cp.array([[9, 0], [0, 4]])


with h5py.File('../CypA/xtc/md295_cov_weight_mc_diff_intens_16_9_0.012.h5', 'r') as f:
    Y_mc  = f['Y_mc'][:]
Y_mc = cp.array(Y_mc)



print('Calculate diff_intens using PCs_run_linear')

#with h5py.File('../CypA/xtc/md295_vecs_2B.h5', 'r') as f:
#    sigma = f['sigma'][:]

sigma = cp.array([3,2])
print('sigma' + str(sigma))

#vec_weights_linear = np.full((2,), 1)
#pcd.vec_weights = vec_weights_linear
#print('vec_weights' + str(pcd.vec_weights))

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

print('pcdiff' + str(X))
print('done I_linear')    

'''out_fname = 'md295_linear_diff_intens.h5'

with h5py.File('../CypA/xtc/'+out_fname, 'w') as f:
              f['diff_intens'] = data=TD)
              f.create_dataset('diff_intens_mode', data = DI)'''


print(' LLS fit: mc_diff vs linear_diff')

modemat = cp.dot(cp.linalg.inv(cp.dot(X, X.T)), X)
B = cp.dot(modemat, Y_mc)
print('B' +str(B))
sigma_new = cp.array(cp.sqrt(B) * cp.sqrt(cp.diag(cp.array([[9, 0], [0, 4]]))))

s = cp.array(cp.sqrt(B) * cp.sqrt(cp.diag(cp.diag(cp.array([[9,0], [0, 4]])))))

Cor_new = cp.dot(cp.linalg.inv(s), cp.dot(pcd.cov_weights, cp.linalg.inv(s)))
Cov_weights_new  = cp.dot(s, cp.dot(Cor_new, s))
pcd.cov_weights = Cov_weights_new

print('cor_new'+ str(Cor_new))
print('Cov_weights_new' + str(Cov_weights_new))

vec_weights = sigma_new
print('sigma_new' +str(vec_weights))
'''
b=[]
w=[]
cov =[]
cor =[]

#vec_weights = sigma_new
#print('sigma_new' +str(vec_weights))


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
    vec_weights = cp.array(cp.sqrt(B) * vec_weights )  #cp.sqrt(cp.diag(pcd.cov_weights)))
 
                                                        #pcd.vec_weights = vec_weights
    print('new_sigma' + str(vec_weights))
    w.append(vec_weights)
    s= vec_weights #cp.array(cp.sqrt(B) * cp.sqrt(cp.diag(cp.diag(pcd.cov_weights))))

    Cor_new = cp.dot(cp.linalg.inv(s), cp.dot(pcd.cov_weights, cp.linalg.inv(s)))
    cor.append(Cor_new)
    print('Cor_new'+str(Cor_new))
    
    Cov_weights_new  = cp.dot(s, cp.dot(Cor_new, s))
    pcd.cov_weights = Cov_weights_new
    cov.append(Cov_weights_new)
    print('new cov_weights' + str(Cov_weights_new))
    

out_fname = 'md295_cov_weight_linear_diff_intens3_2_0.h5'

with h5py.File('../CypA/xtc/'+out_fname, 'w') as f:
              f.create_dataset('diff_intens', data=TD.get())
              f['diff_intens_mode'] =DI.get()
              f['B'] = (cp.array(b)).get()
              f['vec_weights'] = (cp.array(w)).get()
              f['cov_weights'] =cp.array(cov).get()
              f['corr'] = cp.array(cor).get()'''
