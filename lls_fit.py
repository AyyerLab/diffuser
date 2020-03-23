import numpy as np
import h5py

print('Calculate radius of every voxel')
ind = np.arange(-200,201)
x, y, z = np.meshgrid(ind, ind, ind, indexing='ij')
rad = np.sqrt(x*x + y*y + z*z)
sel = (rad > 50).ravel() # selecting only those voxels with radius greater than 50 voxels
sel1 = (rad < 50).ravel()
sel2 = (rad < 150).ravel()

with h5py.File('../CypA/xtc/md295_mc_diff_intens.h5', 'r') as f:
    Y = f['diff_intens'][:].ravel()


with h5py.File('../CypA/xtc/md295_pc2_llsfitb_diff_intens.h5', 'r') as f:
    X=f['fdiff_intens'][:]

print('Calculate B parameter..')
modemat = np.dot(np.linalg.inv(np.dot(X, X.T)), X)
B =np.dot(modemat, Y)
print(B)
print(np.sqrt(B))

selX = X[:, sel]
sel_modemat = np.dot(np.linalg.inv(np.dot(selX, selX.T)), selX)
selB =np.dot(sel_modemat, Y[sel])

sel1X = X[:, sel1]
sel1_modemat = np.dot(np.linalg.inv(np.dot(sel1X, sel1X.T)), sel1X)
sel1B =np.dot(sel1_modemat, Y[sel1])


sel2X = X[:, sel2]
sel2_modemat = np.dot(np.linalg.inv(np.dot(sel2X, sel2X.T)), sel2X)
sel2B =np.dot(sel2_modemat, Y[sel2])

print('LLS fit: MD_dif vs pc_diff')

diff_fit = np.dot(X.T, B).reshape(3*(401,))

diff_fit_sel = np.dot(X.T, selB).reshape(3*(401,))

diff_fit_sel1 = np.dot(X.T, sel1B).reshape(3*(401,))

diff_fit_sel2 = np.dot(X.T, sel2B).reshape(3*(401,))

print('saving..')

with h5py.File('../CypA/xtc/'+'md295_llsfit_step2.h5', 'w') as f:
    f['fit_diff_intens'] = diff_fit
    f['B'] = B
    f['fit_diff_intens_sel'] = diff_fit_sel
    f['selB'] = selB
    f['fit_diff_intens_sel1'] = diff_fit_sel1
    f['sel1B'] = sel1B
    f['fit_diff_intens_sel2'] = diff_fit_sel2
    f['selB2'] = sel2B
