import argparse
import h5py
import matplotlib.pylab as P
import seaborn as sns

parser = argparse.ArgumentParser(description='Genarating projection of PCs on trajectory')
parser.add_argument('-i', '--proj_fname', help='Path to saved trajectory projections file')
args = parser.parse_args()

with h5py.File(args.proj_fname, 'r') as fptr:
    pca_proj = fptr['pca'][:]
    #time = fptr['time_ns'][:]
print('%d vectors projected in %s' % (pca_proj.shape[1], args.proj_fname))

P.figure(figsize=(6, 18))
#P.suptitle('Pos_projection_pc v/s time')
#ax=P.axes(xlim=(0, 1000))

for i in range(20):
    P.subplot(10, 2, i+1)
    #P.plot(time, pca_proj[:,i], label = 'PC'+str(i+1))
    #P.xlabel('time (ns)')
    sns.distplot(pca_proj[:, i], hist=False, label='PC' + str(i+1))
    P.legend()
#P.xlabel('time (ns)')
#P.ylabel('PC_projection')
P.show()
