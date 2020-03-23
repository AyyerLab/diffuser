import traj_diff
import h5py
import cupy as cp

cp.cuda.Device(2).use()

my_class = traj_diff.TrajectoryDiffuse('config.ini')

my_class.run

diff_intens=my_class.diff_intens 

out_fname = 'md300_trajdiff_intens.h5'

with h5py.File('../CypA/xtc/'+out_fname, 'w') as f:
              f.create_dataset('diff_intens', data=diff_intens)
