import cupy as cp
import numpy as np
import MDAnalysis as md
import h5py
import matplotlib.pylab as P
import argparse
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description = 'Genarating porjection of PCs on trajectory')
    parser.add_argument('-P', '--pca', help = 'calculate projection pf PC on trajectory, Default = False', action = 'store_true')
    parser.add_argument('-s', '--tpr', help = 'path to tpr file')
    parser.add_argument('-f', '--xtc', help = 'path to xtc file')
    parser.add_argument('-c', '--cov', help = 'path to displacement cov file')
    parser.add_argument('-v', '--vecs', help = 'path to principle vectors file')
    parser.add_argument('-b', '--start', help = 'Start', type = int, default=0)
    parser.add_argument('-e', '--stop', help = 'stop', type = int, default= -1)
    parser.add_argument('-t', '--step', help = 'step', type = int, default = 1)
    parser.add_argument('-o', '--out_traj', help = 'path to  output trajectory file')
    parser.add_argument('-i', '--pca_proj', help = 'path to saved trajectory file')

    args = parser.parse_args()

    if args.pca:

        u = md.Universe(args.tpr, args.xtc)

        protein = u.select_atoms('protein')

        with h5py.File(args.vecs, 'r') as f:
            pc = f['vecs'][:]


        with h5py.File(args.cov, 'r') as f:
            mean_pos = f['mean_pos'][:]
 
        protein = protein.atoms
        traj = protein.universe.trajectory
        traj_dt = traj.dt
        start, stop, step = traj.check_slice_indices(start = args.start, stop = args.stop, step = args.step)

        n_frames = len(range(start, stop, step))

        pc20 = pc[:, : 20]
        dim = pc20.shape[-1]


        pca_proj = np.zeros((n_frames, dim))

    
        for i, ts in enumerate(traj[start:stop:step]):
            xyz =protein.positions.ravel() - mean_pos.ravel()
            pca_proj[i] =np.dot(xyz, pc20)
        print(pca_proj.shape)
        print(pca_proj)

        with h5py.File(args.out_traj, 'w') as f:
            f['pca'] = pca_proj

        
    else:
    
        with h5py.File(args.pca_proj, 'r') as f:
              pca_proj = f['pca'][:]

        
        
    time = np.arange(len(pca_proj)) * args.step * 10 * 0.001 

    P.figure(figsize=(30, 90))
    #P.suptitle('Pos_projection_pc v/s time')
    #ax=P.axes(xlim=(0, 1000))

    for i in range(20):
        P.subplot(10, 2, i+1)
        #P.plot(time, pca_proj[:,i], label = 'PC'+str(i+1))
        #P.xlabel('time (ns)')
        sns.distplot(pca_proj[:, i], hist = False, label = 'PC' + str(i+1))
        P.legend()
    #P.xlabel('time (ns)')
    #P.ylabel('PC_projection')    
    P.show()

if __name__ == '__main__':
    main()

