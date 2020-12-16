import argparse
import numpy as np
import MDAnalysis as md
import h5py

def main():
    parser = argparse.ArgumentParser(description='Calculate projection of PCs on trajectory')
    parser.add_argument('-s', '--tpr', help='path to tpr file')
    parser.add_argument('-f', '--xtc', help='path to xtc file')
    parser.add_argument('-c', '--cov', help='path to displacement cov file')
    parser.add_argument('-v', '--vecs', help='path to principle vectors file')
    parser.add_argument('-b', '--start', help='Starting frame', type=int, default=0)
    parser.add_argument('-e', '--stop', help='End frame', type=int, default=-1)
    parser.add_argument('-t', '--step', help='Frame step size', type=int, default=1)
    parser.add_argument('-n', '--num_vecs', help='Number of PC vectors to project (default: 20)', type=int, default=20)
    parser.add_argument('-o', '--out_proj', help='Path to output file')
    args = parser.parse_args()

    univ = md.Universe(args.tpr, args.xtc)
    protein = univ.select_atoms('protein')

    with h5py.File(args.vecs, 'r') as fptr:
        pcs = fptr['vecs'][:]

    with h5py.File(args.cov, 'r') as fptr:
        mean_pos = fptr['mean_pos'][:]

    protein = protein.atoms
    traj = protein.universe.trajectory
    start, stop, step = traj.check_slice_indices(start=args.start,
                                                 stop=args.stop,
                                                 step=args.step)
    n_frames = len(range(start, stop, step))
    pc20 = pcs[:, :args.num_vecs]
    pca_proj = np.zeros((n_frames, args.num_vecs))

    for i, _ in enumerate(traj[start:stop:step]):
        xyz = protein.positions.ravel() - mean_pos.ravel()
        pca_proj[i] = np.dot(xyz, pc20)

    with h5py.File(args.out_proj, 'w') as fptr:
        fptr['pca'] = pca_proj
        fptr['time_ns'] = np.arange(len(pca_proj)) * args.step * 10 * 0.001

if __name__ == '__main__':
    main()
