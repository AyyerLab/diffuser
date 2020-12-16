import argparse
import numpy as np
import h5py
import MDAnalysis as md

def main():
    parser = argparse.ArgumentParser(description='Project trajectory along PC')
    parser.add_argument('-s', '--tpr', help='Path to tpr file')
    parser.add_argument('-f', '--xtc', help='Path to xtc file')
    parser.add_argument('-c', '--cov', help='Path to displacement cov file')
    parser.add_argument('-v', '--vecs', help='Path to principle vectors file')
    parser.add_argument('-p', '--proj', help='Path to projected pc file')
    parser.add_argument('-o', '--out_pdb', help='Path to output pdbs file')
    args = parser.parse_args()

    univ = md.Universe(args.tpr, args.xtc)
    protein = univ.select_atoms('protein')

    with h5py.File(args.vecs, 'r') as fptr:
        pcs = fptr['vecs'][:]

    with h5py.File(args.cov, 'r') as fptr:
        mean_pos = fptr['mean_pos'][:]

    with h5py.File(args.proj, 'r') as fptr:
        pc_proj = fptr['pca'][:]

    # Projected coordinate of first principal component
    pc1 = pcs[:, 0]
    trans1 = pc_proj[:, 0]
    coordinates = (np.outer(trans1, pc1) + mean_pos.ravel()).reshape(len(trans1), -1, 3)

    # Create new universe to visualise the movement over the first principal component
    proj1 = md.Merge(protein)
    proj1_new = proj1.load_new(coordinates, order="fac")

    # Save the trajectory as PDB format
    with md.Writer(args.out_pdb, protein.n_atoms) as writer:
        for _ in proj1_new.trajectory:
            writer.write(proj1_new)

if __name__ == '__main__':
    main()
