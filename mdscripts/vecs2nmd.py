import os
import argparse
import csv
import numpy as np
import h5py
import MDAnalysis as md

def main():
    parser = argparse.ArgumentParser(description='Convert vecs h5 file to NMD file for visualization')
    parser.add_argument('vecs_fname', help='Path to vecs file (h5 format)')
    parser.add_argument('pdb_fname', help='Path to PDB file containing average structure')
    parser.add_argument('-n', '--num_vecs', help='Number of vectors to write (default: all)', type=int)
    parser.add_argument('-o', '--output_fname', help='Output filename')
    args = parser.parse_args()

    univ = md.Universe(args.pdb_fname)
    atoms = univ.select_atoms('protein and not altloc B and not altloc C')
    print(atoms.n_atoms, 'atoms selected from PDB file')

    with h5py.File(args.vecs_fname, 'r') as fptr:
        vecs = fptr['vecs'][:, :args.num_vecs]
        weights = np.diagonal(fptr['cov_weights'][:])[:args.num_vecs]
    print('Processing', vecs.shape[1], 'modes')
    args.num_vecs = vecs.shape[1]
    try:
        assert vecs.shape[0] == atoms.n_atoms * 3
    except AssertionError as exc:
        raise AssertionError('Number of atoms of PDB and vecs file do not match. Confirm selection string.') from exc

    if args.output_fname is None:
        args.output_fname = os.path.splitext(args.vecs_fname)[0] + '_%dvecs.nmd' % args.num_vecs

    print('Writing output to', args.output_fname)
    with open(args.output_fname, 'w') as fptr:
        writer = csv.writer(fptr, delimiter=' ')
        writer.writerow(['title', args.vecs_fname])
        writer.writerow(['names'] + list(atoms.names))
        writer.writerow(['resnames'] + list(atoms.resnames))
        writer.writerow(['resnums'] + list(atoms.resids - atoms.resids.min()))
        writer.writerow(['coordinates'] + list(atoms.positions.ravel()))
        for i in range(args.num_vecs):
            writer.writerow(['mode', str(i+1), str(weights[i])] + list(vecs[:, i]))

if __name__ == '__main__':
    main()
