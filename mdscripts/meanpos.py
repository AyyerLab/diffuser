import argparse
import h5py
import MDAnalysis as md

def main():
    parser = argparse.ArgumentParser(description='Generate Protein mean position pdb')
    parser.add_argument('-s', '--tpr', help='Path to tpr file')
    parser.add_argument('-f', '--xtc', help='Path to xtc file')
    parser.add_argument('-c', '--cov', help='Path to displacement covariance file')
    parser.add_argument('-o', '--out_pdb', help='Path to output pdb file')
    args = parser.parse_args()

    univ = md.Universe(args.tpr, args.xtc)
    atoms = univ.select_atoms("protein")

    with h5py.File(args.cov, 'r') as fptr:
        p_avg = fptr['mean_pos'][:]

    atoms.positions = p_avg
    atoms.write(args.out_pdb)

if __name__ == '__main__':
    main()
