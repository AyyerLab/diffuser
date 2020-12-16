import numpy as np
import cupy as cp
import MDAnalysis as md
import h5py
import argparse

def main():
    parser = argparse.ArgumentParser(description = 'Genarating Protein mean position pdb')
    parser.add_argument('-s', '--tpr', help = 'path to tpr file')
    parser.add_argument('-f', '--xtc', help = 'path to xtc file')
    parser.add_argument('-c', '--cov', help = 'path to displacement covariance file')
    parser.add_argument('-o', '--out_pdb', help = 'path to output pdb file')

    args = parser.parse_args()

    u = md.Universe(args.tpr, args.xtc) #"data/Lysozyme/lyso_crys.tpr", "data/Lysozyme/lyso295_crys800ns_fit.xtc")

    p = u.select_atoms("protein")
    
    with h5py.File(args.cov, 'r') as f:
            p_avg = f['mean_pos'][:]

    p.positions = p_avg

    p.write(args.out_pdb)

if __name__ == '__main__':
    main()
