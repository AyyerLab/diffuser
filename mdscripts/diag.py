import sys
import argparse
import h5py
import numpy as np

def get_allcov(cov):
    allcov = np.zeros((3, cov.shape[1], 3, cov.shape[2]))

    allcov[0, :, 0] = cov[0]
    allcov[0, :, 1] = cov[3].T
    allcov[0, :, 2] = cov[5]
    allcov[1, :, 0] = cov[3]
    allcov[1, :, 1] = cov[1]
    allcov[1, :, 2] = cov[4].T
    allcov[2, :, 0] = cov[5].T
    allcov[2, :, 1] = cov[4]
    allcov[2, :, 2] = cov[2]

    return allcov.reshape(cov.shape[1]*3, cov.shape[2]*3)

def main():
    parser = argparse.ArgumentParser(description='Diagonizing covariance matrix and saving eigenvectors of atomic displacement')
    parser.add_argument('cov_file', help='Path to covariance file')
    parser.add_argument('-o', '--vecs_fname', help='Path to output vecs file (necessary)')
    parser.add_argument('-n', '--num_vecs', help='Number of eigenvectors to save (default: 100)', type=int, default=100)
    args = parser.parse_args()

    if args.vecs_fname is None:
        raise ValueError('Need output filename -o')

    # Loading covariance matrix (6, N, N)
    with h5py.File(args.cov_file, 'r') as fptr:
        allcov = get_allcov(np.array(fptr['corr'][:]))
        f0 = np.array(fptr['f0'][:])

    # Diagonalizing (3N, 3N) matrix
    sys.stderr.write('Diagonalizing...')
    vals, vecs = np.linalg.eigh(allcov)
    sys.stderr.write('done\n')

    # Note that eigenvalues are sorted in INCREASING order with sign
    # To get sorting acc. to absolute value with max first...
    sorter = np.abs(vals).argsort()[::-1]

    # We need to remove the scattering f-weighting from the eigenvectors
    vecs = (vecs.T / np.tile(f0, 3)).T

    # Select first N eigenvectors
    vals_n = vals[sorter[:args.num_vecs]].astype('f4')
    vecs_n = vecs[:, sorter[:args.num_vecs]].astype('f4')

    vec_weights = np.diag(np.ones(args.num_vecs) * 10.)

    # Save to file
    with h5py.File(args.vecs_fname, 'w') as fptr:
        fptr['vecs'] = vecs_n
        fptr['cov_weights'] = vec_weights
        fptr['orig_vals'] = vals_n

if __name__ == '__main__':
    main()
