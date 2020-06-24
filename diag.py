import h5py
import numpy as np
import  utils
import argparse


def main():
    parser = argparse.ArgumentParser(description = 'Diagonizing Covariance matrix and savinf first 100 eigenvector of atomic displacement')
    parser.add_argument('cov_file', help = 'Path to covariance file')
    parser.add_argument('-o', '--vecs_fname', help = 'Path to output vecs file')

    args = parser.parse_args()




    with h5py.File(args.cov_file, 'r') as f:
            cov = f['corr'][:]
            mean_pos = f['mean_pos'][:]
            f0 = f['f0'][:]
    allcov = utils.get_allcov(cov)
    print('Diagonalizing...')
    vals, vecs = np.linalg.eigh(allcov)
    print('done')

    # Note that eigenvalues are sorted in INCREASING order
    sorter = np.abs(vals).argsort()[::-1] # To get sorting acc. to absolute value with max first

    # We need to remove the F-weighting from the eigenvectors
    vecs = (vecs.T / np.tile(f0, 3)).T

    # Select fisrt 100 eigenvector
    vals100 = vals[sorter[:100]].astype('f4')
    vecs100 = vecs[:, sorter[:100]].astype('f4')

    def get_modepos(mode, param):
        return mean_pos + (vecs[:,mode]*param).reshape(3,-1).T


    vec_weights =np.full((100, ), 10)
    

    with h5py.File(args.vecs_fname, 'w') as f:
           f['vecs']  = vecs100
           f['weights'] = vec_weights
           f['orig_vals'] = vals100


if __name__ == '__main__':
    main()
