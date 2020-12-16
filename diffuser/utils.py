import sys
import numpy as np
import cupy as cp

def norm_cov(cov):
    '''Normalize covariance matrix to get CC matrix'''
    ccmat = np.copy(cov)
    ind = np.arange(cov.shape[-1])
    if len(cov.shape) == 3:
        std = np.sqrt(cov[:3, ind, ind])
        ccmat[0] /= np.outer(std[0], std[0])
        ccmat[1] /= np.outer(std[1], std[1])
        ccmat[2] /= np.outer(std[2], std[2])
        ccmat[3] /= np.outer(std[0], std[1])
        ccmat[4] /= np.outer(std[1], std[2])
        ccmat[5] /= np.outer(std[2], std[0])
    else:
        std = np.sqrt(cov[ind, ind])
        ccmat /= np.outer(std, std)
    return ccmat

def get_allcov(cov):
    '''Convert (3, N, N) cov matrix to (3N, 3N) cov matrix'''
    allcov = np.zeros((3,cov.shape[1],3,cov.shape[2]))
    allcov[0,:,0] = cov[0]
    allcov[0,:,1] = cov[3].T
    allcov[0,:,2] = cov[5]
    allcov[1,:,0] = cov[3]
    allcov[1,:,1] = cov[1]
    allcov[1,:,2] = cov[4].T
    allcov[2,:,0] = cov[5].T
    allcov[2,:,1] = cov[4]
    allcov[2,:,2] = cov[2]
    return allcov.reshape(cov.shape[1]*3, cov.shape[2]*3)

def get_densproj(tdiff, mean_pos, fvec, mtype='traj', size=301):
    '''Get Z-projection of distorted molecule's electron density'''
    if mtype not in ['traj', 'linear', 'raw']:
        raise ValueError('mtype must be one of traj, linear or raw')

    if mtype == 'linear':
        num_frames = 100
    else:
        num_frames = len(tdiff.univ.trajectory)

    densproj = cp.zeros((num_frames, size, size), dtype='f4')
    dens = cp.zeros(3*(size,), dtype='f4')
    tfvec = fvec.reshape(3,-1).T

    for i in range(num_frames):
        if mtype == 'traj':
            _ = tdiff.univ.trajectory[i]
            pos = tdiff.atoms.positions - mean_pos
            dpos = np.dot(pos.T.ravel(), fvec) * tfvec
        elif mtype == 'linear':
            dpos = np.linspace(-20, 20, num_frames)[i] * tfvec
        elif mtype == 'raw':
            _ = tdiff.univ.trajectory[i]
            dpos = tdiff.atoms.positions - mean_pos
        pos = cp.array(mean_pos + dpos).astype('f4')
        pos /= 0.5 # 0.5 A voxel size
        pos += size//2
        dens[:] = 0
        tdiff.k_gen_dens((len(mean_pos)//32+1,), (32,),
                      (pos, tdiff.atom_f0, len(mean_pos), size, dens))
        densproj[i] = dens.sum(2)
    return densproj

def get_projections(tdiff, mean_pos, vec):
    '''Get projections of trajectory along PC vectors'''
    num_frames = len(tdiff.univ.trajectory)
    proj = np.zeros((num_frames, vec.shape[1]))

    for i in range(num_frames):
        _ = tdiff.univ.trajectory[i]
        pos = tdiff.atoms.positions
        pos -= pos.mean(0)
        pos -= mean_pos
        proj[i] = np.dot(pos.T.ravel(), vec)
        sys.stderr.write('\r%d/%d'%(i+1, num_frames))
    sys.stderr.write('\n')
    return proj
