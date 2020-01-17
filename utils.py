import numpy as np
import pylab as P

def plot_cc(cc):
    P.clf()
    titles = 'xx yy zz xy yz zx'.split()
    for i in range(6):
        s = P.subplot(2,3,i+1)
        im = P.imshow(cc[i], vmin=-1, vmax=1, cmap='coolwarm')
        P.title(r'CC$_{%s}$'%titles[i], fontsize=12)
    P.suptitle('CC matrices', fontsize=14)
    P.subplots_adjust(right=0.88)
    cax = P.gcf().add_axes([0.9,0.125,0.02,0.75])
    P.colorbar(im, cax=cax)
    
def plot_ccpos(cc, mean_pos, ind):
    P.clf()
    titles = 'xx yy zz xy yz zx'.split()
    for i in range(6):
        s = P.subplot(2,3,i+1)
        im = P.scatter(mean_pos[:,0], mean_pos[:,1], 3, c=cc[i,ind], cmap='coolwarm', vmin=-1, vmax=1)
        P.plot(mean_pos[ind,0], mean_pos[ind,1], marker='o', markersize=14, color='green', markerfacecolor='none')
        if i > 2:
            P.xlabel(r'X ($\AA$)')
        if i % 3 == 0:
            P.ylabel(r'Y ($\AA$)')
        P.title(r'CC_${%s}$'%titles[i])
    P.subplots_adjust(right=0.88)
    cax = P.gcf().add_axes([0.9,0.125,0.02,0.75])
    P.colorbar(im, cax=cax)
    
def norm_cov(cov):
    cc = np.copy(cov)
    ind = np.arange(cov.shape[-1])
    if len(cov.shape) == 3:
        std = np.sqrt(cov[:3, ind, ind])
        cc[0] /= np.outer(std[0], std[0])
        cc[1] /= np.outer(std[1], std[1])
        cc[2] /= np.outer(std[2], std[2])
        cc[3] /= np.outer(std[0], std[1])
        cc[4] /= np.outer(std[1], std[2])
        cc[5] /= np.outer(std[2], std[0])
    else:
        std = np.sqrt(cov[ind, ind])
        cc /= np.outer(std, std)
    return cc
    
def get_allcov(cov):
    allcov = np.zeros((3,1092,3,1092))
    allcov[0,:,0] = cov[0]
    allcov[0,:,1] = cov[3].T
    allcov[0,:,2] = cov[5]
    allcov[1,:,0] = cov[3]
    allcov[1,:,1] = cov[1]
    allcov[1,:,2] = cov[4].T
    allcov[2,:,0] = cov[5].T
    allcov[2,:,1] = cov[4]
    allcov[2,:,2] = cov[2]
    return allcov.reshape(1092*3, 1092*3)

def get_densproj(td, mean_pos, fvec, mtype='traj'):
    if mtype not in ['traj', 'linear', 'raw']:
        raise ValueError('mtype must be one of trj, linear or raw')
    num_frames = len(td.univ.trajectory)

    densproj = cp.zeros((num_frames,301,301), dtype='f4')
    dens = cp.zeros(3*(301,), dtype='f4')
    tfvec = fvec.reshape(3,-1).T

    for i in range(num_frames):
        td.univ.trajectory[i]
        pos = td.atoms.positions
        pos -= mean_pos
        if mtype == 'traj':
            dpos = np.dot(pos.T.ravel(), fvec) * tfvec
        elif mtype == 'linear':
            dpos = np.linspace(-20,20,601)[i] * tfvec
        elif mtype == 'raw':
            dpost = pos - mean_pos
        pos = cp.array(mean_pos + dpos).astype('f4')
        pos /= 0.5
        pos += 150
        dens[:] = 0
        td.k_gen_dens((1092//32+1,),(32,),(pos, td.atom_f0, 1092, 301, dens))
        densproj[i] = dens.sum(2)
    return densproj
    
def get_projections(td, mean_pos, vec):
    num_frames = len(td.univ.trajectory)
    proj = np.zeros((num_frames, vec.shape[1]))

    for i in range(num_frames):
        td.univ.trajectory[i]
        pos = td.atoms.positions
        pos -= pos.mean(0)
        pos -= mean_pos
        proj[i] = np.dot(pos.T.ravel(), vec)
        sys.stderr.write('\r%d/%d'%(i+1, num_frames))
    sys.stderr.write('\n')
    return proj

