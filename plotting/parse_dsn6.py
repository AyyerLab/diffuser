import numpy as np
from skimage import transform

def parse(fname):
    '''Parse DSN6 file


    Returns:
        voxels, cell
    where cell has parameters describing voxel size and angles
    Each voxel is a parallelopiped with sides given by cell[:3]
    and angles between sides given by cell[3:6] in radians
    '''
    fptr = open(fname, 'rb')
    header = np.fromfile(fptr, dtype='i2', count=256).byteswap()
    data = np.fromfile(fptr, dtype='u1').reshape(-1, 2)[:, ::-1].ravel()
    fptr.close()

    origin = header[:3]
    extent = header[3:6][::-1]
    grid = header[6:9]
    cell = header[9:15] / float(header[17])
    prod = header[15] / header[18]
    plus = header[16]

    nb = np.ceil(extent / 8.).astype('i2') # pylint: disable=C0103
    vol = data.reshape(tuple(nb) + (8, 8, 8))
    vol = vol.transpose(0, 3, 1, 4, 2, 5)
    vol = vol.reshape(nb[0]*8, nb[1]*8, nb[2]*8).astype('f4')
    vol = vol[:extent[0], :extent[1], :extent[2]]
    vol = (vol - plus) / prod

    cell[:3] /= grid
    cell[3:] *= np.pi / 180.

    return vol, cell

def remap(pvol, pcell, size, vox_size):
    '''Remap volume according to cell parameters into target volume

    Parameters:
        pvol - Array to be remapped
        pcell - Box parameters, [a, b, c, alpha, beta, gamma] (output of parse())
        size - Size of 3D cube to be mapped into
        vox_size - Voxel size in remapped volume in Angstroms

    Returns:
        size^3 volume containing remapped voxels
    '''
    x, y, z = np.indices(3 * (size,))
    cen = size // 2
    x -= cen - int(pvol.shape[0] / 2 * pcell[0] / vox_size)
    y -= cen - int(pvol.shape[1] / 2 * pcell[1] / vox_size)
    z -= cen - int(pvol.shape[2] / 2 * pcell[2] / vox_size)

    # Reversing axes to switch from zyx to xyz
    basis = gen_basis(pcell)[::-1, ::-1]

    # Target coordinates
    cmap = np.array([x, y, z]).reshape(3, -1)

    # Inverse mapping
    imap = np.dot(np.linalg.inv(basis), cmap).reshape((3,) + x.shape) * vox_size

    # Warped volume using inverse mapping
    wvol = transform.warp(pvol, imap)

    return wvol

def gen_basis(cell):
    ''' Generate basis matrix from cell parameters

    Parameters:
        cell - Array of 6 numbers representing unit cell
    Returns:
        3x3 transformation matrix
    '''
    c3 = np.cos(cell[3]) # pylint: disable=C0103
    c4 = np.cos(cell[4]) # pylint: disable=C0103
    c5 = np.cos(cell[5]) # pylint: disable=C0103
    s5 = np.sin(cell[5]) # pylint: disable=C0103
    basis = np.array([[1, c5, c4],
                      [0, s5, (c3 - c4 * c5) / s5],
                      [0, 0, 1]])
    basis[2, 2] = np.sqrt(1 - basis[0, 2]**2 - basis[1, 2]**2)
    basis *= cell[:3]
    basis[np.isclose(basis, 0)] = 0
    return basis
