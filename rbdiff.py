'''Calculate diffuse scattering from rigid body motions of electron density'''

import sys
import numpy
try:
    import cupy as np
    from cupyx.scipy import ndimage
    CUPY = True
    print('Using CuPy')
    np.disable_experimental_feature_warning = True
except ImportError:
    import numpy as np
    from scipy import ndimage
    CUPY = False
    print('Using NumPy/SciPy')
import mrcfile

def rot_fdens(fdens, ang, plane):
    out = np.empty_like(fdens)
    ndimage.rotate(fdens.real, ang, axes=plane, reshape=False, order=0, output=out.real)
    ndimage.rotate(fdens.imag, ang, axes=plane, reshape=False, order=0, output=out.imag)
    return np.array(out)

def trans_fdens(fdens, vec, qx, qy, qz):
    return fdens * np.exp(-1j * (qx*vec[0] + qy*vec[1] + qz*vec[2]))

if __name__ == '__main__':
    fname = 'data/2w5j_cutout_rot_remap.ccp4'
    out_fname = 'data/2w5j_diffcalc.ccp4'
    sigma_deg = 7
    sigma_vox = 0.8
    rot_plane = (1, 2) # Rotate about x-axis
    num_steps = 201

    # Read density map
    with mrcfile.open(fname, 'r') as f:
        dens = np.array(f.data)
        cella = f.header.cella['x']

    # Generate molecular transform
    fdens = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(dens)))
    mean_fdens = np.zeros_like(fdens)
    mean_intens = np.zeros(fdens.shape, dtype='f4')

    '''
    # Rotate and average
    angles = np.linspace(-3*sigma_deg, 3*sigma_deg, num_steps)
    weights = np.exp(-angles**2/2./sigma_deg**2)
    for i in range(num_steps):
        rotfdens = rot_fdens(fdens, angles[i], rot_plane) 
        mean_fdens += rotfdens * weights[i]
        mean_intens += np.abs(rotfdens)**2 * weights[i]
        sys.stderr.write('\r%d/%d: %+.3f deg (%.3e)   ' % (i+1, num_steps, angles[i], weights[i]))
    sys.stderr.write('\n')

    '''
    # Monte Carlo sampling of rotations and translations
    # -- Generate phase ramp basis
    x, y, z = np.indices(fdens.shape, dtype='f4')
    cen = fdens.shape[0] // 2
    x  = (x-cen) / fdens.shape[0] * 2. * np.pi
    y  = (y-cen) / fdens.shape[0] * 2. * np.pi
    z  = (z-cen) / fdens.shape[0] * 2. * np.pi

    # -- Generate 3d shifts
    shifts = np.random.multivariate_normal(np.zeros(3), np.diag(np.ones(3)*sigma_vox**2), size=num_steps)
    angles = np.random.randn(num_steps)*sigma_deg
    weights = np.ones(shifts.shape[0])
    for i in range(num_steps):
        if shifts[i].max() != 0.:
            modfdens = trans_fdens(fdens, tuple(shifts[i]), x, y, z)
        else:
            modfdens = fdens

        if angles[i] != 0.:
            modfdens = rot_fdens(modfdens, angles[i], rot_plane)

        mean_fdens += modfdens
        mean_intens += np.abs(modfdens)**2
        sys.stderr.write('\r%d/%d' % (i+1, num_steps))
    sys.stderr.write('\n')

    mean_fdens /= weights.sum()
    mean_intens /= weights.sum()
    diff_intens = mean_intens - np.abs(mean_fdens)**2
    # If using GPU, move output back to host
    if CUPY:
        diff_intens = diff_intens.get()

    # Save to file
    with mrcfile.new(out_fname, overwrite=True) as f:
        f.set_data(diff_intens.astype('f4'))
        for key in f.header.cella.dtype.names:
            f.header.cella[key] = cella
