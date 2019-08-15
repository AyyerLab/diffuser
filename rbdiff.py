'''Calculate diffuse scattering from rigid body motions of electron density'''

import sys
import numpy
try:
    import cupy as np
    from cupyx.scipy import ndimage
    print('Using CuPy')
except ImportError:
    import numpy as np
    from scipy import ndimage
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
    fname = '/home/ayyerkar/acads/ATP/2w5j_cutout_rot_remap.ccp4'
    out_fname = '/home/ayyerkar/acads/ATP/2w5j_diffcalc.ccp4'
    sigma_deg = 10
    rot_plane = (1, 2) # Rotate about x-axis
    num_steps = 21

    # Read density map
    with mrcfile.open(fname, 'r') as f:
        dens = np.array(f.data)
        cella = f.header.cella['x']

    # Generate molecular transform
    fdens = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(dens)))
    x, y, z = np.indices(fdens.shape, dtype='f4')
    cen = fdens.shape[0] // 2
    x  = (x-cen) / fdens.shape[0] * 2. * np.pi
    y  = (y-cen) / fdens.shape[0] * 2. * np.pi
    z  = (z-cen) / fdens.shape[0] * 2. * np.pi

    mean_fdens = np.zeros_like(fdens)
    mean_intens = np.zeros(fdens.shape, dtype='f4')

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
    # Translate and average
    shifts = np.linspace(-3*sigma_deg, 3*sigma_deg, num_steps)
    weights = np.exp(-shifts**2/2./sigma_deg**2)
    for i in range(num_steps):
        rotfdens = trans_fdens(fdens, (shifts[i], 0, 0), x, y, z)
        mean_fdens += rotfdens * weights[i]
        mean_intens += np.abs(rotfdens)**2 * weights[i]
        sys.stderr.write('\r%d/%d: %+.3f deg (%.3e)   ' % (i+1, num_steps, shifts[i], weights[i]))
    sys.stderr.write('\n')
    '''

    mean_fdens /= weights.sum()
    mean_intens /= weights.sum()
    diff_intens = mean_intens - np.abs(mean_fdens)**2

    # Save to file
    with mrcfile.new(out_fname, overwrite=True) as f:
        f.set_data(numpy.array(diff_intens.astype('f4')))
        for key in f.header.cella.dtype.names:
            f.header.cella[key] = cella
