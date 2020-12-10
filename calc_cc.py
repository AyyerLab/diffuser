import csv
import mrcfile
import numpy
import h5py
try:
    import cupy as np
    from cupyx.scipy import ndimage
    import cupyx
    CUPY = True
    np.disable_experimental_feature_warning = True
    scatter_add = cupyx.scatter_add
except ImportError:
    import numpy as np
    from scipy import ndimage
    CUPY = False
    scatter_add = np.add.at

def calc_rad(size, binning):
    cen = size // 2
    ind = np.arange(size, dtype='f4') - cen
    x, y, z = np.meshgrid(ind, ind, ind, indexing='ij')
    rad = np.sqrt(x*x + y*y + z*z)
    return rad, np.rint(rad / binning).astype('i4')

def subtract_radavg(vol, intrad, num_bins, mask=None):
    radavg = np.zeros(num_bins)
    radcount = np.zeros(num_bins)
    if mask is None:
        mask = ~np.isnan(vol)
    
    scatter_add(radcount, intrad[mask], 1)
    scatter_add(radavg, intrad[mask], vol[mask])
    sel = (radcount > 0)
    radavg[sel] /= radcount[sel]

    vol -= radavg[intrad]
    return radavg

def calc_cc(vol1, vol2, intrad, num_bins, mask=None):
    v1v2 = np.zeros(num_bins)
    v1v1 = np.zeros(num_bins)
    v2v2 = np.zeros(num_bins)
    if mask is None:
        mask = ~(np.isnan(vol1) | np.isnan(vol2))

    scatter_add(v1v2, intrad[mask], vol1[mask]*vol2[mask])
    scatter_add(v1v1, intrad[mask], vol1[mask]**2)
    scatter_add(v2v2, intrad[mask], vol2[mask]**2)

    denr = v1v1 * v2v2
    sel = (denr > 0)
    v1v2[sel] /= np.sqrt(denr[sel])
    return v1v2

def save_to_file(fname, cc, q):
    print('Writing output to', fname)
    with open(fname, 'w') as f:
        w = csv.writer(f, delimiter='\t')
        if q is not None:
            w.writerow(['Radius', 'q (1/A)', 'CC'])
            w.writerows(zip(np.arange(len(cc)), q, cc))
        else:
            print('Resolution at edge not provided so CC only given vs radius')
            w.writerow(['Radius', 'CC'])
            w.writerows(zip(np.arange(len(cc)), cc))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='CC vs radius/q calculator')
    parser.add_argument('volume1', help='Path to first volume ccp4 map')
    parser.add_argument('volume2', help='Path to second volume ccp4 map')
    parser.add_argument('-r', '--res_edge', help='Resolution at center-edge in A', type=float, default=-1.)
    parser.add_argument('-b', '--bin_size', help='Radial bin size in voxels. Default: 1', type=int, default=1)
    parser.add_argument('-o', '--out_fname', help='Path to output file. Default: cc.dat', default='cc.dat')
    args = parser.parse_args()
    
    #with mrcfile.open(args.volume1, 'r') as f:
    #    vol1 = np.array(np.copy(f.data))
    with h5py.File(args.volume1, 'r') as f:
        vol1 = np.array(f['diff_intens'][:])
    #with mrcfile.open(args.volume2, 'r') as f:
    #    vol2 = np.array(np.copy(f.data))
    with h5py.File(args.volume2, 'r') as f:
        vol2 = np.array(f['diff_intens'][:])
    mask = ~(np.isnan(vol1) | np.isnan(vol2))
    
    assert vol1.shape == vol2.shape
    size = vol1.shape[-1]

    rad, rbin = calc_rad(size, args.bin_size)
    num_bins = int(rbin.max() + 1)

    subtract_radavg(vol1, rbin, num_bins, mask)
    subtract_radavg(vol2, rbin, num_bins, mask)

    cc = calc_cc(vol1, vol2, rbin, num_bins, mask)

    if args.res_edge > 0.:
        cen = size//2
        q = np.arange(num_bins) * args.bin_size / cen / args.res_edge
    else:
        q = None
    save_to_file(args.out_fname, cc, q)

if __name__ == '__main__':
    main()
