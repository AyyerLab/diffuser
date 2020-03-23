import sys
import os.path as op
import argparse
import numpy as np
import pylab as P
from matplotlib import colors
import mrcfile
import parse_dsn6
import h5py

def parse(fname, dset=None):
    ext = op.splitext(fname)[1]
    if ext == '.dsn6' or ext == '.omap':
        cvol, ccell = parse_dsn6.parse(fname)
        if cvol[0].mean() > cvol[cvol.shape[0]//2].mean():
            cvol = np.fft.fftshift(self.cvol)
    elif ext == '.ccp4' or ext == '.mrc':
        with mrcfile.open(fname, 'r') as f:
            cvol = f.data
    elif ext == '.h5':
        if dset is None:
            print('Need dset name for h5 file')
            sys.exit(1)
        with h5py.File(fname, 'r') as f:
            cvol = f[dset][:]
            if len(cvol.shape) == 4:
                cvol = self.cvol[0]
    else:
        raise IOError('Unknown file extension: %s'%ext)

    return cvol

parser = argparse.ArgumentParser(description='Simple slices viewer')
parser.add_argument('fname', help='File to view')
parser.add_argument('-d', '--dset', help='Dataset name if HDF5 file')
parser.add_argument('-g', '--gamma', help='Gamma factor to change color scale', type=float, default=0.4)
parser.add_argument('--cmap', help='Matplotlib color map', default='coolwarm')
args = parser.parse_args()

vol = parse(args.fname, dset=args.dset)
cen = vol.shape[0] // 2

P.figure(figsize=(15,6))

P.subplot(131)
P.imshow(vol[cen], norm=colors.PowerNorm(gamma=args.gamma), cmap=args.cmap)
P.title('XY plane')
P.subplot(132)
P.imshow(vol[:,cen], norm=colors.PowerNorm(gamma=args.gamma), cmap=args.cmap)
P.title('YZ plane')
P.subplot(133)
P.imshow(vol[:,:,cen], norm=colors.PowerNorm(gamma=args.gamma), cmap=args.cmap)
P.title('XZ plane')

P.suptitle(op.basename(args.fname), fontsize=14)
P.tight_layout()

P.show()
