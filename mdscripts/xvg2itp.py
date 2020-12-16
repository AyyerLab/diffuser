import numpy as np
import csv
import MDAnalysis as md
import os
import argparse

parser = argparse.ArgumentParser(description='Create new restraint itp file')
parser.add_argument('prefix', help='Prefix to relevant files')
parser.add_argument('-b', '--beta', help='Relaxation parameter, default: 0.02', type=float, default=0.02)
parser.add_argument('--pdb_fname', help='Path to experimental PDB file (bfac_7.pdb)', default='bfac_7.pdb')
args = parser.parse_args()

md_xvg_fname = args.prefix+'.xvg'
old_itp_fname = args.prefix.split('_')[0]+'_%.3d_fc.itp'%(int(args.prefix.split('_'_[1])-1))
new_itp_fname = args.prefix+'_fc.itp'

# Get experimental force constants (from pdb)
u = md.Universe(args.pdb_fname)
bfac_crys = u.select_atoms('protein').positions[:, 0] # bfac_expt Angstrom**2
FC_exp = (8*np.pi**2*0.008314*295/3)/(bfac_crys*0.01) #8 *np.pi**2* 0.008314*295 /bfac_crys/3/0.01 # Fc kJmol-1 nm-2

# Get current MD rmsf values (from xvg)
rmsf_data = np.loadtxt(md_xvg_fname, skiprows = 17) # rmsf in nm
rmsf = rmsf_data[:, 1]
FC_md = 0.008314 *295 /(rmsf**2) # Fc kJmol-1 nm-2

# Get old restraint force constants (from itp)
itp_data = np.loadtxt(old_itp_fname, skiprows=4)
FC_old = itp_data[:,2:]

# Calculate new restraint force constants
FC_new = FC_old + args.beta * (FC_exp - FC_md)
np.clip(FC_new, 0, None, out=FC_new)

# Write new itp file
with open(new_itp_fname, 'w') as f:
    f.write('; position restraints for Protein of GROningen MAchine for Chemical Simulation in water\n')
    f.write('\n')
    f.write('[ position_restraints ]\n')
    f.write(';  i funct       fcx        fcy        fcz\n')
    w = csv.writer(f, delimiter='\t')
    ind = np.arange(len(FC_new), dtype='i4')+1
    funct = np.ones(len(FC_new), dtype='i4')
    w.writerows(zip(ind, funct, FC_new[:,0], FC_new[:,1], FC_new[:,2]))

