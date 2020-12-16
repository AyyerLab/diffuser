# Scripts for MD output/control

This folder contains various scripts to work with Gromacs

## Scripts

There are two kinds of scripts, to generate and work with principal components (PC) vectors from the trajectory's covariance matrix and to try to optimize restraining force constants.

### Working with PC-vectors
 * `diag.py` - Diagonalize covariance matrix and calculate principal component (PC) vectors
 * `calc_proj.py` - Calculate projected values of PC-vectors on trajectory versus time
 * `meanpos.py` - Write mean position PDB from PC-vectors file
 * `traj_pc.py` - Calculate projected trajectory on given PC vector
 * `vecs2nmd.py` - Convert PC vector to NMD file for visualization

### Optimizing force constants
 * `mditer.sh` - Run iterative MD to optimize force constants w.r.t. experimental B-factors
 * `xvg2itp.py` - Create `.itp` file with updated force constants
