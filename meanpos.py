import numpy as np
import cupy as cp
import MDAnalysis as md
import h5py

u = md.Universe("../CypA/xtc/ini.tpr", "../CypA/xtc/md295_fit.xtc")

p = u.select_atoms("protein")
f = h5py.File("../CypA/xtc/md295_fit_diffcalc_cov.h5", "r")
p_avg = f['mean_pos'][:]

p.positions = p_avg

p.write("../CypA/xtc/md295_meanpos.pdb")
