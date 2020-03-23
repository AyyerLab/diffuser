import mrcfile
import numpy as np
with mrcfile.open('data/top/100ns_protein-H_diffcalc.ccp4', 'r') as f:
    intens = np.copy(f.data)
intens = 0.5 * (intens + intens[:,:,::-1]) # point group symmetrization
intens = np.transpose(intens, (2,1,0))
with mrcfile.new('data/transformed_100ns_Protein-H_diffcalc_rot_8_trans0_8.ccp4', overwrite=True) as f:
    f.set_data(intens)
