# Quick start with lysozyme

Here is a short tutorial on the usage of _diffuser_ with some test data. 

In this toy example, we will generate diffuse scattering from a short MD trajectory and try to fit it with a few principal-component (PC) modes of the same trajectory.

1. Install _diffuser_. Go to the root directory, setup your conda environment or virtualenv and run
```
$ pip install -e .
```

2. Unpack the contents of `lyso.tar.gz`.
```
$ tar -xzvf lyso.tar.gz
```

3. Generate the diffuse scattering from the MD trajectory along with a bit of rigid body translation.
```
$ diffuser.traj_diff traj.ini
```
Have a look at the `traj.ini` file to adjust parameters of this calculation. Like all the CLI programs in this package, more information can be obtained by running them with the `-h` flag.

4. Process the MD trajectory to prepare for optimization.
```
$ diffuser.process_md -C -D -M -n 10 vecs.ini
```
The options say to calculate the covariance matrix (`-C`), diagonalize it (`-D`) and save the average positions as a PDB file (`-M`).

5. (Optional) Generate diffuse scattering from the first 10 modes equally weighted
```
$ diffuser.pcdiff vecs.ini
```
By default, the modes in the vecs file are equally weighted, though you can set different weights by updating the `...vecs.h5` file.

6. Optimize the weights of the first 3 PC modes such that the generated data best matches the diffuse scattering from the whole trajectory.
```
$ diffuser.bgo_optimize vecs.ini 100
```
Explore `vecs.ini` in more detail to understand what is being optimized.
