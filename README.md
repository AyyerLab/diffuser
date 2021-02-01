# _Diffuser_

Optimizing internal dynamics of proteins by fitting against their diffuse scattering intensities.

## Installation

This is a pure-python package using CuPy for GPU computation. Create a conda environment or virtualenv and install using the command:
```
$ pip install -e .
```

This will add the scripts described in the next section to your path in the environment.

## Scripts

All scripts take their parameters from a config file.

 * `diffuser.rbdiff` - Rigid body diffuse scattering calculated from an electron density map and from disorder parameters
 * `diffuser.traj_diff` - Diffuse scattering from an MD trajectory
 * `diffuser.process_md` - Process MD trajectory to get pcincipal-component modes and average structure
 * `diffuser.pcdiff` - Diffuse scattering by distorting the molecule along principal-component modes
 * `diffuser.bgo_optimize` - Bayesian optimization by tuning the weights in `pcdiff` to fit against a target intensity distribution

### Tutorial
Go to the `examples/` folder for a basic tutorial.
