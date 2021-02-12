# _Diffuser_

Discovering internal dynamics of proteins by fitting against their diffuse scattering intensities.

## Installation

This is a pure-python package using CuPy for GPU computation. Create a conda environment or virtualenv and install Cupy using the documentation given [here](https://docs.cupy.dev/en/stable/install.html) (you shouldn't need to compile it by installing using pip). _Diffuser_ itself is installed using the command in the root directory of the repository:
```
$ pip install -e .
```

This will add the scripts described in the next section to your path in the environment.

## Tutorial
Go to the `examples/` folder for a basic tutorial.

## Scripts

All scripts take their parameters from a config file.

 * `diffuser.rbdiff` - Rigid body diffuse scattering calculated from an electron density map and from disorder parameters
 * `diffuser.traj_diff` - Diffuse scattering from an MD trajectory
 * `diffuser.process_md` - Process MD trajectory to get pcincipal-component modes and average structure
 * `diffuser.pcdiff` - Diffuse scattering by distorting the molecule along principal-component modes
 * `diffuser.bgo_optimize` - Bayesian optimization by tuning the weights in `pcdiff` to fit against a target intensity distribution

