# Scripts for analysis and plotting
This folder contains various helper scripts to plot output.

## Scripts for plotting

 * `plot_cc_q.py`- Generate the diffuse scattering from the optimized PC modes and calculate the cross-corelation (CC) with target diffuse scattering. Plot CC vs q. Example:
   ```
   $ python plot_cc_q.py ../examples/vecs.ini -m -o ../examples/lyso_vecs
   ``` 
   Have a look at the `plot_cc_q.py` file to select options. More information can be obtained by running them with the `-h` flag.
    * `-m` - calculate diffuse map (in .h5 format), then calculate CC with target map and plot CC vs q.
    * `-i` - path to the calculated diffuse map. Calculate CC with target map and plot CC vs q.
    * `-c` - path to the CC.dat file. Parse file and plot CC vs q.

 *  `plot_proj.py`- Plot projection of PCs on trajectory.

 
### Scripts for visualization of intensity distribution
* `view.py` - Simple slice viewer. 
* `view3d.py`- 3D isosurface viewer.

#### Script for some plotting funtions
  `plot_utils.py` - Plotting functions for BGO results and atom-pair displacement correlation
