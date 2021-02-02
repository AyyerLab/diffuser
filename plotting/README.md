# Scripts for analysis and plotting
  
This folder contains various sctipts to analyse and plot data.

## Scripts for plotting

 * `calc_cc.py` -  Calculate cros-correlation between two different intensity distribution. CC vs radius/q calculator.
   ```
   $ python calc_cc.py diff_intens1(target).h5 diff_intens2(calculated).h5 -r res_edge -b bin_size -o out_fname.dat
   ``` 
   
 * `plot_CC_q.py`- Generate the diffuse scattering from the optimized PC modes and calculate the cross-corelation (CC) with target diffuse scattering. Plot CC vs q.
   ```
   $ python plot_CC_q.py ../examples/vecs.ini -mc -i diff_intens.h5 -o out_fname.dat -c CC.dat 
   ``` 
   Have a look at the `plot_CC_q.py` file to select options. More information can be obtained by running them with the `-h` flag.
   with option (`-mc`) - calculate diffuse map (in .h5 format), then calculate CC with target map (in .dat format) and plot CC vs q.
   with option (`-i`) path to the calculated diffuse map - calculate CC with target map and plot CC vs q.
   With option (`-c`) path to the CC.dat - plot CC vs q.
   

 *  `plot_proj.py`- Genarate projection of PCs on trajectory plot.

 
### Scripts for visualization of intensity distribution
   
* `view.py` - Simple slice viewer. 
* `view3d.py`- 3D isosurface viewer.

#### Script for some plotting funtions
  `plot_utils.py` - Plotting functions for BGO results and atom-pair displacement correlation
