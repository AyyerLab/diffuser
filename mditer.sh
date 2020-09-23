#!/bin/bash

root_dir=/home/mazumdep/Lysozyme/lyso/iter
gro_fname=${root_dir}/pr1ns.gro
top_fname=${root_dir}/lyso.top
mdp_fname=${root_dir}/lyso.mdp
prot_itp_fname=${root_dir}/lyso.itp

cd $root_dir

# For each iteration
for i in {1..10}
do
	iter=`printf "%.3d" $i`
	prefix=lyso_${iter}
	tpr_fname=${root_dir}/${prefix}.tpr

	# Make TPR file
	grompp -f $mdp_fname -c $gro_fname -r $gro_fname -p $top_fname -o $tpr_fname

	# Run GROMACS
	gmx mdrun -gputasks 0123 -nb gpu -pme gpu -npme 1 -ntmpi 4 -ntomp 4 -s $tpr_fname -deffnm $prefix

	# Calculate RMSF
	echo Protein|gmx rmsf -f ${prefix}.xtc -s $tpr_fname -o ${prefix}.xvg
	
	# Update force constant itp
	python xvg2itp.py ${prefix}

	# Update path in protein.itp
	sed -i "s/.*include.*/#include \"${prefix}_fc.itp\"/" $prot_itp_fname
done

cd -
