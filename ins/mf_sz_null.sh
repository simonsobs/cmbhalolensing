#!/bin/bash
#SBATCH -N 8
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J mf_sz_null
#SBATCH --mail-user=eunseong.lee@manchester.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -t 6:00:00

#run the application:
srun -n 16 -c 8 --cpu_bind=cores python -W ignore /global/homes/e/eunseong/cmbhalolensing/stack.py v01daynullszmask hilton_beta --day-null --debug-stack --is-meanfield --hres-lmax=5000 --o
