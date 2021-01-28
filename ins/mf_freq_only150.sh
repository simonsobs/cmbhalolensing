#!/bin/bash
#SBATCH -N 8
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J mf_only150
#SBATCH --mail-user=eunseong.lee@manchester.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -t 4:30:00

#run the application:
srun -n 16 -c 8 --cpu_bind=cores python -W ignore /global/homes/e/eunseong/cmbhalolensing/stack.py v06only150 hilton_beta --night-only --is-meanfield --no-90 --o
