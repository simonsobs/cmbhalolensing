import matplotlib.pyplot as plt
from pixell import enmap,utils as u,lensing as plensing,bench
from orphics import io, lensing as olensing, maps, cosmology,stats, mpi
import numpy as np
import symlens
import sys
import analysis as chutils

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Run flatsky sims to estimate a multiplicative correction. Alternatively, use this to explore choices and their effect on SNR.')
parser.add_argument("nsims", type=int,help='Number of sims.')
parser.add_argument("outname", type=str,help='Output name root. A directory will be made containing this root name and cannot already exist.')
parser.add_argument("--choices",     type=str,  default='choices.yaml',help="File containing analysis choices.")
args = parser.parse_args()



nsims = args.nsims
comm,rank,my_tasks = mpi.distribute(nsims)
nlen = len(my_tasks)

# Initialize flatsky simulator
# This will self-generate lensed CMB maps
# See choices.yaml
analyzer = chutils.Analysis(args.choices,'flatsky_sim', comm=comm, debug=True,outname=args.outname)

for i,task in enumerate(my_tasks):

    # Get lensed sim; (no beam, no noise)
    omap = analyzer.get_stamp(seed=task)

    # Get reconstruction; it also adds stuff to stats
    recon = analyzer.get_recon(omap, do_real=True, debug = (i==0))

    if rank==0 and (i+1)%10==0: print(f"Rank {rank} done with {i+1} / {nlen}..." )

analyzer.finish(save_transfer=True)
