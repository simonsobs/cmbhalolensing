import matplotlib.pyplot as plt
from pixell import enmap,utils as u,lensing as plensing,bench
from orphics import io, lensing as olensing, maps, cosmology,stats, mpi
import numpy as np
import symlens
import sys
import analysis as chutils


nsims = int(sys.argv[1])
comm,rank,my_tasks = mpi.distribute(nsims)
nlen = len(my_tasks)

# Initialize flatsky simulator
# This will self-generate lensed CMB maps
# See choices.yaml
analyzer = chutils.Analysis('choices.yaml','flatsky_sim', comm=comm, debug=True)

for i,task in enumerate(my_tasks):

    # Get lensed sim; (no beam, no noise)
    omap = analyzer.get_stamp(seed=task)

    # Get reconstruction; it also adds stuff to stats
    recon = analyzer.get_recon(omap, do_real=True, debug = (i==0))

    if rank==0 and (i+1)%10==0: print(f"Rank {rank} done with {i+1} / {nlen}..." )

analyzer.finish()
