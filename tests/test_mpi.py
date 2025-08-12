from orphics import maps, mpi, io, stats, cosmology,catalogs
import numpy as np
from pixell import enmap, reproject, enplot, utils, wcsutils


Nobj = 10
dec_min = -70.*utils.degree
dec_max = 20.*utils.degree

shape,wcs = enmap.band_geometry((dec_min,dec_max),res=0.5*utils.arcmin)
omap = maps.rand_map(shape,wcs,pol=False)
poss = catalogs.get_random_catalog(Nobj,dec_min=dec_min,dec_max=dec_max)
print(poss.shape)

# MPI paralellization
comm, rank, my_tasks = mpi.distribute(Nobj)

# An MPI statistics collector
s = stats.Stats(comm)


for task in my_tasks:
    print("Rank ", rank, " doing task ",task)
    thumb = reproject.thumbnails(omap,poss[:,0],res=0.5*utils.arcmin,r=60*utils.arcmin)
    if thumb is None: continue
    s.add_to_stack("stack",thumb)


s.get_stacks()
if rank==0:
    print(s.stacks['stack'].shape)

    

