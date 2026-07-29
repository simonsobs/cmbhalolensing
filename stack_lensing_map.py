import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils as cutils
from pixell import enmap, reproject, utils, wcsutils, curvedsky
from orphics import maps, mpi, io, stats, cosmology, lensing
from scipy.optimize import curve_fit
from numpy import save
import time
import symlens
import healpy as hp
import os, sys
from enlib import bench
import warnings

"""
Stacking on public ACT lensing maps: NO STAMP RECONSTRUCTION!!
Same procedure as using the --dr6-lensing flag in stack.py.
Uses default ell cuts from lensing map. 

!! Run
python stack_lensing_map.py -h 
!! to see options
"""

start_time,paths,defaults,args,tags,rank,data_choice = cutils.initialize_pipeline_config()
if rank==0:
    print("Paths: ",paths)
    print("Tags: ",tags)
    print("Defaults: ",defaults)
    print("Arguments: ",args)
    print("Data: ",data_choice)

# Load the catalog
ras, decs, zs, ws, cdata = cutils.catalog_interface(
    args.cat_type, 
    args.is_meanfield, 
    args.nmax, 
    args.zmin, 
    args.zmax, 
    bcg=args.bcg, 
    snmin=args.snmin, 
    snmax=args.snmax, 
    y0min=args.y0min, 
    y0max=args.y0max, 
    decmin=args.decmin
)

# Load the map
kappa_map = paths.act_data + "release/dr6_lensing_v1/maps/baseline/kappa_alm_data_act_dr6_lensing_v1_baseline.fits"
kmap = np.nan_to_num(hp.read_alm(kappa_map).astype(np.complex128))

# reading mask for geometry, mask is NOT applied to the map 
mask = enmap.read_map(paths.act_data + "DR6_lensing/masks/act_mask_20220316_GAL060_rms_70.00_d2sk.fits") 
k_map = curvedsky.alm2map(kmap, enmap.empty(mask.shape, mask.wcs, dtype=np.float64))

# stamp size and resolution
stamp_width_deg = args.swidth / 60.0        # stamp_width_arcmin: 128.0
pixel = args.pwidth                         # pix_width_arcmin: 0.5
maxr = stamp_width_deg * utils.degree / 2.0 # max radius for projection geometry 

""" 
!! CATALOG TRIMMING BASED ON ACT MASK
"""
# Remove objects that lie in unobserved regions
Norig = len(ras)
with bench.show("cull"):
    coords = np.stack([decs, ras]) * utils.degree
    # Convert catalog coords to pixel coords
    ipixs = mask.sky2pix(coords).astype(int)
    Ny, Nx = mask.shape
    pixs = []
    # Select pixels that fall within map
    sel = np.logical_and.reduce(
        (ipixs[0] > 0, ipixs[0] < Ny, ipixs[1] > 0, ipixs[1] < Nx)
    )
    ras = ras[sel]
    pixs.append(ipixs[0][sel])
    decs = decs[sel]
    pixs.append(ipixs[1][sel])
    ws = ws[sel]
    zs = zs[sel]
    for key in cdata.keys():
        cdata[key] = cdata[key][sel]
    pixs = np.stack(pixs)
    # Then select pixels within mask
    nsel = mask[pixs[0, :], pixs[1, :]] > 0.99
    ras = ras[np.argwhere(nsel)][:, 0]
    decs = decs[np.argwhere(nsel)][:, 0]
    ws = ws[np.argwhere(nsel)][:, 0]
    zs = zs[np.argwhere(nsel)][:, 0]
    for key in cdata.keys():
        cdata[key] = cdata[key][np.argwhere(nsel)][:, 0]
    nsims = len(ras)
    assert len(decs)==nsims
    assert len(ws)==nsims

    if (args.nmax is not None) and args.nmax < nsims:
        nsims = args.nmax
    del pixs, ipixs

print(f"After applying the mask, {Norig} -> {nsims}.")
try:
    print(f"zmin {min(zs)} zmax {max(zs)}")
except:
    pass

""" 
!! BINNING
"""

# for binned kappa profile
bin_edges = np.arange(0, args.arcmax, args.arcstep) # 15 arcmin, 1.5 arcmin
centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

def bin(data, modrmap, bin_edges):
    binner = stats.bin2D(modrmap, bin_edges)
    cents, ret = binner.bin(data)
    return ret

""" 
!! LOOP OVER ASSIGNED TASKS
"""

# MPI paralellization
comm, rank, my_tasks = mpi.distribute(nsims)


# An MPI statistics collector
s = stats.Stats(comm)

j = 0  # local counter for this MPI task
for task in my_tasks:
    i = task  # global counter for all objects
    coords = np.array([decs[i], ras[i]]) * utils.degree
    z = zs[i]
    cper = int((j + 1) / len(my_tasks) * 100.0)
    if rank == 0:
        print(f"Rank {rank} performing task {task} as index {j} ({cper}% complete.).")

    """ 
    !! CUT OUT STAMP
    """       
    kstamp = reproject.thumbnails(
        k_map,
        coords,
        r=maxr,
        res=pixel * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    )     
    shape, wcs = kstamp.shape, kstamp.wcs
    modrmap = enmap.modrmap(shape, wcs)
    s.add_to_stack("kstamp", kstamp)
    binned_kappa = bin(kstamp, modrmap * (180 * 60 / np.pi), bin_edges)
    s.add_to_stats("tk1d", binned_kappa)
    j+=1

# collect from all MPI cores and calculate stacks
s.get_stacks()
s.get_stats()

savedir=f"{paths.savedir}/../{args.cat_type}_dr6_lensing/"

if rank == 0:
    with bench.show("dump"):
        with open(f"{savedir}/cat_data_columns.txt",'w') as f:
                    f.write(' '.join(['z','weight',*[key for key in sorted(cdata.keys())]]))
        s.dump(savedir)
        enmap.write_map_geometry(f"{savedir}/map_geometry.fits", shape, wcs)
        # enmap.write_map(f"{savedir}/kmask.fits", kmask)
        enmap.write_map(f"{savedir}/modrmap.fits", modrmap)
        np.savetxt(f"{savedir}/bin_edges.txt", bin_edges)

elapsed = time.time() - start_time
print("\r ::: entire run took %.1f seconds" % elapsed)