import numpy as np
import matplotlib.pyplot as plt
import utils as cutils
from pixell import enmap, reproject, enplot, utils, wcsutils,lensing as plensing
from orphics import maps, mpi, io, stats,cosmology,lensing # github.com/msyriac/orphics ; pip install -e . --user
from numpy import save
import os, sys
import time as t
import warnings

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Stacked CMB lensing.')
parser.add_argument("version", type=str,help='Version label.')
parser.add_argument("-N", "--nsims",     type=int,  default=None,help="Number of sims.")
parser.add_argument("--stamp-width-arcmin",     type=float,  default=128.0,help="Stamp width arcmin.")
parser.add_argument("--pix-width-arcmin",     type=float,  default=0.5,help="Stamp width arcmin.")
parser.add_argument("--pix-scale",     type=int,  default=5,help="Scale by this much.")
parser.add_argument("--buffer-fact",     type=int,  default=2,help="Buffer by this much.")
parser.add_argument("--lens-order",     type=int,  default=5,help="Lensing order.")
parser.add_argument("-o","--overwrite", action='store_true',help='Overwrite existing version.')
args = parser.parse_args()

## SOME IO/PREP
nsims = args.nsims
p = cutils.p # directory paths dictionary
comm, rank, my_tasks = mpi.distribute(nsims)

vstr = f"lensed_maps_{args.version}_swidth_{args.stamp_width_arcmin:.2f}_pwidth_{args.pix_width_arcmin:.2f}_bfact_{args.buffer_fact}_pfact_{args.pix_scale}_lorder_{args.lens_order}"

savedir = p['scratch'] + f"/{vstr}/"
overwrite = args.overwrite
if not(overwrite):
    assert not(os.path.exists(savedir)), \
   "This version already exists on disk. Please use a different version identifier or use the overwrite argument."
if rank==0:
    try: 
        os.makedirs(savedir)
    except:
        if overwrite: pass
        else: raise
comm.Barrier()


## Set up the lensing simulation geometry
Npix = int(args.stamp_width_arcmin * args.buffer_fact / (args.pix_width_arcmin/args.pix_scale))
dNpix = int(args.stamp_width_arcmin * args.buffer_fact / (args.pix_width_arcmin))
ddNpix = int(args.stamp_width_arcmin / (args.pix_width_arcmin))
shape,wcs = enmap.geometry(pos=(0,0),shape=(Npix,Npix),res=args.pix_width_arcmin*utils.arcmin/args.pix_scale,proj='plain')

# Select the fiducial cluster profile from Hu, DeDeo, Vale 2007
massOverh = 2e14
z = 0.7
c = 3.2
cc = cutils.get_hdv_cc()

# Set up lensing
def lens_map(imap):
    return plensing.displace_map(imap, alpha, order=args.lens_order)

# Map of distances from center
modrmap = enmap.modrmap(shape,wcs)
# The convergence ~ projected mass density
kappa = lensing.nfw_kappa(massOverh,modrmap,cc,zL=z,concentration=c,overdensity=180.,critical=False,atClusterZ=False)
if rank==0: enmap.write_map(f'{savedir}kappa.fits',kappa)

# Deflection field
alpha = lensing.alpha_from_kappa(kappa)
# Fourier magnitude map
modlmap = enmap.modlmap(shape,wcs)

# Unlensed CMB theory
theory = cosmology.default_theory()
cltt2d = theory.uCl('TT',modlmap)
clte2d = theory.uCl('TE',modlmap)
clee2d = theory.uCl('EE',modlmap)
power = np.zeros((3,3,shape[0],shape[1]))
power[0,0] = cltt2d
power[1,1] = clee2d
power[1,2] = clte2d
power[2,1] = clte2d

# Unlensed CMB generator
mgen = maps.MapGen((3,)+shape,wcs,power)


for j,task in enumerate(my_tasks):
    print(f'Rank {rank} performing task {task} as index {j}')
    cmb = mgen.get_map(seed=cutils.get_seed('lensed',task,False)) # unlensed map ; you can save it if you want, before lensing it
    cmb = lens_map(cmb) # do the lensing
    dcmb = cmb.resample((3,dNpix,dNpix)) # downsample the lensed CMB map (if saving unlensed, do the same to it)
    # For some reason, I was saving the Fourier transforms... probably since it is easier to apply a beam later
    kmap = enmap.map2harm(dcmb,iau=False)
    enmap.write_map(f'{savedir}lensed_kmap_real_{task:06d}.fits',kmap.real)
    enmap.write_map(f'{savedir}lensed_kmap_imag_{task:06d}.fits',kmap.imag)
    


