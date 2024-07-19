import time as t
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pixell import bunch
from orphics import io
import healpy as hp
from websky_cosmo import *
import argparse
import pandas as pd

start = t.time()

paths = bunch.Bunch(io.config_from_yaml("input/sim_data.yml"))
print("Paths: ", paths)


# RUN SETTING ------------------------------------------------------------------

parser = argparse.ArgumentParser() 
parser.add_argument(
    "which_sim", type=str, help="Choose the sim e.g. websky or sehgal or agora."
)
parser.add_argument(
    "sim_version", type=str, help="Choose the nemo-sim-kit version for act tsz cat."
)
args = parser.parse_args() 


output_path = paths.cat_path
sim_version = args.sim_version
if args.which_sim == "websky": 
    print(" ::: producing catalogues for WEBSKY sim")
    sim_path = paths.websky_sim_path
elif args.which_sim == "sehgal": 
    print(" ::: producing catalogues for SEHGAL sim")
    sim_path = paths.sehgal_sim_path
elif args.which_sim == "agora": 
    print(" ::: producing catalogues for AGORA sim")
    sim_path = paths.agora_sim_path    
save_dir = f"{output_path}/{sim_version}/"
io.mkdir(f"{save_dir}")
print(" ::: saving to", save_dir)






# CATALOGUE RESAMPLE -----------------------------------------------------------

##### this needs to be sped up! 
def resample_halo(ras, decs, masses, zs, tsz_cat, snr_cut, num_bins):
    # resample websky or sehgal or agora halo catalogue 
    # to match the mass and redshift distribution of a given tsz catalogue
    # num_bins: number of mass and redshift bins for resampling 

    hdu = fits.open(tsz_cat)
    tsz_zs = hdu[1].data["redshift"]
    if args.which_sim == "websky": 
        tsz_masses = hdu[1].data["M200m"]
    elif args.which_sim == "sehgal" or args.which_sim == "agora":
        tsz_masses = hdu[1].data["M200c"]    
    tsz_snr = hdu[1].data["SNR"]

    keep = tsz_snr > snr_cut
    tsz_zs = tsz_zs[keep] 
    tsz_masses = tsz_masses[keep] 
    print(" ::: snr cut = %.1f" %snr_cut) 
    print(" ::: total number of clusters in tsz cat =", len(tsz_masses))  

    # create mass and redshift bins based on the tsz catalogue
    mbins = np.linspace(tsz_masses.min(), tsz_masses.max(), num_bins)
    zbins = np.linspace(tsz_zs.min(), tsz_zs.max(), num_bins)
    mzhist, mbin_edges, zbin_edges = np.histogram2d(tsz_masses, tsz_zs, bins=[mbins, zbins])

    resampled_ras = []
    resampled_decs = []
    resampled_masses = []
    resampled_zs = []    

    # loop through each bin in the 2D histogram and resample the input catalogue
    for i in range(len(mbin_edges)-1):
        for j in range(len(zbin_edges)-1):
            min_masses = mbin_edges[i]
            max_masses = mbin_edges[i+1]
            min_zs = zbin_edges[j]
            max_zs = zbin_edges[j+1]        

            # find indices of input cat entries within the current mass and redshift bin
            mask = (masses > min_masses) & (masses <= max_masses) & (zs > min_zs) & (zs <= max_zs)
            bin_ras = ras[mask]
            bin_decs = decs[mask]
            bin_masses = masses[mask]
            bin_zs = zs[mask]        

            # skip empty bins
            if int(mzhist[i, j]) == 0 or len(bin_masses) == 0:
                continue

            # resample the entries to match the histogram count
            np.random.seed(100)
            ind = np.random.choice(len(bin_masses), size=int(mzhist[i, j]), replace=True)
            resampled_ras.append(bin_ras[ind])
            resampled_decs.append(bin_decs[ind])
            resampled_masses.append(bin_masses[ind])
            resampled_zs.append(bin_zs[ind])

    ras = np.concatenate(resampled_ras, axis=None)
    decs = np.concatenate(resampled_decs, axis=None)
    masses = np.concatenate(resampled_masses, axis=None)
    zs = np.concatenate(resampled_zs, axis=None)

    print(" ::: min and max redshift = %.2f and %.2f" %(zs.min(), zs.max()), "(mean = %.2f)" %zs.mean()) 
    print(" ::: min and max M200 = %.2f and %.2f" %(masses.min(), masses.max()), "(mean = %.2f)" %masses.mean()) 
    print(" ::: total number of halos after resampling =", len(ras)) 

    return ras, decs, masses, zs


# PREPARING HALO CATALOGUES ----------------------------------------------------

if args.which_sim == "websky": 

    cat = f"{sim_path}halos.pksc"
    print(" ::: loading the entire websky halo catalogue:", cat) 
    f = open(cat, "r")
    N = np.fromfile(f, count=3, dtype=np.int32)[0]
    catalog = np.fromfile(f, count=N*10, dtype=np.float32)
    catalog = np.reshape(catalog, (N,10))

    x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # in Mpc (comoving)
    R  = catalog[:,6] # in Mpc

    # convert to mass, redshift, RA and DEC
    rho        = 2.775e11*omegam*h**2     # in Msun/Mpc^3
    M200m      = 4*np.pi/3.*rho*R**3      # in Msun 
    chi        = np.sqrt(x**2+y**2+z**2)  # in Mpc
    theta, phi = hp.vec2ang(np.column_stack((x,y,z))) # in radians

    ras      = np.rad2deg(phi)
    decs     = np.rad2deg(np.pi/2. - theta)  
    masses     = M200m / 1e14 
    zs       = zofchi(chi)                 

    act_cat = paths.websky_tsz_cat

elif args.which_sim == "sehgal": 

    cat = f"{sim_path}halo_nbody.ascii"
    print(" ::: loading the entire segal halo catalogue:", cat)
    f_coords = open(cat, "r")
    data = np.genfromtxt(f_coords)
    ras, decs = data[:,1], data[:,2] # degrees
    zs = data[:,0]
    M200c = data[:,12] # in Msun
    # M500c = data[:,14] # in Msun
    masses = M200c / 1e14

    act_cat = paths.sehgal_tsz_cat

elif args.which_sim == "agora":

    df = pd.read_parquet(sim_path+'../../agora_halo.parquet.gzip') 
    ras = df['totra'].to_numpy()
    decs = df['totdec'].to_numpy()
    zs = df['totz'].to_numpy()
    M200c = df['totm200'].to_numpy() # in Msun/h

    h = 0.6777
    masses = M200c / 1e14 * h  

    min_mass = 0.1 
    keep = masses > min_mass
    ras = ras[keep] 
    decs = decs[keep]
    zs = zs[keep]
    masses = masses[keep]

    print(" ::: mass cut applied: N =", len(ras))

    act_cat = paths.agora_tsz_cat

print(" ::: tsz cat used is:", act_cat)
print(" ::: min and max redshift = %.2f and %.2f" %(zs.min(), zs.max()), "(mean = %.2f)" %zs.mean()) 
print(" ::: min and max M200 = %.4f and %.2f" %(masses.min(), masses.max()), "(mean = %.2f)" %masses.mean()) 
print(" ::: total number of halos = ", len(ras))

ras, decs, masses, zs = resample_halo(ras, decs, masses, zs, act_cat, snr_cut=4, num_bins=30)
np.savetxt(f"{save_dir}{args.which_sim}_halo.txt", np.c_[ras, decs, zs, masses])

ras, decs, masses, zs = resample_halo(ras, decs, masses, zs, act_cat, snr_cut=5.5, num_bins=30)
np.savetxt(f"{save_dir}{args.which_sim}_halo_snr5p5.txt", np.c_[ras, decs, zs, masses])


elapsed = t.time() - start
print("\r ::: entire run took %.1f seconds" %elapsed)

