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

start = t.time()

paths = bunch.Bunch(io.config_from_yaml("input/sim_data.yml"))
print("Paths: ", paths)


# RUN SETTING ------------------------------------------------------------------

parser = argparse.ArgumentParser() 
parser.add_argument(
    "save_name", type=str, help="Name you want for your output."
)
parser.add_argument(
    "which_sim", type=str, help="Choose the sim e.g. websky or sehgal."
)
args = parser.parse_args() 


output_path = paths.simsuite_path
save_name = args.save_name
if args.which_sim == "websky": 
    print(" ::: producing catalogues for WEBSKY sim")
    save_name = "websky_" + save_name
    sim_path = paths.websky_sim_path
elif args.which_sim == "sehgal": 
    print(" ::: producing catalogues for SEHGAL sim")
    save_name = "sehgal_" + save_name
    sim_path = paths.sehgal_sim_path
save_dir = f"{output_path}/{save_name}/"
io.mkdir(f"{save_dir}")
print(" ::: saving to", save_dir)






# CATALOGUE RESAMPLE -----------------------------------------------------------

def resample_halo(ras, decs, mass, zs, tsz_cat, snr_cut, num_bins):
    # resample websky or sehgal halo catalogue 
    # to match the mass distribution of a given tsz catalogue
    # num_bins: number of mass bins for resampling 

    hdu = fits.open(tsz_cat)
    tsz_mass = hdu[1].data["M200m"] # in units of 1e14 Msun
    snr = hdu[1].data["SNR"]
    tsz_mass = tsz_mass[snr > snr_cut] 
    print(" ::: snr cut = %.1f" %snr_cut) 
    print(" ::: total number of clusters in tsz cat =", len(tsz_mass))  

    # create mass bins based on the tsz masses
    mbins = np.linspace(tsz_mass.min(), tsz_mass.max(), num_bins)
    mhist, mbin_edges = np.histogram(tsz_mass, bins=mbins)

    resampled_ras = []
    resampled_decs = []
    resampled_mass = []
    resampled_zs = []    

    # loop through each mass bin and resample the input catalogue
    for i in range(len(mhist)):
        min_mass = mbin_edges[i]
        max_mass = mbin_edges[i+1]

        # find indices of input cat entries within the current mass bin
        ind0 = np.where((mass > min_mass) & (mass < max_mass))[0]
        bin_ras = ras[ind0]
        bin_decs = decs[ind0]
        bin_mass = mass[ind0]
        bin_zs = zs[ind0]        

        # skip empty bins
        if mhist[i] == 0 or len(bin_mass) == 0:
            continue

        # resample the entries to match the histogran count
        np.random.seed(100)
        ind1 = np.random.choice(len(bin_mass), size=mhist[i], replace=True) 
        resampled_ras.append(bin_ras[ind1])
        resampled_decs.append(bin_decs[ind1])
        resampled_mass.append(bin_mass[ind1])
        resampled_zs.append(bin_zs[ind1])

    ras = np.concatenate(resampled_ras, axis=None)
    decs = np.concatenate(resampled_decs, axis=None)
    mass = np.concatenate(resampled_mass, axis=None)
    zs = np.concatenate(resampled_zs, axis=None)

    print(" ::: min and max redshift = %.2f and %.2f" %(zs.min(), zs.max()), "(mean = %.2f)" %zs.mean()) 
    print(" ::: min and max M200m = %.2f and %.2f" %(mass.min(), mass.max()), "(mean = %.2f)" %mass.mean()) 
    print(" ::: total number of halos after resampling =", len(ras)) 

    return ras, decs, mass, zs




# PREPARING HALO CATALOGUES ----------------------------------------------------

if args.which_sim == "websky": 

    cat = f"{sim_path}halos.pksc"
    print(" ::: loading the entire websky halo catalogue:", cat) 
    f = open(cat, "r")
    N = np.fromfile(f, count=3, dtype=np.int32)[0]
    catalog = np.fromfile(f, count=N*10, dtype=np.float32)
    catalog = np.reshape(catalog, (N,10))

    x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
    R  = catalog[:,6] # Mpc

    # convert to mass, redshift, RA and DEC
    rho        = 2.775e11*omegam*h**2     # Msun/Mpc^3
    M200m      = 4*np.pi/3.*rho*R**3      # in Msun 
    chi        = np.sqrt(x**2+y**2+z**2)  # Mpc
    theta, phi = hp.vec2ang(np.column_stack((x,y,z))) # in radians

    ras      = np.rad2deg(phi)
    decs     = np.rad2deg(np.pi/2. - theta)  
    mass     = M200m / 1e14
    zs       = zofchi(chi)                 

    act_cat = paths.websky_tsz_cat


elif args.which_sim == "sehgal": 

    cat = f"{sim_path}halo_nbody.ascii"
    print(" ::: loading the entire segal halo catalogue:", cat)
    f_coords = open(cat, "r")
    data = np.genfromtxt(f_coords)
    ras, decs = data[:,1], data[:,2] # degrees
    zs = data[:,0]
    mass = data[:,12] # this is M200c
    # mass = data[:,14] # this is M500c
    mass = mass / 1e14    

    act_cat = paths.sehgal_tsz_cat

print(" ::: tsz cat used is:", act_cat)
print(" ::: min and max redshift = %.2f and %.2f" %(zs.min(), zs.max()), "(mean = %.2f)" %zs.mean()) 
print(" ::: min and max M200m = %.2f and %.2f" %(mass.min(), mass.max()), "(mean = %.2f)" %mass.mean()) 
print(" ::: total number of halos = ", len(ras))


ras, decs, mass, zs = resample_halo(ras, decs, mass, zs, act_cat, snr_cut=4, num_bins=30)
np.savetxt(f"{save_dir}{args.which_sim}_halo.txt", np.c_[ras, decs, zs, mass])

ras, decs, mass, zs = resample_halo(ras, decs, mass, zs, act_cat, snr_cut=5.5, num_bins=30)
np.savetxt(f"{save_dir}{args.which_sim}_halo_snr5p5.txt", np.c_[ras, decs, zs, mass])


elapsed = t.time() - start
print("\r ::: entire run took %.1f seconds" %elapsed)

