import matplotlib.colors as mcolors
import numpy as np
from orphics.io import plot_img, Plotter
from pixell import utils as u, enmap
from sklearn.covariance import LedoitWolf
import sys
import os
sys.path.append(os.path.abspath("/home3/nehajo/scripts/"))
from profiles import errors

names = sys.argv[1:]

plot_mf=True

root = "/home3/nehajo/projects/CMASS-cmbhalolensing/post"

if len(names) == 1:
    name = names[0]
    thetas, profile = np.loadtxt(f"{root}/{name}/{name}_opt_profile.txt", unpack=True)
    _, errs = np.loadtxt(f"{root}/{name}/{name}_opt_profile_errs.txt", unpack=True)
    _, mf = np.loadtxt(f"{root}/{name}/{name}_mf.txt", unpack=True)
    _, mf_errs = np.loadtxt(f"{root}/{name}/{name}_mf_errs.txt", unpack=True)
    
    #write shrunk covmat
    data0 = np.genfromtxt(f"{root}/{name}/{name}_all_profiles.txt") #using shrunk covmat
    lw = LedoitWolf()
    lw.fit(data0)
    scov = lw.covariance_ / data0.shape[0]
    np.savetxt(f"{root}/{name}/{name}_shrunk_cov.txt", scov)

    # write map geometry
    shape, wcs = enmap.read_map_geometry(f"{root}/{name}/{name}_kmask.fits")
    enmap.write_map_geometry(f"{root}/{name}/{name}_map_geometry.fits", shape, wcs)

    pl = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
    pl.add_err(thetas, profile, errs, ls="-", label = "Filtered kappa, meanfield subtracted (optimal)")
    pl.hline(y=0)
    pl.done(f"{root}/{name}/{name}_opt_profile_clean_nomf.png")

    if plot_mf: 
        plmf = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
        plmf = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
        plmf.add_err(thetas, profile, errs, ls="-", label = "Filtered kappa, meanfield subtracted (optimal)")
        plmf.add_err(thetas, mf, mf_errs, ls="-", label = "Mean-field (optimal)")
        plmf.hline(y=0)
        plmf.done(f"{root}/{name}/{name}_opt_profile_clean.png")

else:
    pl = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
    for name in names:
        thetas, profile = np.loadtxt(f"{root}/{name}/{name}_opt_profile.txt", unpack=True)
        _, errs = np.loadtxt(f"{root}/{name}/{name}_opt_profile_errs.txt", unpack=True)
        label = name.replace("_", " ")
        pl.add_err(thetas, profile, errs, ls="-", label = label)
    pl.hline(y=0)
    pl.done(f"{root}/{names[-1]}/{names[-1]}_profile_comp.png")


