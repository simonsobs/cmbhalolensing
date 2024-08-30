import matplotlib.colors as mcolors
import numpy as np
from orphics.io import plot_img, Plotter
import pixell.utils as u
import sys
import os
sys.path.append(os.path.abspath("/home3/nehajo/scripts/"))
from profiles import errors

savename = sys.argv[1]
path = sys.argv[2]
prefixes = sys.argv[3:]


ymin=-0.02
ymax=0.04

tk1d = {}   # true kappa stack
stack = {}  # recon stack
mf = {}     # meanfield
corr = {}
stack1d = {}
mf1d = {}
tk1d_err = {}
err = {}
mf_err = {}
r = {}
k1d = {}    # recon stack - mf
all_k1d = {}


for prefix in prefixes: #TODO: load true kappa (mf sub?)

    pathname = f"{path}/{prefix}/{prefix}"

    tk1d[prefix] = np.load(f"{pathname}_1tkappa.npy")
    stack[prefix] = np.load(f"{pathname}_1rkappa.npy")
    mf[prefix] = np.load(f"{pathname}_mf_1rkappa.npy")
    corr[prefix] = np.load(f"{pathname}_1rkappa_corr.npy")
    all_k1d[prefix] = np.load(f"{pathname}_mstats/mstats_dump_vectors_k1d.npy")

    r[prefix], stack1d[prefix] = np.loadtxt(f"{pathname}_1binned_rkappa.txt", unpack=True)
    r[prefix], mf1d[prefix] = np.loadtxt(f"{pathname}_mf_1binned_rkappa.txt", unpack=True)
    
    tk1d_err[prefix] = np.loadtxt(f"{pathname}_1tkappa_errs.txt")
    err[prefix] = np.loadtxt(f"{pathname}_1rkappa_errs.txt")
    mf_err[prefix] = np.loadtxt(f"{pathname}_mf_1rkappa_errs.txt")

    kmap = stack[prefix] - mf[prefix]
    k1d[prefix] = stack1d[prefix] - mf1d[prefix]
    kmap_zoom = kmap[100:140,100:140]

    np.save(f"{pathname}_post_1rkappa.npy", kmap)
    plot_img(kmap,f"{pathname}_post_1rkappa.png", flip=False, ftsize=12, ticksize=10,cmap='coolwarm',
            label=r'$\kappa$',arc_width=120,xlabel=r"$\theta_x$ (arcmin)",ylabel=r"$\theta_y$ (arcmin)")
    plot_img(kmap_zoom,f"{pathname}_post_1rkappa_zoom.png", flip=False, ftsize=12, ticksize=10,
            cmap='coolwarm',label=r'$\kappa$',arc_width=20,xlabel=r"$\theta_x$ (arcmin)",ylabel=r"$\theta_y$ (arcmin)")

    plot_img(corr[prefix],f'{pathname}_corr.png')

    pl = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
    pl.add_err(r[prefix], k1d[prefix], yerr=err[prefix], ls="-",label="Filtered kappa, mean-field subtracted")
    pl.add_err(r[prefix], mf1d[prefix], yerr=mf_err[prefix],label="Mean-field",ls="-",alpha=0.5)
    pl.hline(y=0)
    pl._ax.set_ylim(ymin,ymax)
    pl.done(f"{pathname}_post_1rkappa_profile.png")


# overplotting
colors = list(mcolors.TABLEAU_COLORS.values())

pl = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
a=1.    #line opacity
o=0     #offset


for i,prefix in enumerate(prefixes):
    print(prefixes)
    if 'cib' in prefix:
        label = "CIB"
    if 'ksz' in prefix:
        label+= " + kSZ"
    else: 
        label = "no foreground"

    if 'hres' not in prefix:
        label += " (+ tSZ hres)"

    color=colors[i]
    pl.add_err(r[prefix]+o, k1d[prefix], yerr=err[prefix], ls="-",label=f"{label} kappa, mf sub", alpha=a, color=color)
    pl.add_err(r[prefix]+o, mf1d[prefix], yerr=mf_err[prefix],ls="--",alpha=a,color=color)
    a*=0.8
    o+=0.2
pl.add([],[],ls="--", label="mf",color = "black")
pl.hline(y=0)
pl._ax.set_ylim(ymin,ymax)
pl.done(f"{path}/{savename}_1rkappa_profile.png")


# relative difference
colors = list(mcolors.TABLEAU_COLORS.values())

pl_data = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
pl_mf = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')

baseline = None
foregrounds = []
for prefix in prefixes:
    if ('cib' not in prefix) and ('ksz' not in prefix) and ('hres' not in prefix):
        baseline = prefix
    else:
        foregrounds.append(prefix)

a=1.    #line opacity
o=0     #offset
for i,fg in enumerate(foregrounds):
    if 'cib' in fg:
        label = "CIB"
    if 'ksz' in fg:
        label+= " + kSZ"
    else: label = "no foreground"
    if 'hres' not in fg:
        label += " (+ tSZ hres)"

    color = colors[i+1]
    r_diff = r[fg]-r[baseline]
    assert np.sum(r_diff) == 0.
    k1d_diff = k1d[fg]-k1d[baseline]
    mf1d_diff = mf1d[fg]-mf1d[baseline]
    all_k1d_diff = all_k1d[fg] - all_k1d[baseline]
    err_diff = errors(all_k1d_diff)
    mf_err_diff = np.sqrt(mf_err[fg]**2 + mf_err[baseline]**2)
    pl_data.add_err(r[baseline]+o, k1d_diff, yerr=err_diff, ls="-", label=f"{label} relative kappa, mf sub", alpha=a, color=color)
    pl_mf.add_err(r[baseline]+o, mf1d_diff, yerr=err_diff, ls="--", label=f"{label} relative mf", alpha=a, color=color)

    a*=0.8
    o+=0.2
pl_data.hline(y=0)
pl_data.done(f"{path}/{savename}_rel_1rkappa_profile.png")
pl_mf.hline(y=0)
pl_mf.done(f"{path}/{savename}_rel_mf_1rkappa_profile.png")

# true kappa comparison

pl = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')

pl.add_err(r[baseline], k1d[baseline], yerr=err[baseline], ls="-",label=f"{label} kappa recon, mf sub")
pl.add_err(r[baseline], tk1d[baseline], yerr=err[baseline], ls="-", label = "true kappa", color = "black")

pl_data.hline(y=0)
pl_data.done(f"{path}/{savename}_1tkappa_1rkappa_profile.png")