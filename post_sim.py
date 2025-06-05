import matplotlib.colors as mcolors
import numpy as np
from orphics.io import plot_img, Plotter
import pixell.utils as u
import sys
from sklearn.covariance import LedoitWolf
import os
sys.path.append(os.path.abspath("/home3/nehajo/scripts/"))
from profiles import errors, chi_square_pte

savename = sys.argv[1] # prefix for files (e.g. agora_90)
stack_path = sys.argv[2] # where profiles and stacks are saved
sim = sys.argv[3] # agora, websky
freq = sys.argv[4] # 90, 150
title_text = sys.argv[5]
prefixes = sys.argv[6:] # cmb, cmb_cib, cmb_ksz_cib, cmb_tsz, cmb_tsz_cib, cmb_tsz_ksz_cib

plot_mf = True
plot_tk1d = True


ymin=None
ymax=None

stack = {}  # recon stack
mf = {}     # meanfield
corr = {}
stack1d = {}
all_stack1d = {}
all_mf1d = {}
mf1d = {}
tk1d = {}
all_tk1d = {}
tk1d_err = {}
err = {}
mf_err = {}
r = {}
k1d = {}    # recon stack - mf
all_k1d = {}
shrunk_cov = {}


for prefix in prefixes:
    print(prefix)
    pathname = f"{stack_path}/{sim}_{prefix}_{freq}/{sim}_{prefix}_{freq}"

    stack[prefix] = np.load(f"{pathname}_1rkappa.npy")
    mf[prefix] = np.load(f"{pathname}_mf_1rkappa.npy")
    corr[prefix] = np.load(f"{pathname}_1rkappa_corr.npy")

    r[prefix], stack1d[prefix] = np.loadtxt(f"{pathname}_1binned_rkappa.txt", unpack=True)
    all_stack1d[prefix] = np.loadtxt(f"{pathname}_1binned_rkappa_vectors.txt")
    _, mf1d[prefix] = np.loadtxt(f"{pathname}_mf_1binned_rkappa.txt", unpack=True)
    all_mf1d[prefix] = np.loadtxt(f"{pathname}_mf_1binned_rkappa_vectors.txt")
    
    err[prefix] = np.loadtxt(f"{pathname}_1rkappa_errs.txt")
    mf_err[prefix] = np.loadtxt(f"{pathname}_mf_1rkappa_errs.txt")

    if plot_tk1d:
        _, tk1d[prefix] = np.loadtxt(f"{pathname}_1binned_tkappa.txt", unpack=True)
        all_tk1d[prefix] = np.loadtxt(f"{pathname}_1binned_tkappa_vectors.txt")
        tk1d_err[prefix] = np.loadtxt(f"{pathname}_1tkappa_errs.txt")

    k1d[prefix] = stack1d[prefix] - mf1d[prefix]
    np.savetxt(f"{pathname}_post_1binned_rkappa.txt", np.asarray([r[prefix], k1d[prefix]]).T)

    kmap = stack[prefix] - mf[prefix]
    kmap_zoom = kmap[100:140,100:140]
    np.save(f"{pathname}_post_1rkappa.npy", kmap)
    
    plot_img(kmap,f"{pathname}_post_1rkappa.png", flip=False, ftsize=12, ticksize=10,cmap='coolwarm',
            label=r'$\kappa$',arc_width=120,xlabel=r"$\theta_x$ (arcmin)",ylabel=r"$\theta_y$ (arcmin)")
    plot_img(kmap_zoom,f"{pathname}_post_1rkappa_zoom.png", flip=False, ftsize=12, ticksize=10,
            cmap='coolwarm',label=r'$\kappa$',arc_width=20,xlabel=r"$\theta_x$ (arcmin)",ylabel=r"$\theta_y$ (arcmin)")

    plot_img(corr[prefix],f'{pathname}_corr.png')

    pl = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
    pl.add_err(r[prefix], k1d[prefix], yerr=err[prefix], ls="-",label="Filtered kappa, mean-field subtracted")
    if plot_mf:
        pl.add_err(r[prefix], mf1d[prefix], yerr=mf_err[prefix],label="Mean-field",ls="-",alpha=0.5)
    pl.hline(y=0)
    pl._ax.set_ylim(ymin,ymax)
    pl.done(f"{pathname}_post_1rkappa_profile.png")

print(list(tk1d.keys()))

def labeling(prefix):
    label = "CMB"
    if 'tsz' in prefix:
        label += " + tSZ"
    if 'cib' in prefix:
        label += " + CIB"
    if 'ksz' in prefix:
        label += " + kSZ"
    if 'inpaint' in prefix:
        label += " inpainted"
    return label

# overplotting
colors = list(mcolors.TABLEAU_COLORS.values())

pl = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
if plot_mf: plmf = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')

a=1.    #line opacity
o=0     #offset
da = 0.8
do = 0

for i,prefix in enumerate(prefixes):
    # print(prefixes)
    label = labeling(prefix)

    color=colors[i]
    pl.add_err(r[prefix]+o, k1d[prefix], yerr=err[prefix], ls="-",label=label, alpha=a, color=color)
    
    if plot_mf:
        plmf.add_err(r[prefix]+o, k1d[prefix], yerr=err[prefix], ls="-",label=label, alpha=a, color=color)
        plmf.add_err(r[prefix]+o, mf1d[prefix], yerr=mf_err[prefix],ls="--",alpha=a,color=color)

    a*=da
    o+=do
print(prefixes[0])
if plot_tk1d:
    pl.add_err(r[prefixes[0]], tk1d[prefixes[0]], yerr=tk1d_err[prefixes[0]], ls="-", label="true kappa", color="black")
    if plot_mf:
        plmf.add_err(r[prefixes[0]], tk1d[prefixes[0]], yerr=tk1d_err[prefixes[0]], ls="-", label="true kappa", color="black")
pl.hline(y=0)
pl._ax.set_ylim(ymin,ymax)
pl.legend(loc='outside')
pl._ax.set_title(title_text)
pl.done(f"{stack_path}/{savename}_1rkappa_profile.png")
if plot_mf:
    plmf.add([],[],ls="--", label="mf",color = "black")
    plmf.hline(y=0)
    plmf._ax.set_ylim(ymin,ymax)
    plmf.legend(loc='outside')
    plmf._ax.set_title(title_text)
    plmf.done(f"{stack_path}/{savename}_1rkappa_profile_mf.png")

def difference(mean1, mean2, all1, all2):
    diff = (mean1 - mean2)
    all_diff = (all1 - all2)
    # err, covmat = errors(all_diff, Nobj=len(all1))
    ### getting shrunk covmat
    lw = LedoitWolf()
    lw.fit(all_diff)
    covmat = lw.covariance_ / all_diff.shape[0]
    err = np.sqrt(np.diag(covmat))
    return diff, err, covmat

# relative difference
colors = list(mcolors.TABLEAU_COLORS.values())

pl_data = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', 
                  ylabel='$\\kappa_{CMB+} - \kappa_{CMB}$')

# if plot_mf:
#     pl_mf = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
if plot_tk1d:
    pl_tk1d = Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', 
                      ylabel='$\\kappa_{recon} - \kappa_{true}$')
                      

baseline = None
foregrounds = []
for prefix in prefixes:
    if ('cib' not in prefix) and ('ksz' not in prefix) and ('tsz' not in prefix) and ('inpaint' not in prefix):
        baseline = prefix
    else:
        foregrounds.append(prefix)

a=1.    #line opacity
o=0     #offset
da = 0.8
do = 0.1

print("profile diff \t \t n-sigma \t \t pte")

if plot_tk1d:
    tk1d_diff, tk_err_diff, tk_cov_diff = difference(k1d[baseline], tk1d[baseline], all_stack1d[baseline], all_tk1d[baseline])
    tk_chi2_diff, tk_pte_diff = chi_square_pte(tk1d_diff, tk_cov_diff)
    tk_nsigma_diff = np.sqrt(tk_chi2_diff)
    print(f"{baseline}-true \t \t {tk_nsigma_diff} \t \t {tk_pte_diff}")
    pl_tk1d.add_err(r[baseline]+o, tk1d_diff, yerr=tk_err_diff, ls="-", label = labeling(baseline))
    a*=da
    o+=do


for i,fg in enumerate(foregrounds):
    label = labeling(fg)

    color = colors[i+1]
    r_diff = r[fg]-r[baseline]
    assert np.sum(r_diff) == 0.
    k1d_diff, err_diff, cov_diff = difference(k1d[fg], k1d[baseline], all_stack1d[fg], all_stack1d[baseline])
    chi2_diff, pte_diff = chi_square_pte(k1d_diff, cov_diff)
    nsigma_diff = np.sqrt(chi2_diff)
    print(f"{fg}-{baseline} \t \t {nsigma_diff} \t \t {pte_diff}")
    pl_data.add_err(r[baseline]+o, k1d_diff, yerr=err_diff, ls="-", label=label, alpha=a, color=color)
    # if plot_mf:
    #     mf1d_diff, mf_err_diff = difference(all_mf1d[fg], all_mf1d[baseline])
    #     pl_mf.add_err(r[baseline]+o, mf1d_diff, yerr=err_diff, ls="--", label=f"{label} relative mf", alpha=a, color=color)
    if plot_tk1d:
        tk1d_diff, tk_err_diff, tk_cov_diff = difference(k1d[fg], tk1d[baseline], all_stack1d[fg], all_tk1d[baseline])
        tk_chi2_diff, tk_pte_diff = chi_square_pte(tk1d_diff, tk_cov_diff)
        tk_nsigma_diff = np.sqrt(tk_chi2_diff)
        print(f"{fg}-true \t \t {tk_nsigma_diff} \t \t {tk_pte_diff}")
        pl_tk1d.add_err(r[baseline]+o, tk1d_diff, yerr=tk_err_diff, ls="-", label=label, alpha=a, color=color)
    a*=da
    o+=do
pl_data.hline(y=0)
pl_data._ax.set_title(title_text)
pl_data.done(f"{stack_path}/{savename}_rel_1rkappa_profile.png")

# if plot_mf:
#     pl_mf.hline(y=0)
#     pl_mf.done(f"{stack_path}/{savename}_rel_mf_1rkappa_profile.png")

if plot_tk1d:
    pl_tk1d.hline(y=0)
    pl_tk1d._ax.set_title(title_text)
    pl_tk1d.done(f"{stack_path}/{savename}_rel_1rkappa_1tkappa_profile.png")

print("all plots made and saved!")