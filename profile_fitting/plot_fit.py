import camb
from cobaya import load_samples
from halo_funcs import TunableNFW as NFW, filter_profile
from getdist import loadMCSamples, plots
import matplotlib.pyplot as plt
import numpy as np
from orphics import io, maps, stats
from pixell import utils as u, enmap

### paths
data_path = "/home3/nehajo/projects/cmbhalolensing/results/post/cmass_dr6simple/cmass_dr6simple_opt_profile.txt"
err_path = "/home3/nehajo/projects/cmbhalolensing/results/post/cmass_dr6simple/cmass_dr6simple_opt_profile_errs.txt"
chains = "/home3/nehajo/projects/cmbhalolensing/profile_fitting/chains/3bins2/3bins2"
map_geom = "/home3/nehajo/projects/cmbhalolensing/results/post/cmass_dr6simple/map_geometry.fits"
savedir = "/home3/nehajo/projects/cmbhalolensing/profile_fitting/plots/3bins2/3bins2_"
io.mkdir(savedir.rsplit("/", 1)[0])

### set constants
H0 = 67.5
h=H0/100
z_cmb = np.asarray([1100.])

min_R = 0.01
max_R = 15
num_R = 10000
inner_edges = [0.5, 2.5]

Lmin = 200
Lmax = 5000


pars = camb.set_params(H0=H0, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,  
                       As=2e-9, ns=0.965, halofit_version='mead', lmax=3000)
camb_results = camb.get_results(pars)

c = 5.
M = 10**13.3
zs = 0.547
two_halo = 0.

### load data
data_theta, data = np.loadtxt(data_path, unpack = True)
_, err = np.loadtxt(err_path, unpack = True)
shape, wcs = enmap.read_map_geometry(map_geom)
print(data_theta)
modrmap = enmap.modrmap(shape,wcs)

### load sampling fits
def param_bounds(samples, param, limit=1):
    
    #Get the param bounds from the latex output of getdist
    #returns the prior bounds if no upper/lower limit
    
    #inputs:
    #    samples: (MCSamples object)
    #    param: (string) parameter name
    #    limit: (int) 1, 2, 3 for 1, 2, 3 sigma bounds
    #
    #outputs:
    #    low: (float) lower bound on parameter
    #    high: (float) lower bound on parameter
    
    try:
        string = samples.getInlineLatex(param, limit=limit)
    except:
        return np.nan, np.nan # if param not varied, don't plot
    
    if 'pm' in string:
        mean, std = string.split("= ")[1].split("\\pm ")
        mean  = float(mean)
        std = float(std)
        high = mean + std
        low = mean - std
    
    elif '^' in string:
        mean = string.split("= ")[1].split("^")[0]
        upper, lower = string.split("^{+")[1].split("}_{-")
        lower = float(lower[:-1])
        upper = float(upper)
        mean  = float(mean)
        high = mean + upper
        low = mean - lower
    
    elif '>' in string:
        low = string.split("> ")[1]
        low = float(low)
        high = samples.ranges.getUpper(param) # use prior bound
    
    elif '<' in string:
        high = string.split("< ")[1]
        high = float(high)
        low = samples.ranges.getLower(param) # use prior bound
        
    else:
        high = samples.ranges.getUpper(param) # use prior bound
        low = samples.ranges.getLower(param) # use prior bound
        
    return low, high

samples = loadMCSamples(chains, settings={'ignore_rows':0.3})
means = samples.getMeans()[:-2]
full_chain = load_samples(chains, skip=0.33)
print(full_chain)


stat = samples.getParamBestFitDict()
params = list(stat.keys())[:-5]

bf = [stat[param] for param in params]
print("best fits:", bf)

covmat = samples.getCovMat().matrix
fit_err = np.sqrt(np.diag(covmat))

upper = np.zeros(means.size)
lower = np.zeros(means.size)
for i in range(means.size):
    name = params[i]
    lower[i], upper[i] = param_bounds(samples, name)

print("means:", means)
print("upper:", upper)
print("lower:", lower)

gdplot = plots.get_subplot_plotter()
gdplot.triangle_plot([samples], params=params, filled=True)
for a in range(len(bf)):
    ax = gdplot.subplots[a,a]
    ax.axvline(bf[a], color="black", ls='--')
gdplot.export(savedir+"triangle_plot.png")

### compute
comL = camb_results.angular_diameter_distance(zs)
R_edge = np.max(modrmap)*comL
if R_edge > max_R:
    max_R = R_edge
Rs = np.geomspace(min_R, max_R, num_R)
thetas = Rs/comL/u.arcmin
print("Rs min max", Rs.min(), Rs.max())
print("theta min max", thetas.min(), thetas.max())
print("comL:",comL)

data_bins = (data_theta[1:] + data_theta[:-1])/2 # bins in arcmin
model_bins = np.zeros(len(data_bins)+2)
model_bins[0] = 0.
model_bins[-1] = data_bins[-1] + data_bins[0]
model_bins[1:-1] = data_bins
data_Rs = data_bins/comL/u.arcmin

Rs_bins = np.zeros(len(inner_edges) + 2)
Rs_bins[1:-1] = np.asarray(inner_edges)
Rs_bins[0] = min_R
Rs_bins[-1] = max_R

theta_bins = Rs_bins*comL*u.arcmin


rho_pl = io.Plotter(xyscale='loglog', xlabel='R [Mpc]', ylabel='$\\R x rho$ (R)')
rdsigma_pl = io.Plotter(xyscale='loglin', xlabel='R [Mpc]', ylabel='$\\ R x \Delta \Sigma$ [Mpc M$_\odot$ / pc$^2$]')
dsigma_pl = io.Plotter(xyscale='loglog', xlabel='R [Mpc]', ylabel='$\\Delta \Sigma$ [M$_\odot$ / pc$^2$]')
k_pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$ (R)')
k_binned_pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$ (R)')

k_pl.add_err(data_theta, data, err, label = "data", color="black")
k_binned_pl.add_err(data_theta, data, err, label = "data", color="black")


def theory_binner(binned_profile, Rs, bins):
    theory = np.zeros(Rs.size)
    for i in range(len(bins)-1):
        theory = np.where(np.logical_and(Rs >= bins[i], Rs <= bins[i+1]), binned_profile[i], theory)
    return theory

def profiles(amps):
    nfw = NFW(M, c, zs, Rs_bins, amps, two_halo)
    rho = nfw.profile(Rs)
    dsigma = nfw.projected_excess(Rs).flatten()
    kappa = nfw.convergence(Rs, z_cmb).flatten()
    return rho, dsigma, kappa

for amps in chains:
    rho, dsigma, kappa = profiles(amps)
    rho_pl.add(Rs, rho*Rs, color="gray", alpha = 0.01)
    dsigma_pl.add(Rs, dsigma/1.e12, color = "gray", alpha = 0.01)
    rdsigma_pl.add(Rs, Rs*dsigma/1.e12, color = "gray", alpha = 0.01)


rho_mean, dsigma_mean, kappa_mean = profiles(means)
rho_upper, dsigma_upper, kappa_upper = profiles(upper)
rho_lower, dsigma_lower, kappa_lower = profiles(lower)

# R_binner = stats.bin1D(Rs_bins)
# R_cents, Rs_geom = R_binner.bin(Rs, np.log10(Rs))
# Rs_t = 10**Rs_geom
# Rs_t = theory_binner(data_Rs, Rs, Rs_bins)
# print(data_Rs, "\n", R_cents, "\n", Rs_geom)

# rho_t = theory_binner(rho_mean, Rs, Rs_bins)
# rho_u = theory_binner(rho_upper, Rs, Rs_bins)
# rho_l = theory_binner(rho_lower, Rs, Rs_bins)
# rho_pl.add(Rs, rho_t, label="binned fit", color="tab:purple")
rho_pl.add(Rs, Rs*rho_mean, label="fit", color="tab:purple")
# rho_pl._ax.fill_between(Rs, rho_lower, rho_upper, alpha=0.3, color="tab:purple")

# ds_t = theory_binner(dsigma_mean, Rs, Rs_bins)
# ds_u = theory_binner(dsigma_upper, Rs, Rs_bins)
# ds_l = theory_binner(dsigma_lower, Rs, Rs_bins)
# dsigma_pl.add(Rs, Rs_t*ds_t/1.e12, label="fit", color="tab:purple") #same units as weak lensing gg
rdsigma_pl.add(Rs, Rs*dsigma_mean/1.e12, label="fit", color="tab:purple")
# rdsigma_pl._ax.fill_between(Rs, Rs*dsigma_lower/1.e12, Rs*dsigma_upper/1.e12, alpha=0.3, color="tab:purple")

dsigma_pl.add(Rs, dsigma_mean/1.e12, label="fit", color="tab:purple")
# dsigma_pl._ax.fill_between(Rs, dsigma_lower/1.e12, dsigma_upper/1.e12, alpha=0.3, color="tab:purple")


cents, kf1d_mean = filter_profile(kappa_mean, thetas, model_bins, Lmin, Lmax, shape, wcs, plot=True, savedir=savedir)
_, kf1d_upper = filter_profile(kappa_upper, thetas, model_bins, Lmin, Lmax, shape, wcs)
_, kf1d_lower = filter_profile(kappa_lower, thetas, model_bins, Lmin, Lmax, shape, wcs)
fig, ax = plt.subplots()
k_pl.add(cents, kf1d_mean, label = "filtered fit", color="tab:purple")
# ax.fill_between(cents, kf1d_upper, kf1d_lower, alpha=0.3, color="tab:purple")
# ax.plot(thetas, kf, color="gray")
k_pl._ax.axhline(y=0, color="gray", ls="dotted")
k_pl._ax.set_xlabel(r"$\theta$ [arcmin]")
k_pl._ax.set_ylabel(r"$\kappa$")

k_t = theory_binner(kf1d_mean, thetas, model_bins)
k_u = theory_binner(kf1d_upper, thetas, model_bins)
k_l = theory_binner(kf1d_lower, thetas, model_bins)
k_binned_pl.add(thetas, k_t, label="fit", color="tab:blue")
k_binned_pl.add(thetas, kappa_mean, label="unfiltered fit", alpha=0.3, color="black")
k_binned_pl._ax.fill_between(thetas, k_l, k_u, alpha=0.3, color="tab:blue")
# k_pl._ax.set_ylim(-0.001,0.02)
k_binned_pl._ax.set_xlim(0, 15)
k_binned_pl.hline(y=0)

rho_pl.done(savedir+"rho_fit.png")
rdsigma_pl.done(savedir+"Rxdsigma_fit.png")
dsigma_pl.done(savedir+"dsigma_fit.png")
k_pl.done(savedir+"kappa_fit_vs_data.png")
k_binned_pl.done(savedir+"kappa_binned_vs_data.png")
print("plots done")


### calculate SNR of fit
covmat = np.loadtxt(chains+".covmat")
C_inv = np.linalg.inv(covmat)
chi2 = np.dot(means, np.dot(C_inv, means))
SNR = np.sqrt(chi2)
print("SNR: ", SNR)