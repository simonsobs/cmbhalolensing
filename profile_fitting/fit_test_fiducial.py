import camb
from halo_funcs import TunableNFW as profile, filter_profile
import matplotlib.pyplot as plt
import numpy as np
from orphics import io
from pixell import utils as u, enmap

amps = [[1., 1., 1.]]


data_path = "/home3/nehajo/projects/cmbhalolensing/results/post/cmass_dr6simple/cmass_dr6simple_opt_profile.txt"
covmat_path = "/home3/nehajo/projects/cmbhalolensing/results/post/cmass_dr6simple/cmass_dr6simple_opt_covm.txt"
map_geom = "/home3/nehajo/projects/cmbhalolensing/results/post/cmass_dr6simple/map_geometry.fits"
bin_edges = "/home3/nehajo/projects/cmbhalolensing/results/post/cmass_dr6simple/cmass_dr6simple_bin_edges.txt"
savedir = "/home3/nehajo/projects/cmbhalolensing/profile_fitting/plots/full_filt_test/"
### set constants
H0 = 67.5
h=H0/100
z_cmb = np.asarray([1100.])

c = 5.
M = 10**13.3
zmean = 0.547
two_halo = [0]#[10**10]

min_R = 0.01
max_R = 15
num_R = 10000
inner_edges = [0.5, 2.5]

Lmin = 200
Lmax = 5000

pars = camb.set_params(H0=H0, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,  
                       As=2e-9, ns=0.965, halofit_version='mead', lmax=3000)
camb_results = camb.get_results(pars)
comL = camb_results.angular_diameter_distance(zmean)

theta_x, data = np.loadtxt(data_path, unpack=True)
covmat = np.loadtxt(covmat_path, unpack=True)
err = np.sqrt(np.diag(covmat))
data_bins = np.loadtxt(bin_edges, unpack=True)

shape, wcs = enmap.read_map_geometry(map_geom)
modrmap = enmap.modrmap(shape, wcs)
R_edge = np.max(modrmap)*comL
Rs = np.geomspace(min_R, R_edge, num_R)
thetas = Rs/comL/u.arcmin

Rs_bins = np.zeros(len(inner_edges)+2)
Rs_bins[0], Rs_bins[1:-1], Rs_bins[-1] = min_R, inner_edges, R_edge

pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [Mpc]', ylabel='$\\rho$ (R)')
pl.add_err(theta_x, data, yerr=err, label="data", color="black", ls="--")


print(Rs.max())
for amp in amps:
    for th in two_halo:
        prof = profile(M, c, zmean, Rs_bins, amp, th)
        kappa = prof.convergence(Rs, z_cmb).flatten()

        fig, ax = plt.subplots()
        ax.plot(thetas, kappa, label="unfiltered")



        cents, kf1d = filter_profile(kappa, thetas, data_bins, Lmin, Lmax, shape, wcs, plot=True, savedir=savedir)
        ax.plot(cents, kf1d, label="filtered")
        ax.set_xlabel("arcmin")
        ax.set_ylabel("kappa")
        ax.set_ylim((-0.001, 0.02))
        ax.set_xlim((0,15))
        ax.axhline(y=0, ls="dotted", color = "gray")
        fig.legend()
        fig.savefig(savedir + "nfw_kappa.png")
        plt.close(fig)
        pl.add(cents, kf1d, label=f"amps {amp}, log10(2 halo): {np.log10(th)}")
pl.hline(y=0)

pl.done("plots/test_amps_bounds.png")
# io.save_cols("/home3/nehajo/projects/cmbhalolensing/profile_fitting/sim_profile2.txt", (cents, kf1d))
