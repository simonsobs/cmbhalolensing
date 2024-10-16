import camb
import halo_funcs as hf
import matplotlib.pyplot as plt
import numpy as np
from orphics import io, lensing
from pixell import utils as u
from profiley.nfw import TNFW
from halo_funcs import TunableNFW as NFW



H0 = 67.5
h=H0/100
z_cmb = np.asarray([1100.])

pars = camb.set_params(H0=H0, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,  
                       As=2e-9, ns=0.965, halofit_version='mead', lmax=3000)
camb_results = camb.get_results(pars)

c = np.asarray([5.])
M = np.asarray([2.e13])
zs = np.asarray([0.55])

comL = camb_results.angular_diameter_distance(zs[0])
Rs = np.geomspace(0.01, 10, 1000)
thetas = Rs/comL/u.arcmin
print("Rs min max", Rs.min(), Rs.max())
print("theta min max", thetas.min(), thetas.max())
print("comL:",comL)

bins = np.asarray([Rs.min(), Rs.max()])
amps = np.asarray([[1.]])

data_theta, data = np.loadtxt("/home3/nehajo/projects/cmbhalolensing/results/post/cmass_dr6maps/cmass_dr6maps_opt_profile.txt", unpack = True)
_, err = np.loadtxt("/home3/nehajo/projects/cmbhalolensing/results/post/cmass_dr6maps/cmass_dr6maps_opt_profile_errs.txt", unpack = True)



rho_pl = io.Plotter(xyscale='loglog', xlabel='R [Mpc]', ylabel='$\\rho$ (R)')
sigma_pl = io.Plotter(xyscale='loglin', xlabel='R [Mpc]', ylabel='$\\ R x \Delta \Sigma$ [Mpc M$_\odot$ / pc$^2$]')
k_pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$ (R)')

k_pl.add_err(data_theta, data, err, label = "data")

for amp in amps:
    nfw = NFW(M, c, zs, bins, amp)
    rho = nfw.profile(Rs)
    # rho = nfw.tuned_profile(Rs, comL*bin_edges, alphas)
    sigma = nfw.projected_cumulative(Rs)
    dsigma = nfw.projected_excess(Rs)
    kappa = nfw.convergence(Rs, z_cmb)


    rho_pl.add(Rs, rho, label=str(amp))
    sigma_pl.add(Rs, Rs*dsigma.flatten()/1.e12, label=str(amp)) #same units as weak lensing gg
    k_pl.add(thetas, kappa, label="model")

k_pl._ax.set_ylim(-0.001,0.02)
k_pl.hline(y=0)
# rho_pl.done("/home3/nehajo/projects/cmbhalolensing/tests/rho_test.png")
# sigma_pl.done("/home3/nehajo/projects/cmbhalolensing/tests/sigma_test.png")
k_pl.done("/home3/nehajo/projects/cmbhalolensing/tests/kappa_data_comp.png")
print("plots done")