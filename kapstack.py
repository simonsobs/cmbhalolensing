import os,sys
import numpy as np
import healpy as hp
from orphics import maps,io,interfaces as ints,lensing,stats
from pixell import utils,bench,enmap

fname = sys.argv[1]

agora_unlensed = True
agora_lensed = True

rstamp = 30.0 * utils.arcmin
res = 0.25 * utils.arcmin

with bench.show("load kappa"):
    kappa = hp.read_map("/data5/sims/agora_sims/cmbkappa/raytrace16384_ip20_cmbkappa.zs1.kg1g2_highzadded_lowLcorrected.fits")

z_min = 0.43
z_max = 0.70
mmin = 0.677e13
mmax = 6.77e13
nmax = 5000

lmin = 200
lmax = 20000
rmin = 0.
rmax = 10*utils.arcmin
rwidth = 1.0 * utils.arcmin

def get_stack(lensed=False):
    oras,odecs,ozs,oms = ints.get_agora_halos(z_min = z_min, z_max = z_max,
                        mass_min = mmin, mass_max = mmax,
                        massdef='m200', lensed=lensed) # Get agora halos

    num = len(oras)
    if num > nmax:
        cut = np.random.randint(0, num, nmax)
        oras = oras[cut]
        odecs = odecs[cut]
        ozs = ozs[cut]
        oms = oms[cut]

    print(f"Starting stack on {len(oras)}")
    out = 0.
    for i,(ora,odec) in enumerate(zip(oras,odecs)):
        othumb = maps.thumbnail_healpix(kappa,np.asarray((odec,ora))*utils.degree,r=rstamp,res=res) # reproject directly from healpix
        if i%100==0: print(f"Done {i+1} / {len(oras)}")
        out = out + othumb

    out = out / len(oras)

    cents,kap_agora = lensing.filter_bin_kappa2d(out,lmin=lmin,lmax=lmax,rmin=0.,rmax=rmax,rwidth=rwidth)

    return cents, kap_agora, ozs, oms, out


if agora_unlensed:
    agora_unl_cents, agora_unl_kap, ozs, oms, agora_unl_out = get_stack(lensed=False)
if agora_lensed:
    agora_lens_cents, agora_lens_kap, ozs, oms, agora_lens_out = get_stack(lensed=True)

z = ozs.mean()
ez = ozs.std()
m200c = oms.mean()
em200c = oms.std()

thetas,kappa_1h,kappa_2h,tot_kappa,cents_t,b1d1h,b1d2h,b1d_t = lensing.kappa_nfw_profiley(mass=m200c,conc=None,
                                                                                          z=z,z_s=1100.,background='critical',delta=200,apply_filter=True,
                                                                                          lmin=lmin,lmax=lmax,res=res,
                                                                                          rstamp=rstamp,rmin=rmin,rmax=rmax,rwidth=rwidth)
pl = io.Plotter(xyscale='linlin',xlabel='$\\theta$ (arcmin)',ylabel='$\\kappa$', 
                title=f"M={m200c/1e14:.1f} $\\pm$ {em200c/1e14:.1f} ;  z={z:.1f} $\\pm$ {ez:.1f}")
if agora_unlensed:
    pl.add(agora_unl_cents/utils.arcmin,agora_unl_kap,marker='o',ls='-',color='r',label=f'Agora unlensed; N={nmax}')
if agora_lensed:
    pl.add(agora_lens_cents/utils.arcmin,agora_lens_kap,marker='o',ls='--',color='r',label=f'Agora lensed; N={nmax}')
pl.add(cents_t/utils.arcmin,b1d_t,marker='o',ls='--',color='k',label='profiley theory')
pl.hline(y=0)
pl.legend(fontsize="x-small", loc="upper right")
pl._fig.savefig(f'tkappa_tests/{fname}_profile.png', bbox_inches='tight')


if agora_unlensed:
    enmap.write_map(f"tkappa_tests/agora_unlensed_stack.fits",agora_unl_out)
if agora_lensed:
    enmap.write_map(f"tkappa_tests/agora_lensed_stack.fits",agora_lens_out)