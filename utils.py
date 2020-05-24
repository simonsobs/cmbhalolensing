from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,wcsutils,utils as putils
import numpy as np
import os,sys
import warnings

try:
    p = io.config_from_yaml("input/paths_local.yml")
except:
    print("No paths_local.yml file found. Please copy paths.yml to paths_local.yml and edit with your local paths. Do not add the latter file to the git tree.")
    raise



def load_dumped_stats(mvstr,get_extra=False):
    savedir = p['scratch'] + f"/{mvstr}/"
    assert os.path.exists(savedir), f"The path corresponding to {savedir} does not exist. If this is a meanfield, are you sure the parameters for your current run match the parameters in any existing meanfield directories?" 
    s = stats.load_stats(f'{savedir}')
    shape,wcs = enmap.read_map_geometry(f'{savedir}/map_geometry.fits')
    if get_extra:
        kmask = enmap.read_map(f'{savedir}/kmask.fits')
        modrmap = enmap.read_map(f'{savedir}/modrmap.fits')
        bin_edges = np.loadtxt(f'{savedir}/bin_edges.txt')
        assert wcsutils.equal(kmask.wcs,modrmap.wcs)
        assert wcsutils.equal(kmask.wcs,wcs)
        return s, shape, wcs, kmask, modrmap, bin_edges
    else:
        return s, shape, wcs



def analyze(s,wcs):
    N_stamp = s.vectors['kw'].shape[0]
    V1 = s.vectors['kw'].sum()
    V2 = s.vectors['kw2'].sum()
    kmap = enmap.enmap(s.stacks['kmap']*N_stamp / V1,wcs)

    try:
        unweighted_stack = enmap.enmap(s.stacks['ustack'],wcs)
    except:
        unweighted_stack = None

    nmean_weighted_kappa_stack = kmap.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kstack = enmap.enmap((s.stacks['wk_real'] + 1j*s.stacks['wk_imag']) / s.stacks['wk_iwt'],wcs)
    kstack[~np.isfinite(kstack)] = 0
    kmap = enmap.ifft(kstack,normalize='phys').real
    opt_weighted_kappa_stack = kmap.copy()


    opt_binned = s.vectors['wk1d'].sum(axis=0) / V1
    diff = s.vectors['k1d'] - opt_binned
    cov = np.dot((diff * s.vectors['kw']).T,diff) / (V1-(V2/V1))
    opt_covm = cov/N_stamp
    opt_corr = stats.cov2corr(opt_covm)
    opt_errs = np.sqrt(np.diag(opt_covm))

    binned = s.stats['k1d']['mean']
    covm = s.stats['k1d']['covmean']
    corr = stats.cov2corr(s.stats['k1d']['covmean'])
    errs = s.stats['k1d']['errmean']

    return unweighted_stack,nmean_weighted_kappa_stack,opt_weighted_kappa_stack,opt_binned,opt_covm,opt_corr,opt_errs,binned,covm,corr,errs
    
    


def plot(fname,stamp,stamp_width_arcmin,tap_per,pad_per,crop=None):
    kmap = stamp
    trimy = int((tap_per+pad_per)/100. * kmap.shape[0])
    trimx = int((tap_per+pad_per)/100. * kmap.shape[1])
    tmap = kmap[trimy:-trimy,trimx:-trimx]
    if crop is not None:
        tmap = maps.crop_center(tmap,crop)
    zfact = tmap.shape[0]*1./kmap.shape[0]
    twidth = tmap.extent()/putils.arcmin
    pwidth = stamp_width_arcmin*zfact
    print("Reported extent : ", twidth)
    print("Plotted extent : ",pwidth)
    io.plot_img(tmap,fname, flip=False, ftsize=12, ticksize=10,arc_width=pwidth,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')


def get_hdv_cc():
    from szar import counts
    ombh2 = 0.0223
    om = 0.24
    h = 0.73
    ns = 0.958
    omb = ombh2 / h**2
    omc = om - omb
    omch2 = omc * h**2.
    As = cosmology.As_from_s8(sigma8 = 0.76,bounds=[1.9e-9,2.5e-9],rtol=1e-4,omegab = omb, omegac = omc, ns = ns, h = h)
    print(As)
    params = {}
    params['As'] = As
    params['H0'] = h * 100.
    params['omch2'] = omch2
    params['ombh2'] = ombh2
    params['ns'] = ns
    params['mnu'] = 0.0

    conc = 3.2
    cc = counts.ClusterCosmology(params,skipCls=True,skipPower=True,skip_growth=True)
    return cc


class Simulator(object):
    
    def __init__(self,stamp_width_arcmin,pix_arcmin):

        """
        
        """
        pass

        
