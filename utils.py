from __future__ import print_function
from orphics import maps,io,cosmology,stats,catalogs
from pixell import enmap,wcsutils,utils as putils
import numpy as np
import os,sys,re
import warnings
from astropy.io import fits
from enlib import bench

try:
    p = io.config_from_yaml("input/paths_local.yml")
except:
    print("No paths_local.yml file found. Please copy paths.yml to paths_local.yml and edit with your local paths. Do not add the latter file to the git tree.")
    raise

plc_beam_fwhm = 5.0


def catalog_interface(cat_type,is_meanfield,nmax=None):
    if cat_type=='hilton_beta' or (cat_type=='hilton_bcg_merged' and is_meanfield):
        if args.is_meanfield:
            catalogue_name = p['data']+ 'selection/S18d_202003Mocks_DESSNR6Scaling/mockCatalog_combined.fits'
        else:
            catalogue_name = p['data']+ 'AdvACT_S18Clusters_v1.0-beta.fits'
        hdu = fits.open(catalogue_name)
        ras = hdu[1].data['RADeg'][:nmax]
        decs = hdu[1].data['DECDeg'][:nmax]
    elif (cat_type=='hilton_bcg_merged'):
        assert not(is_meanfield)
        import pandas as pd
        catalogue_name = p['data']+ 'AdvACT_S18Clusters_v1.0-beta_bcg_merged.csv'
        df = pd.read_csv(catalogue_name)
        ras = df['ra'].to_numpy()
        decs = df['dec'].to_numpy()
        bra = df['bra'].to_numpy()
        bdec = df['bdec'].to_numpy()
        ras[bra>-98] = bra[bra>-98]
        decs[bra>-98] = bdec[bra>-98]
        ras = ras[:nmax]
        decs = decs[:nmax]
    elif cat_type=='sdss_redmapper':
        if is_meanfield:
            catalogue_name = p['data']+ 'redmapper_dr8_public_v6.3_randoms.fits'
        else:
            catalogue_name = p['data']+ 'redmapper_dr8_public_v6.3_catalog.fits'
        hdu = fits.open(catalogue_name)
        ras = hdu[1].data['RA']
        decs = hdu[1].data['DEC']
        ras = ras[decs<25]
        decs = decs[decs<25]
        ras = ras[:nmax]
        decs = decs[:nmax]
    elif cat_type=='cmass':
        with bench.show("load cmass"):
            if is_meanfield:
                # One random has 50x, more than enough for mean-fields.
                boss_files = [p['boss_data']+x for x in  ['random0_DR12v5_CMASS_North.fits','random0_DR12v5_CMASS_South.fits']]
            else:
                boss_files = [p['boss_data']+x for x in  ['galaxy_DR12v5_CMASS_North.fits','galaxy_DR12v5_CMASS_South.fits']]
            ras,decs,_ = catalogs.load_boss(boss_files,zmin=0.4,zmax=0.7,do_weights=False)
            ras = ras[decs<25]
            decs = decs[decs<25]
            ras = ras[:nmax]
            decs = decs[:nmax]
    else:
        raise NotImplementedError
        
    return ras,decs

def load_beam(freq):
    if freq=='f150': fname = p['data']+'s16_pa2_f150_nohwp_night_beam_tform_jitter.txt'
    elif freq=='f090': fname = p['data']+'s16_pa3_f090_nohwp_night_beam_tform_jitter.txt'
    ls,bls = np.loadtxt(fname,usecols=[0,1],unpack=True)
    assert ls[0]==0
    bls = bls / bls[0]
    return maps.interp(ls,bls)



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


def get_seed(tag,task,is_meanfield):
    if tag=='lensed':
        return (0,task)
    elif tag=='mf':
        return (1,task)
    else:
        i = 1 if is_meanfield else 0
        if tag=='noise_plc':
            return (2,i,task)
        elif tag=='noise_act_150':
            return (3,i,task)
        elif tag=='noise_act_90':
            return (4,i,task)

class Simulator(object):
    
    def __init__(self,is_meanfield,stamp_width_arcmin,pix_arcmin,lensed_version,
                 plc_rms,act_150_rms,act_90_rms):

        """
        
        """
        self.plc_rms=plc_rms 
        self.act_150_rms=act_150_rms 
        self.act_90_rms=act_90_rms
        bfact = float(re.search(rf'bfact_(.*?)_pfact', lensed_version).group(1))
        npix = int(stamp_width_arcmin * bfact /  pix_arcmin)
        self.dnpix = int(stamp_width_arcmin / (pix_arcmin))
        shape,wcs = enmap.geometry(pos=(0,0),res=putils.arcmin * pix_arcmin,shape=(npix,npix),proj='plain')
        cshape,cwcs = enmap.geometry(pos=(0,0),res=putils.arcmin * pix_arcmin,shape=(self.dnpix,self.dnpix),proj='plain')
        self.cwcs = cwcs
        self.ipsizemap = enmap.pixsizemap(cshape,cwcs)
        theory = cosmology.default_theory()
        self.shape,self.wcs = shape,wcs
        self.modlmap = enmap.modlmap(shape,wcs)
        self.is_meanfield = is_meanfield
        self.planck_beam = maps.gauss_beam(self.modlmap,plc_beam_fwhm)
        wy, wx = enmap.calc_window(self.shape)
        act_pixwin   = wy[:,None] * wx[None,:]
        self.act_150_beam = load_beam('f150')(self.modlmap) * act_pixwin
        self.act_90_beam = load_beam('f090')(self.modlmap) * act_pixwin
        if self.is_meanfield:
            ucltt = theory.uCl('TT',self.modlmap)
            self.mgen = maps.MapGen((1,)+self.shape,self.wcs,ucltt[None,None])
        else:
            self.savedir = p['scratch'] + f"/{lensed_version}/"
            
    def load_kmap(self,task):
        if self.is_meanfield:
            return self.mgen.get_map(seed=get_seed("mf",task,self.is_meanfield),harm=True)[0]
        else:
            kreal = enmap.read_map(f'{self.savedir}lensed_kmap_real_{task:06d}.fits',sel=np.s_[0,...])
            kimag = enmap.read_map(f'{self.savedir}lensed_kmap_imag_{task:06d}.fits',sel=np.s_[0,...])
            assert wcsutils.equal(kreal.wcs,self.wcs)
            assert wcsutils.equal(kimag.wcs,self.wcs)
            return enmap.enmap(kreal + 1j*kimag,self.wcs)

    def apply_pix_beam_slice(self,kmap,exp):
        if exp=='plc':
            beam = self.planck_beam
        elif exp=='act_150':
            beam = self.act_150_beam
        elif exp=='act_90':
            beam = self.act_90_beam
        ret = maps.crop_center(enmap.ifft(kmap * beam,normalize='phys').real,self.dnpix)
        assert wcsutils.equal(ret.wcs,self.cwcs)
        return ret
        
    def get_obs(self,task):
        kmap = self.load_kmap(task) 
        kmap *= kmap.pixsize()**0.5 # apply physical normalization, since this is turned off in make_lensed_sims.py and MapGen
        imap_plc = self.apply_pix_beam_slice(kmap,'plc')
        imap_act_150 = self.apply_pix_beam_slice(kmap,'act_150')
        imap_act_90 = self.apply_pix_beam_slice(kmap,'act_90')
        
        shape,wcs = imap_plc.shape,imap_plc.wcs
        noise_planck = maps.white_noise(shape,wcs,self.plc_rms,seed=get_seed("noise_plc",task,self.is_meanfield),ipsizemap=self.ipsizemap)
        noise_act_150 = maps.white_noise(shape,wcs,self.act_150_rms,seed=get_seed("noise_act_150",task,self.is_meanfield),ipsizemap=self.ipsizemap)
        noise_act_90 = maps.white_noise(shape,wcs,self.act_90_rms,seed=get_seed("noise_act_90",task,self.is_meanfield),ipsizemap=self.ipsizemap)

        return imap_plc + noise_planck, imap_act_150 + noise_act_150, imap_act_90 + noise_act_90

