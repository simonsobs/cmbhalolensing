from __future__ import print_function
from orphics import maps,io,cosmology,stats,catalogs,lensing
from orphics.mpi import MPI
from pixell import enmap,wcsutils,utils as putils,bunch
import numpy as np
import os,sys,re
import warnings
from astropy.io import fits
from enlib import bench
import argparse
import time

try:
    paths = bunch.Bunch(io.config_from_yaml("input/paths_local.yml"))
except:
    print("No paths_local.yml file found. Please copy paths.yml to paths_local.yml and edit with your local paths. Do not add the latter file to the git tree.")
    raise

defaults = bunch.Bunch(io.config_from_yaml("input/defaults.yml"))


def initialize_pipeline_config():
    start_time = time.time()
    d = defaults
    tags = bunch.Bunch({})

    # Parse command line
    parser = argparse.ArgumentParser(description="Stacked CMB lensing.")
    parser.add_argument("version", type=str, help="Version label.")
    parser.add_argument(
        "cat_type", type=str, help="Catalog path relative to data directory."
    )
    parser.add_argument(
        "-N",
        "--nmax",
        type=int,
        default=None,
        help="Limit number of objects used e.g. for debugging or quick tests, or for sim injections.",
    )
    parser.add_argument(
        "--grad-lmin", type=int, default=d.gradient_lmin, help="Minimum multipole for Planck."
    )
    parser.add_argument(
        "--grad-lmax", type=int, default=d.gradient_lmax, help="Maximum multipole for Planck."
    )
    parser.add_argument(
        "--hres-lmin", type=int, default=None, help="Minimum multipole for ACT."
    )
    parser.add_argument(
        "--hres-lmax", type=int, default=d.highres_lmax, help="Maximum multipole for ACT."
    )
    parser.add_argument(
        "--klmin", type=int, default=d.kappa_Lmin, help="Minimum multipole for recon."
    )
    parser.add_argument(
        "--klmax", type=int, default=d.kappa_Lmax, help="Maximum multipole for recon."
    )
    parser.add_argument("--hres-lxcut", type=int, default=None, help="Lxcut for ACT.")
    parser.add_argument("--hres-lycut", type=int, default=None, help="Lycut for ACT.")
    parser.add_argument(
        "--zmin", type=float, default=None, help="Minimum redshift."
    )
    parser.add_argument(
        "--zmax", type=float, default=None, help="Maximum redshift."
    )
    parser.add_argument(
        "--snmin", type=float, default=None, help="Minimum SNR."
    )
    parser.add_argument(
        "--snmax", type=float, default=None, help="Maximum SNR."
    )
    parser.add_argument(
        "--full-sim-index", type=int, default=None, help="Use full-sky CMB simulations with this index. Defaults to None."
    )
    parser.add_argument(
        "--arcmax", type=float, default=d.arcmax, help="Maximum arcmin distance for binning."
    )
    parser.add_argument(
        "--arcstep", type=float, default=d.arcstep, help="Step arcmin for binning."
    )
    parser.add_argument(
        "--max-rms",
        type=float,
        default=d.max_rms_noise,
        help="Maximum RMS noise in uK-arcmin, beyond which to reject stamps.",
    )
    parser.add_argument(
        "--swidth", type=float, default=d.stamp_width_arcmin, help="Stamp width arcmin."
    )
    parser.add_argument(
        "--pwidth", type=float, default=d.pix_width_arcmin, help="Pixel width arcmin."
    )
    parser.add_argument(
        "--no-fit-noise",
        action="store_true",
        help="If True, do not fit empirical noise, but use RMS values specified in defaults.yml.",
    )
    parser.add_argument(
        "--day-null",
        action="store_true",
        help="Use day-night as data.",
    )
    parser.add_argument("--tap-per", type=float, default=d.taper_percent, help="Taper percentage.")
    parser.add_argument("--pad-per", type=float, default=d.pad_percent, help="Pad percentage.")
    parser.add_argument("--debug-fit", type=str, default=None, help="Which fit to debug.")
    parser.add_argument(
        "--debug-anomalies",
        action="store_true",
        help="Whether to save plots of excluded anomalous stamps.",
    )
    parser.add_argument(
        "--debug-powers",
        action="store_true",
        help="Whether to plot various power spectra from each stamp.",
    )
    parser.add_argument(
        "--debug-nl",
        action="store_true",
        help="Whether to plot Nl for weighting and stop after one cluster.",
    )
    parser.add_argument("--no-90", action="store_true", help="Do not use the 90 GHz map.")
    parser.add_argument("--inpaint", action="store_true", help="Inpaint gradient.")
    parser.add_argument(
        "--no-sz-sub",
        action="store_true",
        help="Use the high-res maps without SZ subtraction.",
    )
    parser.add_argument(
        "--s19",
        action="store_true",
        help="Use preliminary 2019 data.",
    )
    parser.add_argument(
        "--curl",
        action="store_true",
        help="Do curl null test instead of lensing.",
    )
    parser.add_argument(
        "--inject-sim",
        action="store_true",
        help="Instead of using data, simulate a lensing cluster and Planck+ACT (or unlensed for mean-field).",
    )
    parser.add_argument(
        "--lensed-sim-version",
        type=str,
        default=d.lensed_sim_version,
        help="Default lensed sims to inject.",
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing version."
    )
    parser.add_argument(
        "--is-meanfield", action="store_true", help="This is a mean-field run."
    )
    parser.add_argument(
        "--debug-stack", action="store_true", help="Skip reconstruction and just stack on gradient and high-res."
    )
    parser.add_argument(
        "--bcg", action="store_true", help="Use BCGs for Hilton Catalog."
    )
    parser.add_argument(
        "--rand-rot", action="store_true", help="Rotate high-res stamp by random number of 90 degrees as a null test."
    )
    parser.add_argument("--night-only", action="store_true", help="Use night-only maps.")
    parser.add_argument("--full-nl", action="store_true", help="Do not assume estimator is optimal for Nl weighting.")
    parser.add_argument(
        "--act-only-in-hres",
        action="store_true",
        help="Use ACT only maps in high-res instead of ACT+Planck.",
    )
    args = parser.parse_args()

    if args.hres_lmin is None:
        if args.act_only_in_hres:
            setattr(args, 'hres_lmin', d.conservative_highres_lmin)
        else:
            setattr(args, 'hres_lmin', d.aggressive_highres_lmin)

    if args.hres_lycut is None:
        if args.act_only_in_hres:
            setattr(args, 'hres_lycut', d.conservative_highres_lycut)
        else:
            setattr(args, 'hres_lycut', d.aggressive_highres_lycut)

    if args.hres_lxcut is None:
        if args.act_only_in_hres:
            setattr(args, 'hres_lxcut', d.conservative_highres_lxcut)
        else:
            setattr(args, 'hres_lxcut', d.aggressive_highres_lxcut)


    """
    We will save results to a directory in paths.yml:scratch.
    To decide on the name and to ensure that any meanfields we make
    have identical noise properties, we build some strings:
    """

    tags.dstr = "night" if args.night_only else "daynight"
    tags.apstr = "act" if args.act_only_in_hres else "act_planck"
    tags.mstr = "_meanfield" if args.is_meanfield else ""
    tags.n90str = "_no90" if args.no_90 else ""
    tags.s19str = "s19" if args.s19 else "s18"
    curlstr = "_curl" if args.curl else ""
    findstr = f"_{args.full_sim_index:06d}" if not(args.full_sim_index is None) else ""
    if not(args.full_sim_index is None):
        assert args.night_only and not(args.act_only_in_hres), "Full sims only currently for night-only act_planck"

    # The directory name string
    vstr = f"{args.version}_{args.cat_type}_plmin_{args.grad_lmin}_plmax_{args.grad_lmax}_almin_{args.hres_lmin}_almax_{args.hres_lmax}_klmin_{args.klmin}_klmax_{args.klmax}_lxcut_{args.hres_lxcut}_lycut_{args.hres_lycut}_swidth_{args.swidth:.2f}_tapper_{args.tap_per:.2f}_padper_{args.pad_per:.2f}_{tags.dstr}_{tags.apstr}{tags.n90str}_{tags.s19str}{curlstr}{tags.mstr}{findstr}"

    # File save paths
    savedir = paths.scratch + f"/{vstr}/"
    debugdir = paths.scratch + f"/{vstr}/debug/"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    overwrite = args.overwrite
    if not (overwrite):
        assert not (
            os.path.exists(savedir)
        ), "This version already exists on disk. Please use a different version identifier or use the overwrite argument."
    if rank == 0:
        try:
            os.makedirs(savedir)
        except:
            if overwrite:
                pass
            else:
                raise
        try:
            os.makedirs(debugdir)
        except:
            if overwrite:
                pass
            else:
                raise
    comm.Barrier()  # Wait for other processes to catch up with rank=0 before saving to these directories

    paths.debugdir = debugdir
    paths.savedir = savedir
    return start_time,paths,defaults,args,tags,rank

def cut_z_sn(ras,decs,sns,zs,zmin,zmax,snmin,snmax):
    if zmin is not None:
        ras = ras[zs>zmin]
        decs = decs[zs>zmin]
        sns = sns[zs>zmin]
        zs = zs[zs>zmin]
    if zmax is not None:
        ras = ras[zs<=zmax]
        decs = decs[zs<=zmax]
        sns = sns[zs<=zmax]
        zs = zs[zs<=zmax]
    if snmin is not None:
        ras = ras[sns>snmin]
        decs = decs[sns>snmin]
        zs = zs[sns>snmin]
        sns = sns[sns>snmin]
    if snmax is not None:
        ras = ras[sns<=snmax]
        decs = decs[sns<=snmax]
        zs = zs[sns<=snmax]
        sns = sns[sns<=snmax]
    return ras,decs,sns,zs

def catalog_interface(cat_type,is_meanfield,nmax=None,zmin=None,zmax=None,bcg=False,snmin=None,snmax=None):
    data = {}
    if cat_type=='hilton_beta':
        if is_meanfield:
            catalogue_name = paths.data+ 'selection/S18d_202003Mocks_DESSNR6Scaling/mockCatalog_combined.fits'
        else:
            catalogue_name = paths.data+ 'AdvACT_S18Clusters_v1.0-beta.fits'
        hdu = fits.open(catalogue_name)
        if bcg:
            ras = hdu[1].data['opt_RADeg']
            decs = hdu[1].data['opt_DECDeg']
            decs = decs[ras>=0]
            zs = hdu[1].data['redshift'][ras>=0]
            sns = hdu[1].data['SNR'][ras>=0]
            ras = ras[ras>=0]
        else:
            ras = hdu[1].data['RADeg']
            decs = hdu[1].data['DECDeg']
            zs = hdu[1].data['redshift']
            sns = hdu[1].data['SNR']
        ras,decs,sns,zs = cut_z_sn(ras,decs,sns,zs,zmin,zmax,snmin,snmax)
        ras = ras[:nmax]
        decs = decs[:nmax]
        ws = ras*0 + 1
        data['sns'] = sns

    elif cat_type=='sdss_redmapper':
        if is_meanfield:
            catalogue_name = paths.data+ 'redmapper_dr8_public_v6.3_randoms.fits'
        else:
            catalogue_name = paths.data+ 'redmapper_dr8_public_v6.3_catalog.fits'
        hdu = fits.open(catalogue_name)
        ras = hdu[1].data['RA']
        decs = hdu[1].data['DEC']
        zs = hdu[1].data['Z_LAMBDA']
        lams = hdu[1].data['LAMBDA']
        ras = ras[decs<25]
        zs = zs[decs<25]
        lams = lams[decs<25]
        decs = decs[decs<25]
        ras = ras[:nmax]
        decs = decs[:nmax]
        zs = zs[:nmax]
        lams = lams[:nmax]
        ws = ras*0 + 1
        data['lams'] = lams

    elif cat_type=='des_redmapper':
        if is_meanfield:
            catalogue_name = paths.data+ 'y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_randcat_z0.10-0.95_lgt020_vl02.fit'
        else:
            catalogue_name = paths.data+ 'y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_lgt20_vl02_catalog.fit'
        hdu = fits.open(catalogue_name)
        ras = hdu[1].data['RA']
        decs = hdu[1].data['DEC']
        zs = hdu[1].data['Z_LAMBDA' if not(is_meanfield) else 'ZTRUE']
        sns = hdu[1].data['LAMBDA_CHISQ' if not(is_meanfield) else 'LAMBDA_IN']

        ras,decs,sns,zs = cut_z_sn(ras,decs,sns,zs,zmin,zmax,snmin,snmax)

        ras = ras[:nmax]
        decs = decs[:nmax]
        zs = zs[:nmax]
        sns = sns[:nmax]
        ws = ras*0 + 1
        data['lams'] = sns

    elif cat_type[:5]=='cmass':
        scat = cat_type.split('_')
        if len(scat)==1: raise ValueError("Please specify CMASS catalog as cmass_dr11 or cmass_dr12.")
        dr = scat[1].lower()
        if dr=='dr11':
            broot = paths.boss_dr11_data
            fstr = 'DR11v1'
        elif dr=='dr12':
            broot = paths.boss_dr12_data
            fstr = 'DR12v5'
        if is_meanfield:
            # One random has 50x, more than enough for mean-fields.
            boss_files = [broot+x for x in  [f'random0_{fstr}_CMASS_North.fits',f'random0_{fstr}_CMASS_South.fits']]
        else:
            boss_files = [broot+x for x in  [f'galaxy_{fstr}_CMASS_North.fits',f'galaxy_{fstr}_CMASS_South.fits']]
        if zmin is None: zmin = 0.43
        if zmax is None: zmax = 0.70
        ras,decs,ws,zs = catalogs.load_boss(boss_files,zmin=zmin,zmax=zmax,do_weights=not(is_meanfield),sys_weights=False)
        if ws is None: ws = ras*0 + 1
        ws = ws[decs<25]
        ras = ras[decs<25]
        zs = zs[decs<25]
        decs = decs[decs<25]
        if nmax is not None:
            """
            We have to be a bit more careful when a max number of random galaxies is requested for BOSS, because
            there is a North/South split.
            """
            Ntot = len(ras)
            np.random.seed(100)
            inds = np.random.choice(Ntot,size=nmax,replace=False)
            ras = ras[inds]
            decs = decs[inds]
            ws = ws[inds]
            zs = zs[inds]

    elif cat_type=='wise_panstarrs':
        if is_meanfield:
            # made using mapcat.py followed by randcat.py
            catalogue_name = paths.data+ 'wise_panstarrs_randoms.txt'
        else:
            catalogue_name = paths.data+ 'wise_panstarrs_radec.txt'
        ras,decs = np.loadtxt(catalogue_name,unpack=True)
        ras = ras[:nmax]
        decs = decs[:nmax]
        ws = ras*0 + 1

    elif cat_type=='madcows_photz':
        if is_meanfield:
            # made using mapcat.py followed by randcat.py
            catalogue_name = paths.data+ 'madcows_photz_randoms.txt'
            ras,decs = np.loadtxt(catalogue_name,unpack=True)
            zs = ras*0
        else:
            catalogue_name = paths.data+ 'madcows_cleaned.txt'
            ras,decs,zs,sns = np.genfromtxt(catalogue_name,usecols=[2,3,6,8],unpack=True,delimiter=',')
            ras = ras[zs>0]
            decs = decs[zs>0]
            sns = sns[zs>0]
            zs = zs[zs>0]

            ras = ras[sns>0]
            decs = decs[sns>0]
            zs = zs[sns>0]
            sns = sns[sns>0]

            ras,decs,sns,zs = cut_z_sn(ras,decs,sns,zs,zmin,zmax,snmin,snmax)


            sns = sns[:nmax]
            data['lams'] = sns

        zs = zs[:nmax]
        ras = ras[:nmax]
        decs = decs[:nmax]
        ws = ras*0 + 1

    elif cat_type=='hsc_camira':
        if is_meanfield:
            catalogue_name = paths.data+ 'rand_comb_s19a_wide_sm_z084.dat'
            ras,decs = np.loadtxt(catalogue_name,unpack=True)
            sns = None
            zs = ras*0
        else:
            catalogue_name = paths.data+ 'camira_s19a_wide_sm_v1_01z11.dat'
            ras,decs,sns,zs = np.loadtxt(catalogue_name,unpack=True)
            ras,decs,sns,zs = cut_z_sn(ras,decs,sns,zs,zmin,zmax,snmin,snmax)

        ras = ras[:nmax]
        decs = decs[:nmax]
        zs = zs[:nmax]
        if not(is_meanfield):
            sns = sns[:nmax]
            data['lams'] = sns
        ws = ras*0 + 1

    elif cat_type=='vrec_cmass':
        ras,decs,zs,ws = load_vrec_catalog_boss(paths.boss_vrec_data + 'catalog.txt')
        ws = -ws[decs<25] / 299792. # (-v/c)
        ras = ras[decs<25]
        zs = zs[decs<25]
        decs = decs[decs<25]

        ras = ras[:nmax]
        decs = decs[:nmax]
        zs = zs[:nmax]
        ws = ws[:nmax]

        data['lams'] = ws*0
        
        
    else:
        raise NotImplementedError
        
    return ras,decs,zs,ws,data

def load_beam(freq):
    if freq=='f150': fname = paths.data+'s16_pa2_f150_nohwp_night_beam_tform_jitter.txt'
    elif freq=='f090': fname = paths.data+'s16_pa3_f090_nohwp_night_beam_tform_jitter.txt'
    ls,bls = np.loadtxt(fname,usecols=[0,1],unpack=True)
    assert ls[0]==0
    bls = bls / bls[0]
    return maps.interp(ls,bls)



def load_dumped_stats(mvstr,get_extra=False):
    savedir = paths.scratch + f"/{mvstr}/"
    assert os.path.exists(savedir), f"The path corresponding to {savedir} does not exist. If this is a meanfield, are you sure the parameters for your current run match the parameters in any existing meanfield directories?" 
    s = stats.load_stats(f'{savedir}')
    shape,wcs = enmap.read_map_geometry(f'{savedir}/map_geometry.fits')
    if get_extra:
        kmask = enmap.read_map(f'{savedir}/kmask.fits')
        modrmap = enmap.read_map(f'{savedir}/modrmap.fits')
        bin_edges = np.loadtxt(f'{savedir}/bin_edges.txt')
        assert wcsutils.equal(kmask.wcs,modrmap.wcs)
        assert wcsutils.equal(kmask.wcs,wcs)
        try:
            with open(f'{savedir}/cat_data_columns.txt', 'r') as file:
                columns = file.read().replace('\n', '').split(' ')
            data = {}
            dat = np.load(f"{savedir}/mstats_dump_vectors_data.npy")
            assert len(columns)==dat.shape[1]
            for i,col in enumerate(columns):
                data[col] = dat[:,i]
        except:
            data = None
        try:
            profs = np.loadtxt(f"{savedir}/profiles.txt")
        except:
            profs = None
        return s, shape, wcs, kmask, modrmap, bin_edges,data,profs
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
    
    


def plot(fname,stamp,tap_per,pad_per,crop=None,lim=None,cmap='coolwarm',quiver=None,label='$\\kappa$ (dimensionless)'):
    kmap = stamp
    trimy = int((tap_per+pad_per)/100. * kmap.shape[0])
    trimx = int((tap_per+pad_per)/100. * kmap.shape[1])
    if trimy>0 and trimx>0:
        tmap = kmap[trimy:-trimy,trimx:-trimx]
    else:
        tmap = kmap
    if crop is not None:
        tmap = maps.crop_center(tmap,crop)
    zfact = tmap.shape[0]*1./kmap.shape[0]
    twidth = tmap.extent()[0]/putils.arcmin
    io.plot_img(tmap,fname, flip=False, ftsize=12, ticksize=10,arc_width=twidth,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)',cmap=cmap,lim=lim,quiver=quiver,label=label)


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
        self.planck_beam = maps.gauss_beam(self.modlmap,defaults.planck_smica_beam_fwhm)
        wy, wx = enmap.calc_window(self.shape)
        act_pixwin   = wy[:,None] * wx[None,:]
        self.act_150_beam = load_beam('f150')(self.modlmap) * act_pixwin
        self.act_90_beam = load_beam('f090')(self.modlmap) * act_pixwin
        if self.is_meanfield:
            ucltt = theory.uCl('TT',self.modlmap)
            self.mgen = maps.MapGen((1,)+self.shape,self.wcs,ucltt[None,None])
        else:
            self.savedir = paths.scratch + f"/{lensed_version}/"
            
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



def load_vrec_catalog_boss(pathOutCatalog):
    """
    Code from Emmanuel Schaan to load in a BOSS v_rec catalog
    """
    data = np.genfromtxt(pathOutCatalog)
    nObj = len(data[:,0])
    #
    # sky coordinates and redshift
    RA = data[:,0] # [deg]
    DEC = data[:,1]   # [deg]
    Z = data[:,2]
    #
    # observed cartesian coordinates
    # coordX = data[:,3]   # [Mpc/h]
    # coordY = data[:,4]   # [Mpc/h]
    # coordZ = data[:,5]   # [Mpc/h]
    # #
    # # displacement from difference,
    # # not including the Kaiser displacement,
    # # from differences of the observed and reconstructed fields
    # dX = data[:,6]   # [Mpc/h]
    # dY = data[:,7]   # [Mpc/h]
    # dZ = data[:,8]   # [Mpc/h]
    # #
    # # Kaiser-only displacement
    # # originally from differences of the observed and reconstructed fields
    # dXKaiser = data[:,9]   # [Mpc/h] from cartesian catalog difference
    # dYKaiser = data[:,10]   # [Mpc/h]
    # dZKaiser = data[:,11]   # [Mpc/h]
    # #
    # # velocity in cartesian coordinates
    # vX = data[:,12]   #[km/s]
    # vY = data[:,13]   #[km/s]
    # vZ = data[:,14]   #[km/s]
    #
    # velocity in spherical coordinates,
    # from catalog of spherical displacements
    vR = data[:,15]  # [km/s]   from spherical catalog, >0 away from us
    # vTheta = data[:,16]   # [km/s]
    # vPhi = data[:,17]  # [km/s]
    # #
    # # Stellar masses
    # Mstellar = data[:,18]   # [M_sun], from Maraston et al
    # #
    # # Halo mass
    # hasM = data[:,19]
    # Mvir = data[:,20]  # [M_sun]
    # #
    # # Integrated optical depth [dimless]: int d^2theta n_e^2d sigma_T = (total nb of electrons) * sigma_T / (a chi)^2
    # integratedTau = data[:,21]   # [dimless]
    # #
    # # Integrated kSZ signal [muK * sr]: int d^2theta n_e sigma_T (-v/c) Tcmb
    # integratedKSZ = data[:, 22] # [muK * sr]
    # #
    # # Integrated Y signal [sr]: int d^2theta n_e sigma_T (kB Te / me c^2)
    # # needs to be multiplied by Tcmb * f(nu) to get muK
    # integratedY = data[:, 23] # [sr]
    return RA,DEC,Z,vR


def postprocess(stack_path,mf_path,save_name=None,ignore_param=False,args=None,ignore_last=None):

    if mf_path is not "":
        smf_path = mf_path if (ignore_last is None) else mf_path[:-ignore_last]
        mf_paramstr = re.search(rf'plmin_(.*?)_meanfield', smf_path).group(1)
    sstack_path = stack_path if (ignore_last is None) else stack_path[:-ignore_last]
    st_paramstr = re.search(rf'plmin_(.*)', sstack_path).group(1)

    if not(ignore_param):
        if mf_path is not "":
            try:
                assert mf_paramstr==st_paramstr
            except:
                print(mf_paramstr)
                print(st_paramstr)
                print("ERROR: The parameters for the stack and mean-field do not match.")
                raise

    tap_per = float(re.search(rf'tapper_(.*?)_padper', stack_path).group(1))
    pad_per = float(re.search(rf'padper_(.*?)_', stack_path).group(1))
    stamp_width_arcmin = float(re.search(rf'swidth_(.*?)_tapper', stack_path).group(1))
    klmin = int(re.search(rf'klmin_(.*?)_klmax', stack_path).group(1))
    klmax = int(re.search(rf'klmax_(.*?)_lxcut', stack_path).group(1))


    s_stack, shape_stack, wcs_stack, kmask, modrmap, bin_edges,data,profs = load_dumped_stats(stack_path,get_extra=True)
    if not(save_name is None):
        save_dir = f'{paths.postprocess_path}/{save_name}'
        io.mkdir(f'{save_dir}')
        if data is not None: 
            io.save_cols(f'{save_dir}/{save_name}_catalog_data.txt',[data[key] for key in sorted(data.keys())],header=' '.join([key for key in sorted(data.keys())]))

    if mf_path is not "":
        s_mf, shape_mf, wcs_mf = load_dumped_stats(mf_path)

    if mf_path is not "":
        assert np.all(shape_stack==shape_mf)
        assert wcsutils.equal(wcs_stack,wcs_mf)
    assert np.all(shape_stack==kmask.shape)

    shape = shape_stack
    wcs = wcs_stack
    cents = (bin_edges[:-1]+bin_edges[1:])/2.
    if not(save_name is None):
        crop = int(args.cwidth / defaults.pix_width_arcmin)

    unweighted_stack,nmean_weighted_kappa_stack,opt_weighted_kappa_stack,opt_binned,opt_covm,opt_corr,opt_errs,binned,covm,corr,errs = analyze(s_stack,wcs)
    if mf_path is not "":
        mf_unweighted_stack,mf_nmean_weighted_kappa_stack,mf_opt_weighted_kappa_stack,mf_opt_binned,mf_opt_covm,mf_opt_corr,mf_opt_errs,mf_binned,mf_covm,mf_corr,mf_errs = analyze(s_mf,wcs)

    # if profs is not None:
    #     profs = profs - mf_binned
    #     arcmax = 8.
    #     profs = profs[:,cents<arcmax].sum(axis=1)
    #     mean = profs.mean()
    #     err = profs.std() / np.sqrt(profs.size)
    #     lams = data['lams']
    #     pl = io.Plotter(xlabel='$\\lambda$',ylabel='$\\kappa(\\theta<8)$')
    #     pl._ax.scatter(lams,profs,s=3)
    #     pl.hline(y=0)
    #     pl.done(f"{save_dir}/kscatter.png")
    #     print(lams.shape)
    #     from scipy.stats import linregress
    #     print(lams)
    #     assert np.all(np.isfinite(lams))
    #     assert np.all(np.isfinite(profs))
    #     print(linregress(lams.astype(np.float), profs))

    # sys.exit()

    if not(save_name is None):
        plot(f"{save_dir}/{save_name}_unweighted_nomfsub.png",unweighted_stack,tap_per,pad_per,crop=None,lim=args.plim)
        plot(f"{save_dir}/{save_name}_unweighted_nomfsub_zoom.png",unweighted_stack,tap_per,pad_per,crop=crop,lim=args.plim)

        if mf_path is not "":
            # Opt weighted
            stamp = opt_weighted_kappa_stack - mf_opt_weighted_kappa_stack
            plot(f"{save_dir}/{save_name}_opt_weighted_mfsub.png",stamp,tap_per,pad_per,crop=None,lim=args.plim)
            plot(f"{save_dir}/{save_name}_opt_weighted_mfsub_zoom.png",stamp,tap_per,pad_per,crop=crop,lim=args.plim)

            modlmap = opt_weighted_kappa_stack.modlmap()
            stamp = maps.filter_map(opt_weighted_kappa_stack - mf_opt_weighted_kappa_stack,maps.gauss_beam(modlmap,args.fwhm))
            plot(f"{save_dir}/{save_name}_sm_opt_weighted_mfsub.png",stamp,tap_per,pad_per,crop=None,lim=args.slim)
            plot(f"{save_dir}/{save_name}_sm_opt_weighted_mfsub_zoom.png",stamp,tap_per,pad_per,crop=crop,lim=args.slim)
            enmap.write_map(f"{save_dir}/{save_name}_sm_opt_weighted_mfsub.fits",stamp)

            filt = maps.gauss_beam(modlmap,args.fwhm)/modlmap**2.
            filt[modlmap<200] = 0
            stamp = maps.filter_map(opt_weighted_kappa_stack - mf_opt_weighted_kappa_stack,filt)
            gy,gx = enmap.grad(stamp)
            gy = maps.filter_map(gy,maps.mask_kspace(shape,wcs,lmin=200,lmax=1000))
            gx = maps.filter_map(gx,maps.mask_kspace(shape,wcs,lmin=200,lmax=1000))
            plot(f"{save_dir}/{save_name}_sm_opt_weighted_mfsub_phi.png",stamp,tap_per,pad_per,crop=None,lim=args.slim,cmap='coolwarm',quiver=[gy,gx])
            plot(f"{save_dir}/{save_name}_sm_opt_weighted_mfsub_phi_zoom.png",stamp,tap_per,pad_per,crop=crop,lim=args.slim,cmap='coolwarm',quiver=[gy,gx])


            # Nmean weighted
            stamp = nmean_weighted_kappa_stack - mf_nmean_weighted_kappa_stack
            plot(f"{save_dir}/{save_name}_nmean_weighted_mfsub.png",stamp,tap_per,pad_per,crop=None,lim=args.plim)
            plot(f"{save_dir}/{save_name}_nmean_weighted_mfsub_zoom.png",stamp,tap_per,pad_per,crop=crop,lim=args.plim)

            modlmap = nmean_weighted_kappa_stack.modlmap()
            stamp = maps.filter_map(nmean_weighted_kappa_stack - mf_nmean_weighted_kappa_stack,maps.gauss_beam(modlmap,args.fwhm))
            plot(f"{save_dir}/{save_name}_sm_nmean_weighted_mfsub.png",stamp,tap_per,pad_per,crop=None,lim=args.slim)
            plot(f"{save_dir}/{save_name}_sm_nmean_weighted_mfsub_zoom.png",stamp,tap_per,pad_per,crop=crop,lim=args.slim)

            # Unweighted
            stamp = unweighted_stack - mf_unweighted_stack
            plot(f"{save_dir}/{save_name}_unweighted_mfsub.png",stamp,tap_per,pad_per,crop=None,lim=args.plim)
            plot(f"{save_dir}/{save_name}_unweighted_mfsub_zoom.png",stamp,tap_per,pad_per,crop=crop,lim=args.plim)

            modlmap = unweighted_stack.modlmap()
            stamp = maps.filter_map(unweighted_stack - mf_unweighted_stack,maps.gauss_beam(modlmap,args.fwhm))
            plot(f"{save_dir}/{save_name}_sm_unweighted_mfsub.png",stamp,tap_per,pad_per,crop=None,lim=args.slim)
            plot(f"{save_dir}/{save_name}_sm_unweighted_mfsub_zoom.png",stamp,tap_per,pad_per,crop=crop,lim=args.slim)



        else:
            stamp = opt_weighted_kappa_stack 
            plot(f"{save_dir}/{save_name}_opt_weighted_nomfsub.png",stamp,tap_per,pad_per,crop=None,lim=args.plim)
            plot(f"{save_dir}/{save_name}_opt_weighted_nomfsub_zoom.png",stamp,tap_per,pad_per,crop=crop,lim=args.plim)

            modlmap = opt_weighted_kappa_stack.modlmap()
            stamp = maps.filter_map(opt_weighted_kappa_stack ,maps.gauss_beam(modlmap,args.fwhm))
            plot(f"{save_dir}/{save_name}_sm_opt_weighted_nomfsub.png",stamp,tap_per,pad_per,crop=None,lim=args.slim)
            plot(f"{save_dir}/{save_name}_sm_opt_weighted_nomfsub_zoom.png",stamp,tap_per,pad_per,crop=crop,lim=args.slim)


        io.plot_img(corr,f'{save_dir}/{save_name}_corr.png')

        pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
        if mf_path is not "":
            pl.add_err(cents, opt_binned - mf_opt_binned, yerr=opt_errs,ls="-",label="Filtered kappa, mean-field subtracted (optimal)")
            pl.add_err(cents+0.2, binned - mf_binned, yerr=errs,ls="-",label="Filtered kappa, mean-field subtracted")
            pl.add_err(cents, mf_opt_binned, yerr=mf_opt_errs,label="Mean-field (optimal)",ls="-",alpha=0.5)
            pl.add_err(cents+0.2, mf_binned, yerr=mf_opt_errs,label="Mean-field",ls="-",alpha=0.5)
        else:
            pl.add_err(cents, opt_binned, yerr=opt_errs,ls="-",label="Filtered kappa (optimal)")
            pl.add_err(cents+0.2, binned, yerr=errs,ls="-",label="Filtered kappa")

        pl.hline(y=0)
        #pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pl._ax.set_ylim(args.ymin,args.ymax)
        pl.done(f'{save_dir}/{save_name}_profile.png')

        pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
        if mf_path is not "":
            pl.add_err(cents, opt_binned - mf_opt_binned, yerr=opt_errs,ls="-",label="Filtered kappa, mean-field subtracted (optimal)")
            pl.add_err(cents, mf_opt_binned, yerr=mf_opt_errs,label="Mean-field (optimal)",ls="-",alpha=0.5)
        else:
            pl.add_err(cents, opt_binned, yerr=opt_errs,ls="-",label="Filtered kappa (optimal)")
        pl.hline(y=0)
        #pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pl._ax.set_ylim(args.ymin,args.ymax)
        pl.done(f'{save_dir}/{save_name}_profile_clean.png')


        arcmax = 5.
        nbins = bin_edges[bin_edges<arcmax].size - 1
        if mf_path is "":
            mf_opt_binned = opt_binned*0
            mf_binned = opt_binned*0
        diff = (opt_binned - mf_opt_binned)[:nbins]
        cinv = np.linalg.inv(opt_covm[:nbins,:nbins])
        chisquare = np.dot(np.dot(diff,cinv),diff)
        snr = np.sqrt(chisquare)
        print("Naive SNR wrt null (optimal) : ", snr)

    ret_data = opt_binned - mf_opt_binned
    ret_cov = opt_covm

    if not(save_name is None):
        io.save_cols(f'{save_dir}/{save_name}_profile.txt',(cents,ret_data))
        np.savetxt(f'{save_dir}/{save_name}_covmat.txt',ret_cov)
        np.savetxt(f'{save_dir}/{save_name}_bin_edges.txt',bin_edges)
        enmap.write_map(f'{save_dir}/{save_name}_kmask.fits',kmask)

        diff = (binned - mf_binned)[:nbins]
        cinv = np.linalg.inv(covm[:nbins,:nbins])
        chisquare = np.dot(np.dot(diff,cinv),diff)
        snr = np.sqrt(chisquare)
        print("Naive SNR wrt null : ", snr)

        z = args.z

        print("Mean redshift : ",z)

        conc = args.conc
        cc = None
        sigma_mis = args.sigma_mis
        mguess = args.mass_guess
        merr_guess = (1/args.snr_guess) * mguess
        masses = np.linspace(mguess-args.nsigma*merr_guess,mguess+args.nsigma*merr_guess,args.num_ms)
        masses = masses[masses>0]
        arcmax = args.arcmax
        nbins = bin_edges[bin_edges<arcmax].size - 1
        profile = (opt_binned - mf_opt_binned)[:nbins]
        cov = opt_covm[:nbins,:nbins]
        fbin_edges = bin_edges[:nbins+1]
        fcents = cents[:nbins]
        lnlikes,like_fit,fit_mass,mass_err,fprofiles,fit_profile = lensing.fit_nfw_profile(profile,cov,masses,z,conc,cc,shape,wcs,fbin_edges,lmax=0,lmin=0,
                                                                                           overdensity=args.overdensity,
                                                                                           critical=args.critical,at_cluster_z=args.at_z0,
                                                                                           mass_guess=mguess,sigma_guess=merr_guess,kmask=kmask,sigma_mis=sigma_mis)

        print("Fit mass : " , fit_mass/1e14,mass_err/1e14)
        snr  = fit_mass / mass_err
        print("Fit mass SNR : ", snr)

        pl = io.Plotter(xlabel='$M$',ylabel='$L$')
        likes = np.exp(lnlikes)
        pl.add(masses,likes/likes.max())
        pl.add(masses,like_fit/like_fit.max())
        pl.vline(x=0)
        pl.done(f'{save_dir}/{save_name}_likes.png')

        pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
        pl.add_err(fcents, profile, yerr=np.sqrt(np.diagonal(cov)),ls="-",color='k')
        for fp in fprofiles:
            pl.add(fcents, fp,alpha=0.2)
        pl.add(fcents, fit_profile,color='k',ls='--')
        pl.hline(y=0)
        pl.done(f'{save_dir}/{save_name}_fprofiles.png')


        if args.theory is not None:
            savedir = p['scratch'] + f"/{args.theory}/"
            lensed_version = args.theory
            bfact = float(re.search(rf'bfact_(.*?)_pfact', args.theory).group(1))
            stamp_width_arcmin = float(re.search(rf'swidth_(.*?)_pwidth', lensed_version).group(1))
            pix_width_arcmin = float(re.search(rf'pwidth_(.*?)_bfact', lensed_version).group(1))
            dNpix = int(stamp_width_arcmin * bfact / (pix_width_arcmin))
            ddNpix = int(stamp_width_arcmin / (pix_width_arcmin))
            kappa = enmap.read_map(f'{savedir}kappa.fits')
            dkappa = kappa.resample((dNpix,dNpix))
            tkappa = maps.crop_center(dkappa,ddNpix)
            fkappa = maps.filter_map(tkappa,kmask)
            binner = stats.bin2D(modrmap*180*60/np.pi, bin_edges)
            tcents,t1d = binner.bin(fkappa)
            assert np.all(np.isclose(cents,tcents))

            diff = opt_binned - mf_opt_binned

            pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
            pl.add_err(cents,diff,yerr=opt_errs,ls='-')
            pl.add(tcents,t1d,ls='--')
            pl.done(f'{save_dir}/{save_name}_theory_comp.png')

            pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
            pl.add_err(cents,diff/t1d,yerr=opt_errs/t1d,ls='-')
            pl._ax.set_ylim(0.6,1.4)
            pl.hline(y=1)
            pl.done(f'{save_dir}/{save_name}_theory_comp_ratio.png')

    return cents,ret_data,ret_cov
