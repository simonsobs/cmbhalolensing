from pixell import enmap,utils as u,lensing as plensing,bench,bunch,wcsutils
from orphics import io, lensing as olensing, maps, cosmology,stats, mpi
import numpy as np
import symlens
import sys, shutil, os, warnings

class Analysis(object):
    def __init__(self,
                 choices_yaml,
                 atype, comm=None,
                 debug=False, outname=''):

        self.s = stats.Stats(comm)
        self.rank = comm.Get_rank()

        cdict = io.config_from_yaml(choices_yaml)

        # Start bookkeeping
        outroot = f'{outname}cmbh_out'
        with mpi.mpi_abort_on_exception(comm):
            if self.rank==0:
                if os.path.isdir(outroot):
                    raise FileExistsError("Output directory already exists. Choose a different outname.")
        comm.Barrier()
        self._out = lambda x: f'{outroot}/{x}'
        io.mkdir(outroot,comm=comm)
        if self.rank==0:
            shutil.copy(choices_yaml,self._out('choices.yaml'))
            hashtxt = io.hash_dict(cdict)
            with open(self._out('hash.txt'),'w') as f:
                f.write(hashtxt)
        
        self.c = bunch.Bunch(cdict)
        c = self.c
        self.Lmax = c.Lmax
        self.res = c.px_res_arcmin * u.arcmin
        self.proj = c.proj
        self.atype = atype
        self.debug = debug
        

        # Load template for cross_correlations
        thetas_cross,kappa_1h_cross,kappa_2h_cross = np.loadtxt(c.template_file,unpack=True)
        kappa_cross = kappa_1h_cross + kappa_2h_cross
        self.thetas_cross = thetas_cross
        self.kappa_cross = kappa_cross

        if atype=='flatsky_sim':
            # If we are doing a padded high-res flat sim
            # (e.g. for multiplicative bias correction)
            # we will initialize a FixedLens simulator
            # and get the shape and wcs from it, which
            # will be slightly different from that constructed
            # directly

            # Make template profile for lensing
            mass = c.mass
            z=c.z
            delta = c.delta

            thetas,kappa_1h,kappa_2h,_,_,_,_,_ = olensing.kappa_nfw_profiley(mass=mass,conc=None,
                                                                            z=z,z_s=1100.,background='critical',delta=delta, R_off_Mpc = None,
                                                                            apply_filter=False)
            
            kappa = kappa_1h + kappa_2h
            self.thetas = thetas
            self.kappa = kappa

            self.csim = olensing.FixedLens(self.thetas, self.kappa, width_deg=c.width_deg, pad_fact=1 if c.periodic else c.pad_fact)
            _,_,dummy = self.csim.generate_sim(0) # FIXME: is this necessary
            self.shape, self.wcs = dummy.shape, dummy.wcs

        else:
            if c.periodic: raise ValueError
            self.thumbnail_r = c.width_deg * u.deg / 2.
            self.shape, self.wcs = enmap.thumbnail_geometry(r=self.thumbnail_r, res=self.res, proj=self.proj)

        self.modrmap = enmap.modrmap(self.shape,self.wcs)
        self.rbin_edges = np.arange(0.,c.theta_max_arc*u.arcmin,c.rbin_width_arc*u.arcmin)
        self.rbinner = stats.bin2D(self.modrmap,self.rbin_edges)
        self.rcents = self.rbinner.cents
        self.modlmap = enmap.modlmap(self.shape,self.wcs)
        self.template = enmap.enmap(maps.interp(self.thetas_cross,self.kappa_cross)(self.modrmap),self.wcs)


        self.reconstructor = Recon(self.shape,self.wcs,
                                   xlmin=c.xlmin,xlmax=c.xlmax,
                                   ylmin=c.ylmin,ylmax=c.ylmax,
                                   xlxcut=c.xlxcut,xlycut=c.xlycut,
                                   ylxcut=c.ylxcut,ylycut=c.ylycut,
                                   klmin=c.klmin,klmax=c.klmax,
                                   apodize=True if not(c.periodic) else False,
                                   xbeam_fwhm_arcmin = c.xbeam_fwhm_arcmin,
                                   ybeam_fwhm_arcmin = c.ybeam_fwhm_arcmin,
                                   xbeam_noise_uk_arcmin = c.xbeam_noise_uk_arcmin,
                                   ybeam_noise_uk_arcmin = c.ybeam_noise_uk_arcmin,
                                   est = c.estimator,
                                   taper_percent=c.taper_percent,
                                   pad_percent=c.pad_percent,)

        self.minell = self.reconstructor.minell
        self.lbin_edges = np.arange(self.minell,c.klmax,self.minell*3)
        self.lbinner = stats.bin2D(self.modlmap,self.lbin_edges)
        self.lcents = self.lbinner.cents

        # Taper for templates and reconstruction
        self.ctaper = maps.radial_mask(self.shape,self.wcs,
                                  roll_start=c.theta_max_arc*u.arcmin,
                                  roll_width=c.roll_arc*u.arcmin,
                                  window="cosine")
        self.Lmin = c.Lmin
        self.Lmax = c.Lmax

        # Auto-correlation for template
        self.ktemplate = enmap.fft(self.template * self.ctaper,normalize='phys')
        cents,p1d,p2d = self.power(self.ktemplate,self.ktemplate)
        self.template_auto_p1d = p1d

        if debug and self.rank==0:
            io.plot_img(self.ctaper,self._out('ctaper.png'),arc_width=c.width_deg*60.,
                        xlabel='$\\Theta$ (arcmin)',ylabel='$\\Theta$ (arcmin)')
            io.plot_img((np.fft.fftshift(self.modlmap**2 * p2d)),self._out('template_auto_p2d.png'))
            pl = io.Plotter(xyscale='linlin',xlabel='L',ylabel='L^2 C_L')
            pl.add(cents,cents**2 * p1d,marker='o')
            pl.vline(x=500)
            pl.hline(y=0)
            pl.done(self._out('template_auto_p1d.png'))

        # Cross-correlation of template and lensing template
        self.lensing_template = enmap.enmap(maps.interp(self.thetas,self.kappa)(self.modrmap),self.wcs)
        self.klensing = enmap.fft(self.lensing_template * self.ctaper,normalize='phys')
        cents, p1d_cross, p2d_cross = self.power(self.ktemplate, self.klensing)
        self.template_cross_p1d = p1d_cross

        self.fit_Kmask = maps.mask_kspace(self.shape, self.wcs, lmin=c.Lmin, lmax=c.Lmax)

        if self.rank==0:
            enmap.write_map(self._out('ctaper.fits'),self.ctaper)
            enmap.write_map(self._out('ktemplate_real.fits'),self.ktemplate.real)
            enmap.write_map(self._out('ktemplate_imag.fits'),self.ktemplate.imag)
            enmap.write_map_geometry(self._out('geometry.fits'),self.shape,self.wcs)
            io.save_cols(self._out('rbin_edges.txt'),(self.rbin_edges))
            io.save_cols(self._out('lbin_edges.txt'),(self.lbin_edges))
        comm.Barrier()
        
        

    def get_recon(self,imapx,imapy=None,do_real=False,debug=False):
        out = {}
        out['fourier'] = {}
        
        # Get reconstruction
        krecon = self.reconstructor.recon(imapx, imapy, p2d_plot = self._out('recon_auto_p2d.png') if (debug and self.rank==0) else None)

        if do_real:
            out['real'] = {}
            # Real space stack of reconstruction
            okmap = krecon.copy()
            okmap[~self.fit_Kmask] = 0
            okreal = enmap.ifft(okmap,normalize='phys').real
            _,r1d = self.rbinner.bin(okreal)
            out['real']['map'] = okreal.copy()
            out['real']['profile'] = r1d.copy()
            self.s.add_to_stats('r1d',out['real']['profile'])
            self.s.add_to_stack('stack',out['real']['map'])


        # Mask with circular taper
        kreal = enmap.ifft(krecon,normalize='phys').real
        kmap = enmap.fft(kreal * self.ctaper,normalize='phys')

        # Cross-correlate with template
        cents,cp1d,cp2d = self.power(self.ktemplate,kmap)
        if not(np.all(np.isfinite(cp1d))): raise ValueError
        if debug and self.rank==0:
            io.plot_img((np.fft.fftshift(self.modlmap**2 * cp2d)),self._out('recon_template_cross_p2d.png'))

        out['fourier']['p2d'] = cp2d.copy()
        out['fourier']['profile'] = cp1d.copy()

        self.s.add_to_stats('cp1d',out['fourier']['profile'])
        
        return out

    def get_stamp(self,imap=None,ra_deg=None,dec_deg=None,
                  ivar=False,
                  seed=None):

        if self.atype!='flatsky_sim':
            coords = np.asarray((dec_deg,ra_deg)) * u.degree
            if seed is not None: raise ValueError
            r = self.thumbnail_r
            res = self.res
            proj = self.proj
        
        if self.atype=='data':
            if ivar:
                out = reproject.thumbnails_ivar(
                    imap,
                    coords,
                    r=r,
                    res=res,
                    extensive=True,
                    proj=proj
                )
            else:
                out = reproject.thumbnails(
                    imap,
                    coords,
                    r=r,
                    res=res,
                    proj=proj,
                    oversample=2,
                    pixwin=True
                )       


        elif self.atype=='healpix_sim':
            out = maps.thumbnail_healpix(imap,coords,r=r,res=res,proj=proj)
            # TODO: healpix pixel window needs to be accounted for in beam
        elif self.atype=='car_sim':
            out = reproject.thumbnails(
                imap,
                coords,
                r=r,
                res=res,
                proj=proj,
                oversample=2,
                pixwin=False # only this is different
            )       

        elif self.atype=='flatsky_sim':
            _, _, out = self.csim.generate_sim(seed)

        if not(wcsutils.equal(out.wcs,self.wcs)): raise ValueError
        if out.shape[0]!=self.shape[0]: raise ValueError
        if out.shape[1]!=self.shape[1]: raise ValueError
        return out
        
        
    
    def power(self,kmap1,kmap2):
        p2d = (kmap1*kmap2.conj()).real
        cents, p1d = self.lbinner.bin(p2d)
        return cents, p1d, p2d


    def finish(self, save_transfer=False):
        self.s.get_stats()
        self.s.get_stacks()

        


        if self.rank==0:

            
            rcents = self.rcents
            lcents = self.lcents
            Lmin = self.Lmin
            Lmax = self.Lmax
            if self.atype=='flatsky_sim':
                p1d = self.template_cross_p1d
            else: p1d = self.template_auto_p1d
            cmean = self.s.stats['cp1d']['mean']
            cerr = self.s.stats['cp1d']['errmean']
            ccov = self.s.stats['cp1d']['covmean']

            io.save_cols(self._out('fourier_profile_template_auto.txt'),(lcents,p1d))
            io.save_cols(self._out('fourier_profile_cross_mean.txt'),(lcents,cmean))
            np.savetxt(self._out('fourier_cov.txt'),ccov)

            rsel = rcents<self.c.theta_max_real_arc*u.arcmin
            rmean = self.s.stats['r1d']['mean'][rsel]
            rcov = self.s.stats['r1d']['covmean'][rsel,:][:,rsel]
            rerr = np.sqrt(np.diagonal(rcov))
            
            io.save_cols(self._out('real_profile'),(rcents,self.s.stats['r1d']['mean']))
            np.savetxt(self._out('real_cov.txt'),self.s.stats['r1d']['covmean'])

            stack = self.s.stacks['stack']
            io.plot_img(stack,self._out('stack.png'),arc_width=self.c.width_deg*60.,
                        xlabel='$\\Theta$ (arcmin)',ylabel='$\\Theta$ (arcmin)')

            sel = np.logical_and(lcents>Lmin,lcents<Lmax)
            signal = p1d[sel]
            cov = ccov[sel,:][:,sel]
            err = cerr[sel]



            io.plot_img(stats.cov2corr(ccov),self._out('ccov.png'))
            cinv = np.linalg.inv(cov)
            nobj = sum(self.s.numobj['cp1d'] )
            snr1 = np.sqrt(np.dot(np.dot(signal,cinv),signal)  * (1000/nobj))
            snr2 = np.sqrt(np.sum(signal**2./err**2.)  * (1000/nobj))
            snr3 = np.sqrt(np.dot(np.dot(rmean,np.linalg.inv(rcov)),rmean)  * (1000/nobj))

            outstr =  f"Rough SNR estimates (scaled {nobj} objects to 1000 objects): " + \
                f"\n  Method 1: Fourier w/ covmat :     {snr1:.2f}" + \
                f"\n  Method 2: Fourier diagonal:       {snr2:.2f}" + \
                f"\n  Method 3: Real w/covmat:       {snr3:.2f}" + \
                f"\n The second is quickest to converge. All three should agree in the limit of infinite sims."
            print(outstr)

            if save_transfer:
                if self.atype!='flatsky_sim':
                    warnings.warn("Why are you saving the transfer function if this is not a flatsky sim?")
                
                transfer_fn = cmean / p1d
                transfer_fn_err = cerr / p1d
                io.save_cols(self._out('transfer_fn.txt'),(lcents,transfer_fn,transfer_fn_err))
            
                pl = io.Plotter('rCl',ylabel='$C_L^{\\rm sim} / C_L^{\\rm template}$')
                pl.add_err(lcents,transfer_fn,yerr=transfer_fn_err,marker='o')
                pl.vline(x=500)
                pl.hline(y=1)
                pl.done(self._out('transfer_fn.png'))

            
            pl = io.Plotter(xyscale='linlin',xlabel='L',ylabel='L^2 C_L')
            pl.add(lcents,lcents**2 * p1d,marker='o',label='full template')
            pl.add_err(lcents,lcents**2 * cmean,yerr=cerr * lcents**2.,marker='o')
            pl.add(lcents[sel],lcents[sel]**2 * p1d[sel],marker='d',color='r',label='selected template')
            pl.vline(x=500)
            pl.hline(y=0)
            pl._ax.set_ylim(-3e-5,5.1e-5)
            pl.done(self._out('recon_template_cross_p1d.png'))

            pl = io.Plotter()
            pl.add_err(rcents[rsel]/u.arcmin,rmean,yerr=rerr,marker='o')
            pl.add(self.thetas/u.arcmin, self.kappa) #FIXME: only for flatsky sim?
            pl._ax.set_xlim(0,10)
            pl._ax.set_ylim(-0.01,rmean.max()*1.2)
            pl.hline(y=0)
            pl.done(self._out('recon_profile.png'))
    
    
class Recon(object):
    def __init__(self,shape,wcs,
                 xlmin=None,xlmax=2000,
                 ylmin=None,ylmax=3500,
                 xlxcut=None,xlycut=None,
                 ylxcut=None,ylycut=None,
                 klmin=None,klmax=6000,
                 apodize=True,
                 xbeam_fwhm_arcmin = 7.0,
                 ybeam_fwhm_arcmin = 1.5,
                 xbeam_noise_uk_arcmin = 30.0,
                 ybeam_noise_uk_arcmin = 10.0,
                 est = 'hdv',
                 taper_percent=12.0,
                 pad_percent=3.0,
                 ):

        if apodize:
            self.taper,_ = maps.get_taper(shape,wcs,taper_percent=12.0,pad_percent=3.0)
        else:
            self.taper = 1
        modlmap = enmap.modlmap(shape,wcs)
        self.modlmap = modlmap
        self.shape = shape
        self.wcs = wcs
        theory = cosmology.default_theory()
        ltt2d = theory.lCl('TT',modlmap)
        theory = cosmology.default_theory()
        self.minell = maps.minimum_ell(shape,wcs)
        if klmin is None: klmin = self.minell
        if xlmin is None: xlmin = self.minell
        if ylmin is None: ylmin = self.minell
        self.xmask = maps.mask_kspace(shape, wcs, lmin=xlmin, lmax=xlmax, lxcut = xlxcut, lycut = xlycut)
        self.ymask = maps.mask_kspace(shape, wcs, lmin=ylmin, lmax=ylmax, lxcut = ylxcut, lycut = ylycut)
        self.kmask = maps.mask_kspace(shape, wcs, lmin=klmin, lmax=klmax)

        ibeamy = modlmap*0
        beamy = maps.gauss_beam(ybeam_fwhm_arcmin,modlmap)
        ibeamy[modlmap>0] = 1./beamy[modlmap>0]
        
        ibeamx = modlmap*0
        beamx = maps.gauss_beam(xbeam_fwhm_arcmin,modlmap)
        ibeamx[modlmap>0] = 1./beamx[modlmap>0]
        
        tclxy2d = ltt2d
        tclyy2d = ltt2d + (ybeam_noise_uk_arcmin*np.pi/180./60. * ibeamy)**2
        tclxx2d = ltt2d + (xbeam_noise_uk_arcmin*np.pi/180./60. * ibeamx)**2

        if est=='hdv':
            self.feed_dict = {
                "uC_T_T": ltt2d,
                "tC_A_T_A_T": tclyy2d,
                "tC_P_T_P_T": tclxx2d,
                "tC_A_T_P_T": tclxy2d,
                "tC_P_T_A_T": tclxy2d,
            }

            self.cqe = symlens.QE(
                self.shape,
                self.wcs,
                self.feed_dict,
                estimator="hdv",
                XY="TT",
                xmask=self.xmask,
                ymask=self.ymask,
                field_names=["P", "A"],
                groups=None,
                kmask=self.kmask,
            )
            
        elif est=='hu_ok':
            self.feed_dict = {
                "uC_T_T": ltt2d,
                "tC_T_T": tclyy2d,
            }

            self.cqe = symlens.QE(
                self.shape,
                self.wcs,
                self.feed_dict,
                estimator="hu_ok",
                XY="TT",
                xmask=self.ymask,
                ymask=self.ymask,
                kmask=self.kmask,
            )

        elif est=='hardened':
            self.feed_dict = {
                "uC_T_T": ltt2d,
                "tC_T_T": tclyy2d,
                "pc_T_T": tclyy2d*0 + 1,
            }

            self.cqe = symlens.HardenedTT(
                self.shape,
                self.wcs,
                self.feed_dict,
                estimator="hu_ok",
                xmask=self.xmask,
                ymask=self.ymask,
                kmask=self.kmask,
            )
        
        
    def recon(self,imapx,imapy=None,p2d_plot=None):

        kmapx = enmap.fft(imapx * self.taper,normalize='phys')
        if p2d_plot is not None:
            p2d = (kmapx*kmapx.conj()).real
            io.plot_img(np.log(maps.get_central(np.fft.fftshift(p2d),0.3)),p2d_plot)
        if imapy is None:
            kmapy = kmapx.copy()
        else:
            kmapy = enmap.fft(imapy*self.taper,normalize='phys')
        
        self.feed_dict["X"] = kmapx
        self.feed_dict["Y"] = kmapy

        # Sanity check
        for key in self.feed_dict.keys():
            assert np.all(np.isfinite(self.feed_dict[key]))

        
        # Fourier space lens reconstruction
        krecon = self.cqe.reconstruct(self.feed_dict, xname="X_l1", yname="Y_l2", physical_units=True)
        return krecon
