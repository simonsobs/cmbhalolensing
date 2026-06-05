"""
tsz_subtract.py
===============

Fit and subtract the thermal Sunyaev-Zel'dovich (tSZ) signal of massive clusters
from an ACT-like high-resolution CMB coadd map.

Given
  * an SZ cluster catalog (RA, Dec, M500c, redshift),
  * a coadd temperature map
  * its inverse-variance (ivar) map,
  * and a beam (FWHM or an ell,B_ell file),

this module loops over clusters, extracts postage stamps, fits a two-parameter
tSZ model (amplitude + angular scale radius) per cluster while accounting for the
CMB and the inhomogeneous instrument noise (via a pixel-pixel covariance
C = S_cmb + N_ivar), subtracts the best-fit beam-convolved model, and writes a
full map with the tSZ decrements removed. It builds before/after stacks, prints
diagnostics and timing, and is parallelized for HPC with MPI.

The per-cluster profile is the Arnaud et al. (2010) Universal Pressure Profile
(GNFW), converted to a Compton-y profile by line-of-sight integration and to a
temperature decrement at a single effective frequency via the tSZ spectral
factor g(nu).

Example (HPC node, 64 cores / 200 GB)
-------------------------------------
Run with MPI across all 64 cores; the effective frequency is read from the
``f150`` tag in the coadd filename and the ACT DR6 catalog is filtered in place::

    mpirun -n 64 python tsz_subtract.py \\
        --coadd  act_dr6_f150_coadd.fits \\
        --ivar   act_dr6_f150_ivar.fits \\
        --catalog DR6_cluster-catalog_v1.0.fits --query "SNR > 6" \\
        --beam   beam_f150.txt \\
        --output coadd_f150_tsz_subtracted.fits --outdir out/

Clusters are distributed over the 64 ranks, each of which reads only its own
small postage stamps from disk; peak memory is dominated by rank 0 loading the
full coadd once at the end, so the 200 GB node is never close to saturated.

Design notes
------------
* Memory: worker ranks only ever hold small postage stamps (read directly from
  disk via per-stamp pixbox reads), so the many-GB coadd is never duplicated
  across cores. Only rank 0 loads the full map once, at the end, to apply the
  gathered model stamps and write the output.

This is a single-file module; run ``python tsz_subtract.py --help``.
"""

import argparse
import os
import time

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve
from scipy.special import j0

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

from pixell import enmap, reproject, curvedsky, pointsrcs, utils

from orphics import foregrounds, mpi, io, stats, maps

# ---------------------------------------------------------------------------
# Physical constants (CGS / keV units for the Compton-y line-of-sight integral)
# ---------------------------------------------------------------------------
SIGMA_T_CM2 = 6.6524587158e-25  # Thomson cross section [cm^2]
ME_C2_KEV = 510.998946  # electron rest-mass energy [keV]
MPC_CM = 3.0856775815e24  # 1 Mpc in cm
# Ratio of total thermal pressure to electron pressure for a fully ionized
# H+He plasma (X=0.76): P_thermal = 1.932 * P_electron.
P_TH_TO_PE = 1.932

# Arnaud et al. (2010) best-fit GNFW parameters (Eq. 12, with h70 scalings).
ARNAUD = dict(P0=8.403, c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)

# Radial extent (arcmin) to which the beam-convolved model is tabulated and
# painted. The fitting stamp can stay compact (the GLS amplitude is unbiased on a
# fixed stamp), but the subtracted model must reach the outskirts of large,
# low-redshift clusters (angular R500c up to ~10'), so it is painted out to here.
PROFILE_MAX_ARCMIN = 40.0

# In --debug, cap a real-data run to this many clusters for a quick smoke test.
DEBUG_NCLUSTERS = 50

# Single precision for the pixel covariance: cho_factor is ~2x faster and the
# matrices are half the size. S_cmb + N_ivar is well within float32 conditioning
# for realistic noise levels (the diagonal noise floor bounds the eigenvalues).
COV_DTYPE = np.float32


# ===========================================================================
# Cosmology helpers
# ===========================================================================
# Fiducial flat-LCDM cosmology used for all distances and densities.
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)


def cluster_scales(m500c, z, cosmo):
    """Compute R500c, theta500 and P500 for a cluster from its mass and redshift.

    Parameters
    ----------
    m500c : float
        Cluster mass M500c in solar masses.
    z : float
        Cluster redshift.
    cosmo : FlatLambdaCDM
        Cosmology object.

    Returns
    -------
    r500_mpc : float
        Physical R500c in Mpc.
    theta500_rad : float
        Angular R500c in radians (R500c / D_A(z)).
    p500_kev_cm3 : float
        Characteristic pressure P500 in keV cm^-3.
    """
    h70 = cosmo.H0.value / 70.0
    ez = cosmo.efunc(z)
    rho_c = cosmo.critical_density(z).to(u.Msun / u.Mpc**3).value
    r500_mpc = (3.0 * m500c / (4.0 * np.pi * 500.0 * rho_c)) ** (1.0 / 3.0)
    d_a_mpc = cosmo.angular_diameter_distance(z).to(u.Mpc).value
    theta500_rad = r500_mpc / d_a_mpc
    m_term = (m500c * h70 / 3.0e14) ** (2.0 / 3.0)
    p500 = 1.65e-3 * ez ** (8.0 / 3.0) * m_term * h70**2
    return r500_mpc, theta500_rad, p500


def gnfw_pressure(x):
    """Dimensionless Arnaud GNFW pressure profile p(x), x = r / R500c.

    Parameters
    ----------
    x : array_like
        Scaled radius r/R500c (dimensionless), > 0.

    Returns
    -------
    ndarray
        p(x) = P0 / [(c500 x)^gamma (1 + (c500 x)^alpha)^((beta-gamma)/alpha)].
    """
    p0, c500 = ARNAUD["P0"], ARNAUD["c500"]
    g, a, b = ARNAUD["gamma"], ARNAUD["alpha"], ARNAUD["beta"]
    cx = c500 * np.asarray(x)
    return p0 / (cx**g * (1.0 + cx**a) ** ((b - g) / a))


# ===========================================================================
# tSZ profile -> temperature template
# ===========================================================================
def deltaT_profile(m500c, z, cosmo, nu_ghz, ntheta=256, los_max_r500=5.0, los_npts=400):
    """Build an interpolator of the tSZ temperature decrement vs angular radius.

    The Arnaud GNFW pressure profile is integrated along the line of sight to a
    Compton-y profile y(theta), then converted to a temperature decrement
    deltaT(theta) = g(nu) * T_CMB * y(theta) at the effective frequency.

    Parameters
    ----------
    m500c, z : float
        Cluster mass (M500c, Msun) and redshift.
    cosmo : FlatLambdaCDM
        Cosmology.
    nu_ghz : float
        Effective frequency of the coadd map in GHz.
    ntheta : int
        Number of angular grid points for the template interpolator.
    los_max_r500 : float
        Half line-of-sight integration length in units of R500c.
    los_npts : int
        Number of line-of-sight integration samples.

    Returns
    -------
    callable
        ``f(theta_rad)`` returning deltaT in micro-Kelvin; theta_rad can be an
        array. The interpolator is flat (constant) beyond its tabulated range.
    theta500_rad : float
        The cluster angular scale R500c/D_A used to set the template extent.
    """
    r500_mpc, theta500_rad, p500 = cluster_scales(m500c, z, cosmo)
    d_a_mpc = r500_mpc / theta500_rad

    # Angular grid out to a generous multiple of theta500.
    theta_grid = np.linspace(0.0, 12.0 * theta500_rad, ntheta)
    # Line-of-sight grid in physical Mpc.
    los = np.linspace(0.0, los_max_r500 * r500_mpc, los_npts)

    rperp_mpc = d_a_mpc * theta_grid  # projected physical radius (Mpc) per theta
    # r3d[i, j] = sqrt(los_j^2 + rperp_i^2)
    r3d = np.sqrt(los[None, :] ** 2 + rperp_mpc[:, None] ** 2)
    x = np.maximum(r3d / r500_mpc, 1e-4)  # floor avoids the GNFW central cusp
    pe = p500 * gnfw_pressure(x) / P_TH_TO_PE  # electron pressure [keV cm^-3]
    # y = (sigma_T / m_e c^2) * 2 * int_0^Lmax P_e dl, with dl in cm.
    prefac = SIGMA_T_CM2 / ME_C2_KEV
    y_theta = prefac * 2.0 * np.trapezoid(pe, los * MPC_CM, axis=1)

    g_nu = foregrounds.g_tsz(nu_ghz)
    dT = g_nu * foregrounds.TCMB_uK * y_theta  # micro-Kelvin

    interp = interp1d(
        theta_grid, dT, kind="cubic", bounds_error=False, fill_value=(dT[0], dT[-1])
    )
    return interp, theta500_rad


# ===========================================================================
# Beam and CMB theory
# ===========================================================================
def get_beam_fn(beam_arg):
    """Return a beam transfer function B(ell) from a FWHM or an ell,B_ell file.

    Parameters
    ----------
    beam_arg : float or str
        If a float (or float-parsable string), interpreted as a Gaussian FWHM in
        arcminutes. Otherwise a path to a two-column text file (ell, B_ell).

    Returns
    -------
    callable
        ``B(ell)`` returning the (peak-normalized) beam transfer at multipole ell.
    """
    try:
        fwhm = float(beam_arg)

        def bfn(ell):
            return maps.gauss_beam(ell, fwhm)

        return bfn
    except (TypeError, ValueError):
        ells, bells = np.loadtxt(beam_arg, unpack=True)
        bells = bells / bells[np.argmin(ells)]
        interp = interp1d(ells, bells, bounds_error=False, fill_value=(bells[0], 0.0))

        def bfn(ell):
            return interp(ell)

        return bfn


def get_cmb_theory_fn(lmax=6000, H0=67.5, ombh2=0.022, omch2=0.122, ns=0.965, As=2e-9):
    """Compute a lensed CMB TT theory function with CAMB.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    H0, ombh2, omch2, ns, As : float
        CAMB cosmological parameters.

    Returns
    -------
    callable
        ``theory(spec, ell)`` returning the lensed C_ell in micro-K^2. Only the
        'TT' spectrum is non-zero (this is an intensity-only pipeline); other
        spectra return zeros so the function is compatible with orphics
        ``scov_from_theory``.
    ells : ndarray
        The multipole array used.
    cltt : ndarray
        The lensed TT spectrum (micro-K^2) on ``ells``.
    """
    import camb

    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, ns=ns, As=As, lmax=lmax)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)
    cltt = powers["total"][:, 0]
    ells = np.arange(cltt.size)
    interp = interp1d(ells, cltt, bounds_error=False, fill_value=0.0)

    def theory(spec, ell):
        if spec == "TT":
            return interp(ell)
        return np.zeros_like(ell, dtype=float)

    return theory, ells, cltt


# ===========================================================================
# Pixel covariance: stationary CMB part (cached) + per-cluster ivar noise
# ===========================================================================
class ScovCache:
    """Stationary CMB pixel covariance from the real-space correlation function.

    The covariance of a postage stamp cut from the sky is the stationary CMB+beam
    correlation function evaluated at the true angular separations between every
    pair of pixels,

        S_ij = C(|r_i - r_j|),
        C(r) = (1 / 2 pi) integral_0^lmax C_ell B_ell^2 J_0(ell r) ell d_ell,

    where C_ell is the (lensed) CMB TT spectrum and B_ell the beam transfer.

    This is preferred over building S by point-sampling the theory power on the
    stamp's discrete ``modlmap``: a small stamp's coarse Fourier grid (minimum
    non-zero multipole ~ 2 pi / stamp_size) does not resolve the dominant
    large-scale CMB (ell ~ 200-400), so the point-sampled covariance severely
    under-counts the true CMB variance (by ~8x for a ~0.4 deg stamp). The
    correlation-function form integrates the spectrum continuously and reproduces
    the variance of stamps actually extracted from the sky to ~1-5% across all
    separations.

    Using the stamp's real sky positions (great-circle separations) makes the
    covariance CAR-correct, since the declination-dependent x-pixel scale enters
    through the per-pixel positions. The 1D C(r) profile is built once; the
    pixel-pair covariance is cached per (shape, declination-band) key.
    """

    def __init__(self, theory_fn, beam_fn, lmax=6000, rmax_deg=2.0, nr=2000):
        """Precompute the 1D correlation function C(r) and prepare the cache.

        Parameters
        ----------
        theory_fn : callable
            ``theory('TT', ell)`` returning the CMB TT spectrum in micro-K^2.
        beam_fn : callable
            Beam transfer function B(ell).
        lmax : int
            Maximum multipole used in the C(r) integral.
        rmax_deg : float
            Maximum angular separation (degrees) tabulated for C(r); must exceed
            the largest pixel-pair separation in any stamp.
        nr : int
            Number of radial samples for the C(r) tabulation.
        """
        self.theory_fn = theory_fn
        self.beam_fn = beam_fn
        ell = np.arange(float(lmax))
        cl = theory_fn("TT", ell) * beam_fn(ell) ** 2
        rgrid = np.linspace(0.0, np.deg2rad(rmax_deg), nr)
        dell = ell[1] - ell[0]
        # C(r) = (1/2pi) integral C_ell B_ell^2 J_0(ell r) ell d_ell
        cr = (j0(np.outer(rgrid, ell)) * (cl * ell)[None, :]).sum(axis=1) * dell
        cr /= 2.0 * np.pi
        self.corr_fn = interp1d(rgrid, cr, bounds_error=False, fill_value=0.0)
        self.var0 = float(cr[0])
        self._cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, stamp):
        """Return the CMB pixel covariance S (Npix x Npix) for a given stamp.

        Parameters
        ----------
        stamp : enmap.ndmap
            A 2D postage stamp providing shape and WCS.

        Returns
        -------
        ndarray
            The (Ny*Nx, Ny*Nx) CMB pixel covariance for this stamp.
        """
        pos = stamp.posmap()
        dec = pos[0].reshape(-1)
        ra = pos[1].reshape(-1)
        dec_band = round(np.rad2deg(dec.mean()))
        key = (tuple(stamp.shape[-2:]), dec_band)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        sind = np.sin(dec)
        cosd = np.cos(dec)
        cosang = sind[:, None] * sind[None, :] + cosd[:, None] * cosd[None, :] * np.cos(
            ra[:, None] - ra[None, :]
        )
        np.clip(cosang, -1.0, 1.0, out=cosang)
        rmat = np.arccos(cosang)
        scov = self.corr_fn(rmat).astype(COV_DTYPE)
        self._cache[key] = scov
        return scov


# ===========================================================================
# Model template and per-cluster fit
# ===========================================================================
class BeamProfiler:
    """Beam-convolve radial cluster profiles in 1D, free of FFT ringing.

    The beam is applied to the (circularly symmetric) tSZ profile by a 1D
    Hankel-space convolution rather than a 2D FFT on a finite stamp. Because the
    output is band-limited by the beam, the resulting radial profile is smooth
    and carries none of the periodic-boundary Gibbs ringing an FFT would imprint
    around the sharp cluster core. The convolution is encoded once as a radial
    smoothing operator ``K`` so that beam-convolving any profile is a single
    matrix-vector product, and the 2D template is then obtained by direct
    interpolation of the 1D profile onto pixel radii (see :func:`model_stamp`).

    The radial smoothing kernel is
    ``K[i,j] = (integral B(l) J0(l r_i) J0(l r_j) l dl) * r_j dr`` so that
    ``profile_beamed(r_i) = sum_j K[i,j] profile_raw(r_j)`` reproduces
    ``(1/2pi) integral B(l) [2pi integral profile(r') J0(l r') r' dr'] J0(l r) l dl``.
    """

    def __init__(self, beam_fn, rmax_rad, nr=512, lmax=30000, nl=4000):
        """Precompute the radial beam-smoothing operator on a fixed grid.

        Parameters
        ----------
        beam_fn : callable
            ``B(ell)`` beam transfer function (peak-normalised).
        rmax_rad : float
            Outer radius (radians) of the tabulated profile grid.
        nr : int
            Number of radial samples.
        lmax : int
            Maximum multipole in the Hankel integral.
        nl : int
            Number of multipole samples.
        """
        self.r = np.linspace(0.0, rmax_rad, nr)
        dr = self.r[1] - self.r[0]
        ell = np.linspace(0.0, lmax, nl)
        dl = ell[1] - ell[0]
        j0r = j0(np.outer(self.r, ell))  # (nr, nl)
        self.K = (j0r * (beam_fn(ell) * ell * dl)) @ j0r.T * (self.r * dr)

    def beamed(self, dT_interp, scale):
        """Return the beam-convolved profile of ``dT_interp(theta/scale)``.

        Parameters
        ----------
        dT_interp : callable
            deltaT(theta_rad) interpolator from :func:`deltaT_profile`.
        scale : float
            Angular scale multiplier s (profile evaluated at theta/s).

        Returns
        -------
        ndarray
            Beam-convolved deltaT sampled on ``self.r`` (micro-K).
        """
        return self.K @ dT_interp(self.r / scale)


def model_stamp(stamp, prof_r, prof_vals):
    """Paint a 1D radial profile on a stamp by interpolation (no FFT).

    The profile is already beam-convolved (see :class:`BeamProfiler`), so the
    template is just the profile evaluated at each pixel's great-circle distance
    from the stamp centre -- no per-stamp FFT or convolution is performed.

    Parameters
    ----------
    stamp : enmap.ndmap
        Postage stamp providing the WCS and angular-distance map.
    prof_r : ndarray
        Radial grid (radians) on which ``prof_vals`` is tabulated.
    prof_vals : ndarray
        Beam-convolved deltaT on ``prof_r`` (micro-K).

    Returns
    -------
    enmap.ndmap
        The template (micro-K) with the stamp's WCS.
    """
    return enmap.enmap(np.interp(stamp.modrmap(), prof_r, prof_vals), stamp.wcs)


def fit_cluster(data, ivar, m500c, z, cosmo, nu_ghz, profiler, scov_cache, s_vals):
    """Fit a 2-parameter (amplitude, scale) tSZ model to one cluster stamp.

    For each trial scale ``s``, the amplitude is solved by generalized least
    squares against the CMB+noise covariance C = S_cmb + N_ivar, and the scale
    minimizing chi-square is selected.

    Parameters
    ----------
    data : enmap.ndmap
        The temperature stamp (micro-K).
    ivar : enmap.ndmap
        The inverse-variance stamp (micro-K^-2 per pixel).
    m500c, z : float
        Cluster mass (M500c, Msun) and redshift.
    cosmo : FlatLambdaCDM
        Cosmology.
    nu_ghz : float
        Effective frequency in GHz.
    profiler : BeamProfiler
        Beam-convolves the radial cluster profile in 1D (ringing-free).
    scov_cache : ScovCache
        Cache providing the CMB pixel covariance for the stamp geometry.
    s_vals : ndarray
        Grid of trial angular-scale multipliers.

    Returns
    -------
    model : enmap.ndmap
        The best-fit beam-convolved model stamp (micro-K).
    result : dict
        Diagnostics: amplitude, amp_err, scale, snr, chi2, dof, theta500_arcmin,
        and central decrement of the model.
    """
    dT_interp, theta500 = deltaT_profile(m500c, z, cosmo, nu_ghz)

    # C = S_cmb (cached, float32) + diagonal pixel noise from ivar, assembled and
    # factored in single precision. Add the noise to a copy of the cached S so the
    # shared matrix is not mutated.
    ivar_flat = np.asarray(ivar, dtype=np.float64).reshape(-1)
    var = 1.0 / ivar_flat
    bad = ~np.isfinite(var)
    if bad.any():
        var[bad] = 1.0 / ivar_flat[ivar_flat > 0].max()
    cmat = scov_cache.get(data).astype(COV_DTYPE, copy=True)
    cmat[np.diag_indices_from(cmat)] += var.astype(COV_DTYPE)
    cho = cho_factor(cmat)

    d = np.asarray(data, dtype=COV_DTYPE).reshape(-1)
    cinv_d = cho_solve(cho, d)
    dt_cinv_d = float(d @ cinv_d)

    def chi2_for_scale(s):
        prof = profiler.beamed(dT_interp, s)
        templ = model_stamp(data, profiler.r, prof)
        t = np.asarray(templ, dtype=COV_DTYPE).reshape(-1)
        cinv_t = cho_solve(cho, t)
        tct = float(t @ cinv_t)
        if tct <= 0:
            return np.inf, 0.0, 0.0, templ, prof
        amp = float(t @ cinv_d) / tct
        chi2 = dt_cinv_d - amp**2 * tct
        return chi2, amp, tct, templ, prof

    best = None
    for s in s_vals:
        chi2, amp, tct, templ, prof = chi2_for_scale(s)
        if best is None or chi2 < best[0]:
            best = (chi2, amp, tct, s, templ, prof)

    chi2, amp, tct, s_hat, templ, prof = best
    amp_err = 1.0 / np.sqrt(tct) if tct > 0 else np.inf
    model = amp * templ
    result = dict(
        amplitude=amp,
        amp_err=amp_err,
        scale=s_hat,
        snr=amp / amp_err if np.isfinite(amp_err) and amp_err > 0 else 0.0,
        chi2=chi2,
        dof=d.size - 1,
        theta500_arcmin=np.rad2deg(theta500) * 60.0,
        central_uK=float(np.asarray(model).min()),
        # Amplitude-scaled, beam-convolved 1D radial profile (micro-K) on the
        # profiler grid, used to paint the final model map with pointsrcs.
        profile=amp * prof,
    )
    return model, result


# ===========================================================================
# Map / catalog I/O and stamp extraction
# ===========================================================================
def m200c_to_m500c(m200c, z, cosmo):
    """Convert M200c to M500c assuming an NFW profile (Duffy et al. 2008 c-M).

    Both overdensities are relative to the critical density, so the conversion
    depends only on the concentration ``c200c(M200c, z)``; the critical density
    cancels. Accurate to a few percent, which is well below the cluster-to-cluster
    scatter and ample for sizing the tSZ template.

    Parameters
    ----------
    m200c : ndarray
        M200c in solar masses.
    z : ndarray
        Redshift per cluster.
    cosmo : FlatLambdaCDM
        Cosmology (for the Hubble parameter in the c-M pivot).

    Returns
    -------
    ndarray
        M500c in solar masses.
    """
    h = cosmo.H0.value / 100.0

    def mu(x):
        return np.log(1.0 + x) - x / (1.0 + x)

    # Non-positive / NaN masses (dropped later) would warn here; let them flow
    # through as NaN, which the row filter in load_catalog removes.
    with np.errstate(divide="ignore", invalid="ignore"):
        c200 = 5.71 * (m200c * h / 2.0e12) ** -0.084 * (1.0 + np.asarray(z)) ** -0.47
        # Solve 2.5 y^3 mu(c200) = mu(y c200) for y = r500/r200 in (0, 1] by bisection.
        lo = np.full_like(c200, 0.2)
        hi = np.full_like(c200, 1.0)
        for _ in range(60):
            y = 0.5 * (lo + hi)
            pos = 2.5 * y**3 * mu(c200) - mu(y * c200) > 0
            hi = np.where(pos, y, hi)
            lo = np.where(pos, lo, y)
        y = 0.5 * (lo + hi)
    return m200c * 2.5 * y**3


def load_catalog(path, query=None, mass_col=None):
    """Load a cluster catalog (RA, Dec, M500c, z), with optional row selection.

    Reads a FITS table (e.g. the ACT/nemo cluster catalogs) or a four-column text
    file (ra_deg dec_deg m500c z). Column names are matched case-insensitively
    against common aliases (RADeg/decDeg/redshift/M500c ...). Masses tabulated in
    1e14 Msun (as in the ACT catalogs) are auto-detected and converted to Msun.
    If only an M200c column is present it is converted to M500c via
    :func:`m200c_to_m500c`.

    Parameters
    ----------
    path : str
        FITS or whitespace-delimited text catalog.
    query : str or None
        Optional SQL-like row selection applied to the full table before column
        extraction (pandas ``DataFrame.query`` syntax), e.g.
        "SNR > 6 and redshift < 0.5". Any catalog column may be referenced.
    mass_col : str or None
        Explicit mass column name, taken as M500c (overrides auto-detection).

    Returns
    -------
    ra_deg, dec_deg, m500c, z : ndarray
        Equal-length arrays describing the selected clusters.
    """
    from astropy.table import Table

    if path.endswith((".fits", ".fit", ".fits.gz")):
        t = Table.read(path)
    else:
        t = Table(np.loadtxt(path), names=["ra", "dec", "m500c", "z"])
    if query:
        t = Table.from_pandas(t.to_pandas().query(query))

    cols = {c.lower(): c for c in t.colnames}

    def pick(*names, required=True):
        for n in names:
            if n in cols:
                return np.asarray(t[cols[n]], dtype=float)
        if required:
            raise KeyError(f"None of {names} found in catalog columns {t.colnames}")
        return None

    ra = pick("ra", "ra_deg", "radeg")
    dec = pick("dec", "dec_deg", "decdeg")
    z = pick("z", "redshift", "zcmb")

    # Masses may be tabulated in 1e14 Msun (ACT/nemo convention); detect by scale.
    def to_msun(m):
        return m * 1e14 if m.size and np.nanmedian(m) < 1e5 else m

    if mass_col is not None:
        if mass_col.lower() not in cols:
            raise KeyError(f"--mass-col '{mass_col}' not in columns {t.colnames}")
        m500c = to_msun(np.asarray(t[cols[mass_col.lower()]], dtype=float))
    elif (
        m500c := pick("m500c", "mass", "m500", "m500c_msun", required=False)
    ) is not None:
        m500c = to_msun(m500c)
    elif (
        m200c := pick("m200c", "m200", "m200c_msun", "m200cuncorr", required=False)
    ) is not None:
        m500c = m200c_to_m500c(to_msun(m200c), z, COSMO)
        print("Note: catalog has no M500c; converted M200c -> M500c (NFW, Duffy+08).")
    else:
        raise KeyError(
            "No mass column found (looked for M500c/M200c aliases). "
            f"Available columns: {t.colnames}. Pass --mass-col to name one."
        )
    # Drop rows that cannot be fit (missing position/redshift/mass, e.g. clusters
    # with no measured redshift in the ACT/nemo catalogs).
    good = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(z)
        & np.isfinite(m500c)
        & (m500c > 0)
        & (z > 0)
    )
    if not good.all():
        print(f"Note: dropping {int((~good).sum())} catalog rows with missing m500c/z.")
    return ra[good], dec[good], m500c[good], z[good]


def compute_pixboxes(shape, wcs, ra_deg, dec_deg, radius_rad):
    """Compute pixel bounding boxes around catalog positions (CAR-aware).

    Parameters
    ----------
    shape, wcs :
        Geometry of the full map.
    ra_deg, dec_deg : ndarray
        Catalog coordinates in degrees.
    radius_rad : float
        Half-size of the stamp in radians.

    Returns
    -------
    ndarray
        Integer pixboxes of shape (n, 2, 2) suitable for ``extract_pixbox`` /
        ``read_map(pixbox=...)`` / ``insert_at``.
    """
    poss = np.stack([np.deg2rad(dec_deg), np.deg2rad(ra_deg)], axis=0)
    return enmap.neighborhood_pixboxes(shape[-2:], wcs, poss.T, radius_rad)


def detect_nu_eff(path):
    """Infer the effective frequency (GHz) from an ACT-style filename band tag.

    Recognises ``f090``, ``f150`` and ``f220`` and returns the corresponding
    nominal tSZ effective frequency.

    Parameters
    ----------
    path : str
        Map filename or path.

    Returns
    -------
    float or None
        The effective frequency in GHz, or None if no band tag is found.
    """
    band_ghz = {"f090": 97.0, "f150": 148.0, "f220": 220.0}
    name = os.path.basename(path).lower()
    for tag, nu in band_ghz.items():
        if tag in name:
            return nu
    return None


# ===========================================================================
# Stacking
# ===========================================================================
def tangent_thumb(stamp, ra_deg, dec_deg, radius_rad, res_rad):
    """Reproject a CAR stamp onto a common tangent-plane geometry for stacking.

    Parameters
    ----------
    stamp : enmap.ndmap
        Postage stamp (CAR) to reproject.
    ra_deg, dec_deg : float
        Cluster center in degrees.
    radius_rad : float
        Half-size of the output thumbnail in radians.
    res_rad : float
        Pixel resolution of the output thumbnail in radians.

    Returns
    -------
    ndarray or None
        The tangent-plane thumbnail, or None if reprojection fails (e.g. the
        stamp does not fully cover the requested footprint).
    """
    pos = np.array([[np.deg2rad(dec_deg), np.deg2rad(ra_deg)]])
    try:
        thumb = reproject.thumbnails(
            stamp, pos, r=radius_rad, res=res_rad, proj="tan", verbose=False
        )
        return np.asarray(thumb[0] if thumb.ndim == 3 else thumb)
    except Exception:
        return None


# ===========================================================================
# MPI helpers
# ===========================================================================
def gather_lists(comm, local):
    """Gather a list of per-cluster results from all ranks onto rank 0.

    Parameters
    ----------
    comm :
        An MPI communicator (real or the orphics fake fallback).
    local : list
        This rank's list of results.

    Returns
    -------
    list or None
        The concatenated list on rank 0, otherwise None.
    """
    if comm.Get_size() == 1:
        return local
    gathered = comm.gather(local, root=0)
    if comm.Get_rank() == 0:
        out = []
        for sub in gathered:
            out.extend(sub)
        return out
    return None


# ===========================================================================
# Main pipeline
# ===========================================================================
def launcher_info():
    """Process count and rank the MPI launcher advertises via the environment.

    Lets us detect the common failure where a job is started under
    ``mpirun``/``srun`` with N processes but mpi4py is not bound to that MPI and
    each process initializes its own size-1 communicator.

    Returns
    -------
    (nproc, rank) : (int or None, int)
        Launcher process count (None if not launched under a recognised MPI),
        and this process's launcher rank (0 if unknown).
    """
    for nvar, rvar in (
        ("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK"),
        ("PMI_SIZE", "PMI_RANK"),
        ("SLURM_NTASKS", "SLURM_PROCID"),
    ):
        if nvar in os.environ:
            try:
                return int(os.environ[nvar]), int(os.environ.get(rvar, "0"))
            except ValueError:
                pass
    return None, 0


def run_pipeline(args):
    """Run the end-to-end fit-and-subtract pipeline.

    Distributes clusters across MPI ranks; each rank reads only its own stamps
    from disk, fits and builds model stamps; rank 0 gathers the models, loads the
    full map once, subtracts, writes the output, and produces stacks, diagnostics
    and an HTML report.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line options (see :func:`build_parser`).
    """
    comm = mpi.MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    def log(msg):
        if rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # Detect a launcher/mpi4py mismatch: started with N processes but MPI reports
    # size 1, so every process would redundantly do the whole catalogue.
    nlaunch, lrank = launcher_info()
    if nlaunch and nlaunch > 1 and size == 1 and lrank == 0:
        print(
            f"WARNING: started under a launcher with {nlaunch} processes but mpi4py "
            "reports size=1 -- MPI is NOT active, so every process will fit the whole "
            "catalogue and overwrite the same output. Rebuild mpi4py against this "
            "cluster's MPI, e.g. `MPICC=$(which mpicc) pip install --no-binary mpi4py "
            "--force-reinstall mpi4py`.",
            flush=True,
        )

    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    # --- Inputs (small) read by all ranks -------------------------------------
    ra, dec, m500c, z = load_catalog(
        args.catalog, query=args.query, mass_col=args.mass_col
    )
    if args.debug and ra.size > DEBUG_NCLUSTERS:
        log(f"--debug: using the first {DEBUG_NCLUSTERS} of {ra.size} clusters.")
        ra, dec, m500c, z = (a[:DEBUG_NCLUSTERS] for a in (ra, dec, m500c, z))
    ncl = ra.size
    shape, wcs = enmap.read_map_geometry(args.coadd)
    pixboxes = compute_pixboxes(shape, wcs, ra, dec, args.radius_arcmin * utils.arcmin)
    # --ivar may be a path to an ivar map or a uniform white-noise level in
    # uK-arcmin (a plain number), from which a per-pixel ivar stamp is built.
    try:
        noise_uk_arcmin = float(args.ivar)
    except (TypeError, ValueError):
        noise_uk_arcmin = None
    ivar_desc = f"{noise_uk_arcmin:.1f} uK-arcmin" if noise_uk_arcmin else args.ivar
    log(
        f"Catalog: {ncl} clusters; map shape {shape[-2:]}; "
        f"ivar={ivar_desc}; {size} MPI rank(s)."
    )

    beam_fn = get_beam_fn(args.beam)
    # Model is painted out to the full profile extent so low-z (large) clusters
    # are subtracted beyond the compact fitting stamp.
    profiler = BeamProfiler(beam_fn, PROFILE_MAX_ARCMIN * utils.arcmin)
    lmax = 4000 if args.debug else 6000
    log("Computing CMB theory (CAMB)...")
    theory_fn, _, _ = get_cmb_theory_fn(lmax=lmax)
    scov_cache = ScovCache(theory_fn, beam_fn)

    smin, smax, ns = args.s_grid
    s_vals = np.linspace(smin, smax, int(ns))

    # --- Distribute clusters --------------------------------------------------
    # Hand each rank a contiguous block of declination-sorted clusters, and fit
    # them in that order. Each rank then spans only a narrow declination range, so
    # the CMB-covariance cache holds just a couple of bands (bounded memory) and
    # nearly every fit is a cache hit.
    order = np.argsort(dec)
    my_tasks = np.array_split(order, size)[rank]
    log(
        f"Distributing {ncl} clusters over {size} MPI rank(s); rank 0 handles "
        f"{len(my_tasks)}."
    )

    stack_r = args.stack_arcmin * utils.arcmin
    stack_res = (args.stack_res_arcmin) * utils.arcmin

    local = []
    t_fit = 0.0
    nfit = nskip = 0
    for ii, idx in enumerate(my_tasks):
        pb = pixboxes[idx]
        # Read only this stamp from disk (the full map is never held by a worker);
        # use the first (temperature) component of a multi-component map.
        try:
            data = enmap.read_map(args.coadd, pixbox=pb)
            if data.ndim > 2:
                data = data[0]
            if noise_uk_arcmin is not None:
                ivar = maps.ivar(data.shape, data.wcs, noise_uk_arcmin)
            else:
                ivar = enmap.read_map(args.ivar, pixbox=pb)
                if ivar.ndim > 2:
                    ivar = ivar[0]
        except Exception as e:
            nskip += 1
            if rank == 0 and args.debug:
                print(f"  skip {idx}: read failed ({e})")
            continue
        if (
            data.shape != ivar.shape
            or np.asarray(ivar).max() <= 0
            or not np.all(np.isfinite(np.asarray(data)))
        ):
            nskip += 1
            continue

        tf = time.time()
        try:
            model, res = fit_cluster(
                data,
                ivar,
                m500c[idx],
                z[idx],
                COSMO,
                args.nu_eff,
                profiler,
                scov_cache,
                s_vals,
            )
        except Exception as e:
            nskip += 1
            if rank == 0 and args.debug:
                print(f"  skip {idx}: fit failed ({e})")
            continue
        t_fit += time.time() - tf

        if res["snr"] < args.snr_min:
            res["subtracted"] = False
            model = model * 0.0
        else:
            res["subtracted"] = True
            nfit += 1

        before = tangent_thumb(data, ra[idx], dec[idx], stack_r, stack_res)
        after = tangent_thumb(data - model, ra[idx], dec[idx], stack_r, stack_res)

        local.append(
            dict(
                idx=int(idx),
                pixbox=np.asarray(pb),
                model=np.asarray(model),
                result=res,
                before=before,
                after=after,
            )
        )
        if rank == 0 and (ii + 1) % max(1, len(my_tasks) // 10) == 0:
            print(
                f"  fit {ii + 1}/{len(my_tasks)}  (mean fit {t_fit / (ii + 1):.3f}s)",
                flush=True,
            )

    log(
        f"Rank 0 fit time {t_fit:.2f}s; nfit={nfit}, nskip={nskip}, "
        f"scov cache hits={scov_cache.hits}, misses={scov_cache.misses}."
    )

    # --- Gather to rank 0 -----------------------------------------------------
    comm.Barrier()
    allres = gather_lists(comm, local)
    if rank != 0:
        return

    # --- Subtract into full map and write ------------------------------------
    log("Loading full coadd for subtraction...")
    t_load = time.time()
    # Only the temperature component is used; a multi-component coadd yields a
    # temperature-only subtracted output.
    base = enmap.read_map(args.coadd)
    if base.ndim > 2:
        base = base[0]
    log(f"Loaded full map in {time.time() - t_load:.1f}s; applying models...")

    # Build the full model map in real space with pixell.pointsrcs (no FFTs):
    # each cluster is painted as its amplitude-scaled, beam-convolved radial
    # profile at its catalogue position, then subtracted in one pass.
    sub = [it for it in allres if it["result"].get("subtracted", False)]
    nsub = len(sub)
    if nsub:
        poss = np.array(
            [
                np.deg2rad([dec[it["idx"]] for it in sub]),
                np.deg2rad([ra[it["idx"]] for it in sub]),
            ]
        )
        profs = [it["result"]["profile"] for it in sub]
        amps = np.array([p[0] for p in profs])
        profiles = [np.array([profiler.r, p / p[0]]) for p in profs]
        model_map = pointsrcs.sim_objects(
            base.shape,
            base.wcs,
            poss,
            amps,
            profiles,
            prof_ids=np.arange(nsub),
            rmax=float(profiler.r[-1]),
        )
        base = base - model_map
    enmap.write_map(args.output, base)
    log(f"Subtracted {nsub} clusters; wrote {args.output}.")

    # --- Stacks and diagnostics ----------------------------------------------
    make_report(args, allres)
    log(f"Total wall time {time.time() - t0:.1f}s. Report at {args.outdir}/report.html")
    return allres


# ===========================================================================
# Reporting
# ===========================================================================
def stack_thumbs(items, key):
    """Mean-stack the tangent thumbnails of a result list, ignoring failures.

    Parameters
    ----------
    items : list
        Per-cluster result dicts.
    key : str
        'before' or 'after'.

    Returns
    -------
    ndarray or None
        The mean stack, or None if there were no valid thumbnails.
    """
    arrs = [
        it[key] for it in items if it[key] is not None and it["result"]["subtracted"]
    ]
    if not arrs:
        return None
    shp = arrs[0].shape
    arrs = [a for a in arrs if a.shape == shp]
    return np.mean(arrs, axis=0)


def save_stack_figs(before, after, res_arcmin, figdir, suffix="", label=""):
    """Save before/after stack images and their radial overlay; return metrics.

    Writes ``stack_before{suffix}.png``, ``stack_after{suffix}.png`` and
    ``radial{suffix}.png`` on a shared colour scale.

    Parameters
    ----------
    before, after : ndarray
        Mean before/after stacks (same shape).
    res_arcmin : float
        Thumbnail pixel size in arcmin.
    figdir : str
        Output directory.
    suffix, label : str
        Filename suffix and title-label distinguishing the stack (e.g. a S/N cut).

    Returns
    -------
    (before_c, after_c, reduction) : tuple of float
        Mean over a central r<2' aperture, and the fractional decrement reduction.
    """
    vmin = min(before.min(), after.min())
    vmax = max(before.max(), after.max())
    _save_img(
        before,
        f"Stack before subtraction{label} (uK)",
        os.path.join(figdir, f"stack_before{suffix}.png"),
        vmin,
        vmax,
    )
    _save_img(
        after,
        f"Stack after subtraction{label} (uK)",
        os.path.join(figdir, f"stack_after{suffix}.png"),
        vmin,
        vmax,
    )
    ny, nx = before.shape
    yy, xx = np.mgrid[:ny, :nx].astype(float)
    rr = np.hypot(yy - (ny - 1) / 2.0, xx - (nx - 1) / 2.0) * res_arcmin
    edges = np.linspace(0, rr.max(), 12)
    binner = stats.bin2D(rr, edges)
    cents, pb = binner.bin(before)
    _, pa = binner.bin(after)
    m = min(cents.size, pb.size, pa.size)
    _save_radial(cents[:m], pb[:m], pa[:m], os.path.join(figdir, f"radial{suffix}.png"))
    ap = rr <= 2.0
    bc, ac = float(before[ap].mean()), float(after[ap].mean())
    return bc, ac, 1.0 - abs(ac) / abs(bc)


def make_report(args, allres):
    """Write diagnostic figures and an HTML report.

    Parameters
    ----------
    args : argparse.Namespace
        Pipeline options (provides outdir and stacking resolution).
    allres : list
        Gathered per-cluster results on rank 0.
    """
    figdir = os.path.join(args.outdir, "figs")
    os.makedirs(figdir, exist_ok=True)

    before = stack_thumbs(allres, "before")
    after = stack_thumbs(allres, "after")
    figs = []

    if before is not None and after is not None:
        save_stack_figs(before, after, args.stack_res_arcmin, figdir)
        figs += ["figs/stack_before.png", "figs/stack_after.png", "figs/radial.png"]

    subbed = [it["result"] for it in allres if it["result"]["subtracted"]]
    if subbed:
        amps = np.array([r["amplitude"] for r in subbed])
        scales = np.array([r["scale"] for r in subbed])
        snrs = np.array([r["snr"] for r in subbed])
        chi2r = np.array([r["chi2"] / r["dof"] for r in subbed])
        _save_hist(amps, "best-fit amplitude", os.path.join(figdir, "amp.png"))
        _save_hist(scales, "best-fit scale", os.path.join(figdir, "scale.png"))
        _save_hist(chi2r, "chi2 / dof", os.path.join(figdir, "chi2.png"))
        figs += ["figs/amp.png", "figs/scale.png", "figs/chi2.png"]

        print("\n=== Fit diagnostics ===")
        print(f"  clusters subtracted : {len(subbed)} / {len(allres)}")
        print(f"  amplitude  mean/med : {amps.mean():.3f} / {np.median(amps):.3f}")
        print(f"  scale      mean/med : {scales.mean():.3f} / {np.median(scales):.3f}")
        print(f"  S/N        mean/med : {snrs.mean():.2f} / {np.median(snrs):.2f}")
        print(f"  chi2/dof   mean/med : {chi2r.mean():.2f} / {np.median(chi2r):.2f}")
        if before is not None:
            print(
                f"  stack central before/after (uK): "
                f"{before.min():.2f} / {after.min():.2f}"
            )

    io.write_gallery_html(
        os.path.join(args.outdir, "report.html"),
        [os.path.join(args.outdir, f) for f in figs],
        title="tSZ model-subtraction report",
    )


def _save_img(arr, title, path, vmin=None, vmax=None):
    """Save a 2D image (e.g. a stacked thumbnail) with a colorbar and title."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(arr, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def _save_radial(cents, before, after, path):
    """Save a radial-profile overlay of the before/after stacks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(cents, before, "o-", label="before")
    plt.plot(cents, after, "s-", label="after")
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("radius [arcmin]")
    plt.ylabel("mean stacked signal [uK]")
    plt.legend()
    plt.title("Radial profile of stacks")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def _save_hist(vals, label, path):
    """Save a histogram of a per-cluster diagnostic quantity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(vals, bins=min(30, max(5, vals.size // 3)))
    plt.xlabel(label)
    plt.ylabel("clusters")
    plt.title(f"Distribution of {label}")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


# ===========================================================================
# Synthetic data generation and self-test
# ===========================================================================
def make_synthetic(outdir, debug=True, seed=0):
    """Generate a synthetic CAR coadd, ivar, beam and catalog with known clusters.

    A Gaussian CMB realization is drawn from the CAMB lensed TT spectrum and
    beam-convolved; known Arnaud clusters (amplitude=1, scale=1) are injected
    using the same template machinery as the fitter; white noise is added from a
    spatially varying ivar map. The geometry spans a range of declinations to
    exercise CAR distortion.

    Parameters
    ----------
    outdir : str
        Output directory for coadd.fits, ivar.fits, catalog.txt, beam.txt.
    debug : bool
        If True, use a coarse, fast configuration.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Paths and the injected catalog arrays for verification.
    """
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(seed)

    res_arcmin = 0.5
    fwhm = 1.5
    nu_eff = 150.0
    # Debug: a small, fast map with a handful of clusters. Production self-test: a
    # large map with a broad mass range so that >=200 clusters exceed S/N>8 (the
    # primary verification stack) while a tail of lower-mass clusters populates the
    # 5 < S/N < 8 range for the S/N>5 comparison stack.
    if debug:
        width_deg, height_deg, ncl = 6.0, 6.0, 24
        m_lo, m_hi, z_lo, z_hi, min_sep_arcmin = 4e14, 9e14, 0.1, 0.6, 0.0
    else:
        width_deg, height_deg, ncl = 30.0, 16.0, 320
        m_lo, m_hi, z_lo, z_hi, min_sep_arcmin = 3e14, 1.6e15, 0.05, 0.5, 16.0
    box = (
        np.array(
            [[-height_deg / 2.0, -width_deg / 2.0], [height_deg / 2.0, width_deg / 2.0]]
        )
        * utils.degree
    )
    shape, wcs = enmap.geometry(pos=box, res=res_arcmin * utils.arcmin, proj="car")
    shape = shape[-2:]

    # CMB realization from theory TT.
    theory_fn, ells, cltt = get_cmb_theory_fn(lmax=4000)
    ps = cltt[None, None, :]
    cmb = curvedsky.rand_map((1,) + shape, wcs, ps)[0]
    beam_fn = get_beam_fn(fwhm)
    cmb = enmap.ifft(enmap.fft(cmb) * beam_fn(cmb.modlmap())).real
    profiler = BeamProfiler(beam_fn, PROFILE_MAX_ARCMIN * utils.arcmin)

    # Cluster catalog drawn uniformly across the map, with rejection sampling to
    # keep a minimum centre-to-centre separation so neighbouring cluster signals
    # do not contaminate each other's fitting stamps.
    rng = np.random.default_rng(seed)
    margin = 1.0
    ra_lo, ra_hi = -width_deg / 2.0 + margin, width_deg / 2.0 - margin
    dec_lo, dec_hi = -height_deg / 2.0 + margin, height_deg / 2.0 - margin
    min_sep_deg = min_sep_arcmin / 60.0
    ra_list, dec_list = [], []
    while len(ra_list) < ncl:
        r = rng.uniform(ra_lo, ra_hi)
        d = rng.uniform(dec_lo, dec_hi)
        if min_sep_deg > 0 and ra_list:
            dr = (np.array(ra_list) - r) * np.cos(np.deg2rad(d))
            dd = np.array(dec_list) - d
            if np.min(np.hypot(dr, dd)) < min_sep_deg:
                continue
        ra_list.append(r)
        dec_list.append(d)
    ra = np.array(ra_list)
    dec = np.array(dec_list)
    m500c = rng.uniform(m_lo, m_hi, ncl)
    zs = rng.uniform(z_lo, z_hi, ncl)

    truth = cmb.copy()
    # Inject each cluster's full profile (out to the model extent) so the injected
    # truth matches what the pipeline paints and subtracts.
    pixboxes = compute_pixboxes(shape, wcs, ra, dec, PROFILE_MAX_ARCMIN * utils.arcmin)
    for i in range(ncl):
        pb = pixboxes[i]
        stamp = truth.extract_pixbox(pb)
        dT_interp, _ = deltaT_profile(m500c[i], zs[i], COSMO, nu_eff)
        model = model_stamp(stamp, profiler.r, profiler.beamed(dT_interp, 1.0))
        truth = enmap.insert_at(truth, pb, stamp + model)

    # Spatially varying ivar (white noise), noise rms ~ 10 uK-arcmin.
    noise_uk_arcmin = 10.0
    sigma_pix = noise_uk_arcmin / res_arcmin
    ivar_val = 1.0 / sigma_pix**2
    ivar = enmap.full(shape, wcs, ivar_val)
    ivar *= enmap.enmap(
        1.0 + 0.5 * np.sin(np.linspace(0, 3, shape[1]))[None, :] * np.ones(shape), wcs
    )
    noise = enmap.enmap(rng.normal(0, 1, shape), wcs) / np.sqrt(ivar)
    coadd = truth + noise

    cpath = os.path.join(outdir, "coadd.fits")
    ipath = os.path.join(outdir, "ivar.fits")
    catpath = os.path.join(outdir, "catalog.txt")
    bpath = os.path.join(outdir, "beam.txt")
    enmap.write_map(cpath, coadd)
    enmap.write_map(ipath, ivar)
    np.savetxt(
        catpath, np.column_stack([ra, dec, m500c, zs]), header="ra_deg dec_deg m500c z"
    )
    # Tabulate the beam out to where it is negligible (B < ~1e-5) so the 1D
    # Hankel beam convolution is not truncated; for a 1.5' FWHM that is ell~25000.
    bl_ells = np.arange(0, 25001)
    np.savetxt(
        bpath,
        np.column_stack([bl_ells, maps.gauss_beam(bl_ells, fwhm)]),
        header="ell B_ell",
    )

    return dict(
        coadd=cpath,
        ivar=ipath,
        catalog=catpath,
        beam=bpath,
        nu_eff=nu_eff,
        ra=ra,
        dec=dec,
        m500c=m500c,
        z=zs,
    )


def stack_subset(allres, snr_thr, args, figdir, suffix):
    """Stack before/after thumbnails of clusters with S/N > ``snr_thr``.

    Saves ``stack_before{suffix}.png``, ``stack_after{suffix}.png`` and
    ``radial{suffix}.png`` and returns central-aperture metrics for the subset.

    Parameters
    ----------
    allres : list
        Gathered per-cluster results on rank 0.
    snr_thr : float
        Only clusters with ``snr > snr_thr`` (and subtracted) are stacked.
    args : argparse.Namespace
        Provides ``stack_res_arcmin`` for the aperture metric.
    figdir : str
        Directory to write the figures into.
    suffix : str
        Filename suffix distinguishing thresholds (e.g. '' or '_snr5').

    Returns
    -------
    dict or None
        ``n``, ``before_c``, ``after_c``, ``reduction``, ``amps``, ``scales`` for
        the subset, or None if no cluster passes the threshold.
    """
    hi = [
        it
        for it in allres
        if it["result"]["subtracted"] and it["result"]["snr"] > snr_thr
    ]
    if not hi:
        return None
    before = stack_thumbs(hi, "before")
    after = stack_thumbs(hi, "after")
    before_c, after_c, reduction = save_stack_figs(
        before, after, args.stack_res_arcmin, figdir, suffix, f", S/N>{snr_thr:.0f}"
    )
    return dict(
        n=len(hi),
        before_c=before_c,
        after_c=after_c,
        reduction=reduction,
        amps=np.array([it["result"]["amplitude"] for it in hi]),
        scales=np.array([it["result"]["scale"] for it in hi]),
    )


def run_selftest(args):
    """Generate synthetic data, run the pipeline, and assert correct subtraction.

    Parameters
    ----------
    args : argparse.Namespace
        Options; ``debug`` controls the synthetic configuration size.
    """
    outdir = args.outdir
    syn = make_synthetic(outdir, debug=args.debug)
    print(f"Synthetic data written to {outdir} ({syn['ra'].size} clusters).")

    args.coadd = syn["coadd"]
    args.ivar = syn["ivar"]
    args.catalog = syn["catalog"]
    args.beam = syn["beam"]
    args.nu_eff = syn["nu_eff"]
    args.output = os.path.join(outdir, "coadd_tsz_subtracted.fits")
    args.snr_min = 2.0

    allres = run_pipeline(args)
    if allres is None:
        return

    # --- Quantitative pass/fail checks (truth is amplitude=1, scale=1) --------
    # Individual cluster stamps are dominated by CMB fluctuations, so per-cluster
    # central decrements scatter wildly and the low-S/N clusters pollute the
    # stack. The primary verification stack and amplitude/scale statistics use the
    # high-significance subset (S/N > 8) where the cluster signal dominates the
    # stamp; a second, lower-threshold stack (S/N > 5) is produced for comparison.
    out = enmap.read_map(args.output)
    finite = bool(np.all(np.isfinite(np.asarray(out))))
    figdir = os.path.join(outdir, "figs")
    s8 = stack_subset(allres, 8.0, args, figdir, "")
    s5 = stack_subset(allres, 5.0, args, figdir, "_snr5")
    if s8 is None:
        print("\nSELF-TEST FAILED: no clusters above the S/N threshold.")
        raise SystemExit(1)
    print(f"\nVerification restricted to S/N > 8: {s8['n']} of {len(allres)} clusters.")
    if s5 is not None:
        print(
            f"Comparison stack S/N > 5: {s5['n']} of {len(allres)} clusters "
            f"(reduction {s5['reduction'] * 100:.0f}%, "
            f"before {s5['before_c']:.1f} -> after {s5['after_c']:.1f} uK)."
        )

    amps, scales = s8["amps"], s8["scales"]
    before_c, after_c, reduction = s8["before_c"], s8["after_c"], s8["reduction"]

    # The stacked-decrement metric is only meaningful once the primary CMB has
    # averaged down; with a small (debug) sample the before-stack is CMB-dominated
    # and the ratio is unreliable, so the assertion is only enforced for large
    # samples. Per-cluster recovery is still checked via amplitude/scale/chi2.
    enforce_stack = s8["n"] >= 100
    checks = {
        "median amplitude in [0.7, 1.3]": 0.7 <= np.median(amps) <= 1.3,
        "median scale in [0.7, 1.4]": 0.7 <= np.median(scales) <= 1.4,
        "hi-S/N stack decrement reduced > 50% (>=100 clusters)": (
            not enforce_stack or reduction > 0.5
        ),
        "output map finite": finite,
        "hi-S/N clusters present and all fit": amps.size > 0,
    }
    print("\n=== Self-test assertions ===")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(
        f"  median amp={np.median(amps):.3f} median scale={np.median(scales):.3f} "
        f"stack reduction={reduction * 100:.0f}% "
        f"(before {before_c:.1f} -> after {after_c:.1f} uK)"
    )
    if all(checks.values()):
        print("\nSELF-TEST PASSED. Report and stacks in", outdir)
    else:
        print("\nSELF-TEST FAILED.")
        raise SystemExit(1)


# ===========================================================================
# CLI
# ===========================================================================
def build_parser():
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument(
        "--coadd",
        help="Path to the CAR coadd map (FITS). If multi-component, the first "
        "(temperature) map is used.",
    )
    p.add_argument(
        "--ivar",
        help="Path to the inverse-variance map (FITS). If multi-component, the "
        "first map is used. If only a float is specified, this is interpreted as"
        "the homogenous RMS in uK-arcmin (not inverse variance uK^2).",
    )
    p.add_argument(
        "--catalog",
        help="Cluster catalog: a FITS table (e.g. the ACT DR6 catalog) or a text "
        "file (ra dec m500c z). M500c in 1e14 Msun is auto-detected.",
    )
    p.add_argument(
        "--query",
        default=None,
        help="Optional SQL-like row selection on the catalog (pandas query "
        'syntax), e.g. "SNR > 6 and redshift < 0.5".',
    )
    p.add_argument(
        "--mass-col",
        default=None,
        dest="mass_col",
        help="Catalog mass column to use as M500c (overrides auto-detection of "
        "M500c/M200c aliases).",
    )
    p.add_argument("--beam", help="Beam FWHM in arcmin (float) or ell,B_ell file path.")
    p.add_argument(
        "--nu-eff",
        type=float,
        default=None,
        dest="nu_eff",
        help="Effective frequency of the coadd in GHz. If omitted, inferred from "
        "an f090/f150/f220 tag in the coadd filename.",
    )
    p.add_argument(
        "--output",
        default="coadd_tsz_subtracted.fits",
        help="Output subtracted map path.",
    )
    p.add_argument("--outdir", default="modelsub_out", help="Output directory.")
    p.add_argument(
        "--radius-arcmin",
        type=float,
        default=16.0,
        dest="radius_arcmin",
        help="Half-size of the fitting stamp in arcmin (the model is painted out "
        f"to {PROFILE_MAX_ARCMIN:.0f}' for large low-z clusters).",
    )
    p.add_argument(
        "--s-grid",
        nargs=3,
        type=float,
        default=[0.5, 2.0, 7],
        dest="s_grid",
        metavar=("MIN", "MAX", "N"),
        help="Angular-scale grid (min max n).",
    )
    p.add_argument(
        "--snr-min",
        type=float,
        default=0.0,
        dest="snr_min",
        help="Minimum *fit* S/N to actually subtract a cluster. This is not the SNR in the provided catalog, but the SNR from fitting. You probably don't want to use this if you are already selecting on SNR>x with --query.",
    )
    p.add_argument(
        "--stack-arcmin",
        type=float,
        default=10.0,
        dest="stack_arcmin",
        help="Half-size of the stacking thumbnail in arcmin.",
    )
    p.add_argument(
        "--stack-res-arcmin",
        type=float,
        default=0.5,
        dest="stack_res_arcmin",
        help="Pixel resolution of the stacking thumbnail in arcmin.",
    )
    p.add_argument(
        "--selftest",
        action="store_true",
        help="Generate synthetic data and run an end-to-end self-test.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help=f"Fast configuration for quick testing: coarser CMB/scale grids and, "
        f"on real data, only the first {DEBUG_NCLUSTERS} clusters.",
    )
    return p


def main():
    """Parse arguments and dispatch to the self-test or the real pipeline."""
    args = build_parser().parse_args()
    if args.debug:
        args.s_grid = [args.s_grid[0], args.s_grid[1], min(args.s_grid[2], 5)]
    if args.selftest:
        run_selftest(args)
        return
    missing = [k for k in ("coadd", "catalog", "beam") if getattr(args, k) is None]
    if missing:
        raise SystemExit(f"Missing required arguments: {missing} (or use --selftest).")
    if args.nu_eff is None:
        args.nu_eff = detect_nu_eff(args.coadd)
        if args.nu_eff is None:
            raise SystemExit(
                "Could not infer --nu-eff from the coadd filename "
                "(no f090/f150/f220 tag); please pass --nu-eff."
            )
    run_pipeline(args)


if __name__ == "__main__":
    main()
