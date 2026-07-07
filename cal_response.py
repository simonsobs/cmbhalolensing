"""Calibrate the cmbhalolensing QE response on NFW-lensed Gaussian sims.

Produces a Fourier-space transfer function
    R(L) = <Re[response_k * conj(ktrue_k)]> / <|ktrue_k|^2>
where ``response_k = QE(lensed) - QE(unlensed)`` on paired Gaussian-CMB
sims.  The pairing makes the mean field cancel nearly exactly; no foregrounds
or noise are added (R(L) depends only on the QE filter weights, not on
the data itself).

The same script can produce calibration products for both ``recon_sim.py``
and ``stack.py`` by changing the filter parameters via ``--config`` or
CLI flags.  Two presets are included: ``input/cal_recon_sim.yml`` and
``input/cal_stack.yml``.

Outputs (under ``--output-dir``):
    cal.npz       (ell_edges, ell_centers, R_L, R_L_err, auto_power, cross_mean)
    cal.yaml      (full filter parameter dict + NFW metadata)

Apply with ``--calibration <output_dir>`` in recon_sim.py / stack.py.

MPI-parallel via orphics.mpi.distribute.
"""

import argparse
import os
import time

import numpy as np
import yaml

from pixell import enmap, utils, lensing as enlensing
from orphics import cosmology, lensing as olensing, maps, mpi, stats
from symlens.qe import QE

from inpaint_utils import Inpainter


# NFW kappa

def _gnfw(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    gt = x > 1
    lt = (x < 1) & (x > 0)
    eq = np.abs(x - 1) < 1e-6
    y = x[gt]
    out[gt] = (1 / (y**2 - 1)) * (
        1 - 2 / np.sqrt(y**2 - 1) * np.arctan(np.sqrt((y - 1) / (y + 1)))
    )
    y = x[lt]
    out[lt] = (1 / (y**2 - 1)) * (
        1 - 2 / np.sqrt(1 - y**2) * np.arctanh(np.sqrt((1 - y) / (1 + y)))
    )
    out[eq] = 1.0 / 3.0
    return out


def nfw_kappa_map(shape, wcs, M200c, z, c200c):
    """Projected NFW convergence on an enmap grid (colossus cosmology)."""
    from colossus.cosmology import cosmology as col_cosmo
    col_cosmo.setCosmology("planck18",persistence="")
    cosmo = col_cosmo.getCurrent()
    rho_c = cosmo.rho_c(z)
    R200c_kpc = (3 * M200c / (4 * np.pi * 200 * rho_c)) ** (1 / 3)
    rs_Mpc = R200c_kpc / (c200c * 1000)
    chi_l = cosmo.comovingDistance(0, z)
    h = cosmo.H0 / 100.0
    chi_s = 14000.0 * h
    theta_s = rs_Mpc / chi_l
    win = (chi_s - chi_l) / chi_s
    fc = np.log(1 + c200c) - c200c / (1 + c200c)
    prefactor = 9.571e-20 * chi_l * (1 + z) * win * M200c / (rs_Mpc**2 * fc)
    modrmap = enmap.modrmap(shape, wcs)
    x = modrmap / theta_s
    kappa = enmap.enmap(prefactor * _gnfw(x), wcs)
    ny, nx = shape[-2:]
    if ny % 2 == 1 and nx % 2 == 1:
        cy, cx = ny // 2, nx // 2
        kappa[cy, cx] = kappa[cy - 1, cx]
    return kappa


# ILC

def ilc_two_channel(modlmap, k1, k2, p11, p22, p12, b1, b2, ell_min, ell_max):
    """Beam-deconvolved ILC of two channels with band restriction."""
    sel = np.logical_and(modlmap >= ell_min, modlmap <= ell_max)
    nells = int(sel.sum())
    cov = np.zeros((nells, 2, 2))
    cov[:, 0, 0] = p11[sel]
    cov[:, 1, 1] = p22[sel]
    cov[:, 0, 1] = p12[sel]
    cov[:, 1, 0] = p12[sel]
    ms = np.stack([k1[sel], k2[sel]]).swapaxes(0, 1)
    rs = np.stack([b1[sel], b2[sel]]).swapaxes(0, 1)
    num = np.array([np.linalg.solve(cov[i], ms[i]) for i in range(nells)])
    den = np.array([np.linalg.solve(cov[i], rs[i]) for i in range(nells)])
    tcov = 1.0 / np.einsum("ij,ij->i", rs, den)
    out = np.zeros_like(k1)
    out[sel] = np.einsum("ij,ij->i", rs, num) * tcov
    tret = np.zeros_like(p11)
    tret[sel] = tcov
    return out, tret


# Parameter handling
FILTER_KEYS = (
    "pix_arcmin", "stamp_width_arcmin",
    "taper_percent", "pad_percent",
    "grad_lmin", "grad_lmax",
    "hres_lmin", "hres_lmax", "hres_lxcut", "hres_lycut",
    "kappa_lmin", "kappa_lmax",
    "grad_beam_fwhm", "grad_noise_uK_arcmin",
    "hres_beam_fwhms", "hres_noise_uK_arcmin",
    "inpaint_radius_arcmin",
    "inpaint_context_arcmin", "inpaint_pcov_res_arcmin",
    "qe_estimator",
)

_DEFAULTS = {
    "pix_arcmin": 0.5,
    "stamp_width_arcmin": 120.0,
    "taper_percent": 20.0,
    "pad_percent": 3.0,
    "grad_lmin": 200,
    "grad_lmax": 2000,
    "hres_lmin": 200,
    "hres_lmax": 3500,
    "hres_lxcut": 2,
    "hres_lycut": 2,
    "kappa_lmin": 200,
    "kappa_lmax": 3000,
    "grad_beam_fwhm": 5.0,
    "grad_noise_uK_arcmin": 35.0,
    "hres_beam_fwhms": [1.5, 2.2],
    "hres_noise_uK_arcmin": 15.0,
    "inpaint_radius_arcmin": 0.0,
    "inpaint_context_arcmin": 60.0,
    "inpaint_pcov_res_arcmin": 2.0,
    "qe_estimator": "hdv",
}


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--config", default=None,
                    help="YAML with filter parameters (CLI args override).")
    ap.add_argument("--output-dir", required=True)
    for k, v in _DEFAULTS.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, list):
            ap.add_argument(flag, type=float, nargs="+", default=None)
        elif isinstance(v, bool):
            ap.add_argument(flag, action="store_true", default=None)
        elif isinstance(v, int) and not isinstance(v, bool):
            ap.add_argument(flag, type=int, default=None)
        elif isinstance(v, float):
            ap.add_argument(flag, type=float, default=None)
        else:
            ap.add_argument(flag, type=str, default=None)
    ap.add_argument("--ell-bin-width", type=int, default=200)
    ap.add_argument("--nsims", type=int, default=2000)
    ap.add_argument("--mass", type=float, default=3.5e14)
    ap.add_argument("--redshift", type=float, default=0.5)
    ap.add_argument("--concentration", type=float, default=3.2)
    ap.add_argument("--seed", type=int, default=99999)
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


def resolve_params(args):
    """Merge defaults, --config YAML, and CLI overrides (in that order)."""
    p = dict(_DEFAULTS)
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if k in _DEFAULTS:
                p[k] = v
    for k in _DEFAULTS:
        val = getattr(args, k, None)
        if val is not None:
            p[k] = val
    p["ell_bin_width"] = args.ell_bin_width
    p["nsims"] = 10 if args.debug else args.nsims
    p["mass"] = args.mass
    p["redshift"] = args.redshift
    p["concentration"] = args.concentration
    p["seed"] = args.seed
    p["hres_beam_fwhms"] = [float(f) for f in p["hres_beam_fwhms"]]
    return p


# Sim loop

def main():
    args = parse_args()
    params = resolve_params(args)

    comm, rank, my_tasks = mpi.distribute(params["nsims"])
    size = comm.Get_size()

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[cal] output: {args.output_dir}")
        print(f"[cal] nsims={params['nsims']}, ranks={size}")
        print(f"[cal] stamp={params['stamp_width_arcmin']}', "
              f"pix={params['pix_arcmin']}', taper={params['taper_percent']}%")
        print(f"[cal] grad: B={params['grad_beam_fwhm']}', "
              f"n={params['grad_noise_uK_arcmin']} uK-arcmin, "
              f"ell=[{params['grad_lmin']}, {params['grad_lmax']}]")
        print(f"[cal] hres: B={params['hres_beam_fwhms']}', "
              f"n={params['hres_noise_uK_arcmin']} uK-arcmin, "
              f"ell=[{params['hres_lmin']}, {params['hres_lmax']}], "
              f"lxcut={params['hres_lxcut']}, lycut={params['hres_lycut']}")
        print(f"[cal] inpaint radius: {params['inpaint_radius_arcmin']}'")

    # Geometry
    w = (params["stamp_width_arcmin"] / 60.0) / 2 * utils.degree
    r = params["pix_arcmin"] * utils.arcmin
    shape, wcs = enmap.geometry(pos=[[-w, -w], [w, w]], res=r, proj="car")
    modlmap = enmap.modlmap(shape, wcs)

    taper_arr, _ = maps.get_taper(shape, wcs,
                                  taper_percent=params["taper_percent"],
                                  pad_percent=params["pad_percent"],
                                  weight=None)

    # Beams
    Bplc = maps.gauss_beam(modlmap, params["grad_beam_fwhm"])
    hres_fwhms = list(params["hres_beam_fwhms"])
    Bhres = [maps.gauss_beam(modlmap, f) for f in hres_fwhms]
    n_hres = len(Bhres)
    if n_hres not in (1, 2):
        raise ValueError(f"hres_beam_fwhms must have length 1 or 2, got {n_hres}")

    # Theory
    theory = cosmology.default_theory()
    # Match recon_sim.py / stack.py convention: lensed CMB in uC_T_T.
    ucltt = theory.lCl("TT", modlmap)

    # Filter masks
    xmask = maps.mask_kspace(shape, wcs,
                             lmin=params["grad_lmin"], lmax=params["grad_lmax"])
    ymask = maps.mask_kspace(shape, wcs,
                             lmin=params["hres_lmin"], lmax=params["hres_lmax"],
                             lxcut=params["hres_lxcut"], lycut=params["hres_lycut"])
    kmask = maps.mask_kspace(shape, wcs,
                             lmin=params["kappa_lmin"], lmax=params["kappa_lmax"])

    # Total power spectra (theoretical, no foregrounds)
    n_plc_pix = (params["grad_noise_uK_arcmin"] * utils.arcmin) ** 2
    tclpp = ucltt + n_plc_pix / Bplc**2  # Planck total, beam-deconvolved
    n_hres_pix = (params["hres_noise_uK_arcmin"] * utils.arcmin) ** 2

    if n_hres == 2:
        p11 = ucltt * Bhres[0]**2 + n_hres_pix
        p22 = ucltt * Bhres[1]**2 + n_hres_pix
        p12 = ucltt * Bhres[0] * Bhres[1]
        # Analytic ILC residual var = 1 / (r^T C^-1 r); equals total
        # beam-deconvolved post-ILC power.
        sel = (modlmap >= params["hres_lmin"]) & (modlmap <= params["hres_lmax"])
        nells = int(sel.sum())
        cov = np.zeros((nells, 2, 2))
        cov[:, 0, 0] = p11[sel]; cov[:, 1, 1] = p22[sel]
        cov[:, 0, 1] = p12[sel]; cov[:, 1, 0] = p12[sel]
        rs = np.stack([Bhres[0][sel], Bhres[1][sel]]).swapaxes(0, 1)
        den = np.array([np.linalg.solve(cov[i], rs[i]) for i in range(nells)])
        tclaa = ucltt.copy()
        tclaa[sel] = 1.0 / np.einsum("ij,ij->i", rs, den)
    else:
        tclaa = ucltt + n_hres_pix / Bhres[0]**2

    # Inpainter
    grad_inpainter = None
    if params["inpaint_radius_arcmin"] > 0:
        grad_inpainter = Inpainter(
            fine_shape=shape, fine_wcs=wcs,
            hole_radius_arcmin=params["inpaint_radius_arcmin"],
            theory=theory,
            beam_fn=lambda x: maps.gauss_beam(x, params["grad_beam_fwhm"]),
            noise_uK_arcmin=params["grad_noise_uK_arcmin"],
            context_width_arcmin=params["inpaint_context_arcmin"],
            pcov_res_arcmin=params["inpaint_pcov_res_arcmin"],
        )

    # NFW kappa + deflection
    kappa_true = nfw_kappa_map(shape, wcs, params["mass"],
                               params["redshift"], params["concentration"])
    grad_phi = olensing.alpha_from_kappa(kappa_true)
    ktrue_k =    enmap.fft(enmap.enmap(kappa_true * taper_arr, wcs),normalize="phys") * kmask
    
    if rank == 0:
        print(f"[cal] NFW: M={params['mass']:.2e}, z={params['redshift']}, "
              f"c={params['concentration']}, "
              f"peak_kappa={float(np.max(np.asarray(kappa_true))):.4f}")

    # Ell binning
    ell_edges = np.arange(params["kappa_lmin"],
                          params["kappa_lmax"] + params["ell_bin_width"],
                          params["ell_bin_width"])
    n_ell = len(ell_edges) - 1
    ell_centers = (ell_edges[:-1] + ell_edges[1:]) / 2.0
    ell_binner = stats.bin2D(modlmap, ell_edges)
    _, auto_power = ell_binner.bin(np.abs(ktrue_k) ** 2)

    # CMB generator (unlensed)
    ps_cmb = np.zeros((1, 1) + tuple(shape))
    ps_cmb[0, 0] = theory.uCl("TT", modlmap)
    mgen = maps.MapGen(shape=(1,) + tuple(shape), wcs=wcs, cov=ps_cmb)

    # Sim loop
    s = stats.Stats(comm)
    t0 = time.time()
    n_my = len(my_tasks)

    feed_dict_base = {
        "uC_T_T": ucltt,
        "tC_A_T_A_T": tclaa,
        "tC_P_T_P_T": tclpp,
        "tC_A_T_P_T": ucltt,
        "tC_P_T_A_T": ucltt,
    }

    for j, isim in enumerate(my_tasks):
        if rank == 0 and (j % max(1, n_my // 10) == 0):
            print(f"[cal] sim {j}/{n_my}  ({time.time()-t0:.0f}s)")
        seed = params["seed"] + int(isim)

        unlensed = mgen.get_map(seed=seed)[0]
        lensed = enlensing.lens_map(unlensed, grad_phi,
                                    order=5, border="cyclic")

        krecon_pair = []
        for cmb_in in (lensed, unlensed):

            # Gradient leg: beam-convolve, inpaint (untapered), taper, FFT, deconvolve.
            g_k = enmap.fft(cmb_in, normalize="phys") * Bplc
            g_map = enmap.enmap(enmap.ifft(g_k, normalize="phys").real, wcs)
            if grad_inpainter is not None:
                g_map = grad_inpainter.inpaint(g_map)
            k_grad = (enmap.fft(enmap.enmap(g_map * taper_arr, wcs),
                                normalize="phys") / Bplc)

            # High-res leg: per-freq beam-convolve, taper, FFT, ILC (or single).
            if n_hres == 1:
                h_k = enmap.fft(cmb_in, normalize="phys") * Bhres[0]
                h_map = enmap.ifft(h_k, normalize="phys").real
                k_hres = (enmap.fft(enmap.enmap(h_map * taper_arr, wcs),
                                    normalize="phys") / Bhres[0])
            else:
                kfreq = []
                for B in Bhres:
                    h_k = enmap.fft(cmb_in, normalize="phys") * B
                    h_map = enmap.ifft(h_k, normalize="phys").real
                    kfreq.append(np.asarray(
                        enmap.fft(enmap.enmap(h_map * taper_arr, wcs),
                                  normalize="phys")
                    ))
                k_hres, _ = ilc_two_channel(
                    modlmap, kfreq[0], kfreq[1],
                    p11, p22, p12, Bhres[0], Bhres[1],
                    params["hres_lmin"], params["hres_lmax"],
                )

            feed_dict = dict(feed_dict_base)
            feed_dict["X"] = k_grad
            feed_dict["Y"] = k_hres
            qe_obj = QE(shape, wcs, feed_dict,
                        estimator=params["qe_estimator"], XY="TT",
                        xmask=xmask, ymask=ymask,
                        field_names=["P", "A"], groups=None, kmask=kmask)
            rkmap = qe_obj.reconstruct(feed_dict,
                                       xname="X_l1", yname="Y_l2",
                                       physical_units=True)
            krecon_pair.append(rkmap)

        response_k = krecon_pair[0] - krecon_pair[1]
        if not np.all(np.isfinite(response_k)):
            continue
        cross_2d = np.real(response_k * np.conj(ktrue_k))
        _, cross_binned = ell_binner.bin(cross_2d)
        s.add_to_stats("cross", cross_binned)
        s.add_to_stats("nused", np.array([1.0]))

        # Real-space response stamp (for diagnostic plots only)
        response_real = np.asarray(
            enmap.ifft(enmap.enmap(response_k, wcs), normalize="phys").real
        )
        s.add_to_stack("response_real", response_real)

    s.get_stats()
    s.get_stacks()
    if rank != 0:
        return

    cross_mean = s.stats["cross"]["mean"]
    cross_err = s.stats["cross"]["errmean"]
    n_used = int(np.round(float(s.stats["nused"]["mean"][0]) * size))

    safe = auto_power > 1e-30
    R_L = np.ones(n_ell)
    R_L_err = np.zeros(n_ell)
    R_L[safe] = cross_mean[safe] / auto_power[safe]
    R_L_err[safe] = cross_err[safe] / auto_power[safe]

    # Save
    out_yaml = {k: params[k] for k in FILTER_KEYS}
    out_yaml["nfw"] = {
        "mass": float(params["mass"]),
        "redshift": float(params["redshift"]),
        "concentration": float(params["concentration"]),
        "peak_kappa": float(np.max(np.asarray(kappa_true))),
    }
    out_yaml["nsims_requested"] = int(params["nsims"])
    out_yaml["nsims_used"] = int(n_used)
    out_yaml["ell_bin_width"] = int(params["ell_bin_width"])

    yaml_path = os.path.join(args.output_dir, "cal.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False, default_flow_style=None)

    np.savez(
        os.path.join(args.output_dir, "cal.npz"),
        ell_edges=ell_edges, ell_centers=ell_centers,
        R_L=R_L, R_L_err=R_L_err,
        auto_power=auto_power, cross_mean=cross_mean,
    )
    print(f"[cal] Saved {yaml_path}")
    print(f"[cal] Saved cal.npz  ({n_used}/{params['nsims']} sims used)")
    print(f"[cal] mean R(L) in [{params['kappa_lmin']},{params['kappa_lmax']}]: "
          f"{float(np.mean(R_L[safe])):.4f}")

    # Diagnostic plots
    try:
        response_real_stack = np.asarray(s.stacks["response_real"])
        kappa_filt_real = np.asarray(
            enmap.ifft(enmap.enmap(ktrue_k, wcs), normalize="phys").real
        )
        modrmap_arcmin = enmap.modrmap(shape, wcs) * (180.0 * 60.0 / np.pi)
        _make_plots(
            args.output_dir, params, ell_centers, R_L, R_L_err,
            response_real_stack, kappa_filt_real, modrmap_arcmin,
            n_used,
        )
        print(f"[cal] Saved plots to {args.output_dir}")
    except Exception as e:
        print(f"[cal] WARNING: plot generation failed: {e}")


def _make_plots(out_dir, params, ell_centers, R_L, R_L_err,
                response_real, kappa_filt, modrmap_arcmin, n_used):
    """Three diagnostic plots written under out_dir.

    cal_R_L.png        : R(L) +/- err vs L, with unity reference
    cal_stamps.png     : zoomed 2D stamps of filtered true kappa vs mean
                         recovered response
    cal_profile.png    : 1D radial profiles of the same
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title_suffix = (
        f"hole={params['inpaint_radius_arcmin']}',  "
        f"M={params['mass']:.2e},  z={params['redshift']},  "
        f"nsims={n_used}"
    )

    # ---- R(L) ----
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(ell_centers, R_L, yerr=R_L_err, fmt="o-", color="#1f77b4",
                capsize=2, lw=1.5, ms=5, label=r"$R(L)$")
    ax.axhline(1.0, ls="--", color="k", alpha=0.5, lw=1, label="unity")
    safe = R_L_err < 1.0  # avoid garbage bins outside the kmask band
    if safe.any():
        mean_R = float(np.mean(R_L[safe]))
        ax.axhline(mean_R, ls=":", color="#d62728", lw=1.5,
                   label=fr"$\langle R \rangle = {mean_R:.3f}$")
    ax.set_xlabel(r"multipole $L$")
    ax.set_ylabel(r"$R(L)$")
    ax.set_title("QE response transfer function\n" + title_suffix,
                 fontsize=10)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cal_R_L.png"), dpi=130)
    plt.close(fig)

    # ---- 2D stamps (zoomed) ----
    n = response_real.shape[-1]
    half_arcmin = 20.0
    half_pix = int(half_arcmin / params["pix_arcmin"])
    cy = n // 2
    sl = (slice(cy - half_pix, cy + half_pix),
          slice(cy - half_pix, cy + half_pix))
    extent = [-half_arcmin, half_arcmin, -half_arcmin, half_arcmin]

    true_crop = kappa_filt[sl]
    resp_crop = response_real[sl]
    vmax = max(np.nanmax(true_crop), np.nanmax(resp_crop))
    vmin = min(np.nanmin(true_crop), np.nanmin(resp_crop))
    abs_max = max(abs(vmin), abs(vmax))
    norm_kwargs = dict(vmin=-abs_max, vmax=abs_max, cmap="RdBu_r")

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4),
                            gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax_, data, ttl in zip(
        axs,
        [true_crop, resp_crop, resp_crop - true_crop],
        ["filtered true kappa", "mean recovered response",
         "residual (recov - true)"],
    ):
        im = ax_.imshow(data, origin="lower", extent=extent, **norm_kwargs)
        ax_.set_title(ttl, fontsize=10)
        ax_.set_xlabel("arcmin")
        plt.colorbar(im, ax=ax_, fraction=0.046)
    axs[0].set_ylabel("arcmin")
    fig.suptitle(title_suffix, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out_dir, "cal_stamps.png"), dpi=130)
    plt.close(fig)

    # ---- 1D radial profile ----
    from orphics.stats import bin2D
    r_edges = np.arange(0.0, 15.01, 1.0)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2.0
    rbinner = bin2D(modrmap_arcmin, r_edges)
    _, prof_true = rbinner.bin(kappa_filt)
    _, prof_resp = rbinner.bin(response_real)

    fig, axs = plt.subplots(2, 1, figsize=(7.5, 6.5),
                            gridspec_kw={"height_ratios": [3, 1.4]},
                            sharex=True)
    axs[0].plot(r_centers, prof_true, "o-", color="k", lw=1.5, ms=5,
                label="filtered true kappa")
    axs[0].plot(r_centers, prof_resp, "s-", color="#1f77b4", lw=1.5, ms=5,
                label="mean recovered response")
    axs[0].axhline(0, ls="-", color="k", alpha=0.2, lw=1)
    axs[0].set_ylabel(r"$\kappa(\theta)$ profile")
    axs[0].set_title("Real-space response vs true kappa\n" + title_suffix,
                     fontsize=10)
    axs[0].legend(loc="best", framealpha=0.9)
    axs[0].grid(alpha=0.3)

    safe_r = np.abs(prof_true) > 1e-10
    ratio = np.full_like(prof_true, np.nan)
    ratio[safe_r] = prof_resp[safe_r] / prof_true[safe_r]
    axs[1].plot(r_centers, ratio, "o-", color="#d62728", lw=1.5, ms=5)
    axs[1].axhline(1.0, ls="--", color="k", alpha=0.5, lw=1)
    axs[1].set_xlabel(r"$\theta$ (arcmin)")
    axs[1].set_ylabel("recov / true")
    axs[1].set_ylim(0.0, 1.5)
    axs[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cal_profile.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
