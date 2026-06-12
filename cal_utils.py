"""Load, validate, and apply QE response calibration products.

Used by ``recon_sim.py`` and ``stack.py`` when ``--calibration <dir>`` is
passed.  The directory must contain ``cal.npz`` and ``cal.yaml`` written
by ``cal_response.py``.
"""

import os

import numpy as np
import yaml


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


def load_calibration(cal_dir):
    """Return {'ell_centers', 'R_L', 'R_L_err', 'params', 'path'}."""
    npz_path = os.path.join(cal_dir, "cal.npz")
    yaml_path = os.path.join(cal_dir, "cal.yaml")
    if not (os.path.isfile(npz_path) and os.path.isfile(yaml_path)):
        raise FileNotFoundError(
            f"Calibration directory missing cal.npz or cal.yaml: {cal_dir}"
        )
    data = np.load(npz_path)
    with open(yaml_path) as f:
        params = yaml.safe_load(f)
    return {
        "ell_centers": np.asarray(data["ell_centers"]),
        "R_L": np.asarray(data["R_L"]),
        "R_L_err": np.asarray(data["R_L_err"]),
        "params": params,
        "path": os.path.abspath(cal_dir),
    }


def _equal(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        try:
            a = [float(v) for v in a]
            b = [float(v) for v in b]
        except (TypeError, ValueError):
            return list(a) == list(b)
        if len(a) != len(b):
            return False
        return all(np.isclose(x, y, rtol=1e-6, atol=1e-9) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return np.isclose(float(a), float(b), rtol=1e-6, atol=1e-9)
    return a == b


def check_compatibility(cal_params, **run):
    """Raise ValueError if any filter parameter in run mismatches calibration."""
    mismatches = []
    for key in FILTER_KEYS:
        if key not in run:
            continue
        a = cal_params.get(key)
        b = run[key]
        if not _equal(a, b):
            mismatches.append(f"  {key}: calibration={a!r}, run={b!r}")
    if mismatches:
        raise ValueError(
            "Calibration is incompatible with this run:\n" + "\n".join(mismatches)
        )


def build_R_2d(R_L, ell_centers, modlmap, fill_value=1.0):
    """Linear interpolation of R(L) onto the 2D Fourier grid.

    Outside [ell_centers[0], ell_centers[-1]] -> ``fill_value`` (default 1.0,
    a no-op when dividing).  Bands killed by kmask are unaffected.
    """
    return np.interp(
        np.asarray(modlmap),
        np.asarray(ell_centers),
        np.asarray(R_L),
        left=fill_value,
        right=fill_value,
    )


def apply_R_2d(rkmap, R_2d, eps=1e-3):
    """Safe division: rkmap / R_2d, zero where |R_2d| < eps.

    Preserves the enmap subclass (and wcs) of ``rkmap`` by avoiding
    np.where, which strips the subclass.  Bands where |R_2d| < eps get
    R_safe = inf, so rkmap / R_safe = 0 there.
    """
    R_safe = np.where(np.abs(R_2d) > eps, R_2d, np.inf)
    return rkmap / R_safe
