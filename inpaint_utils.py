"""Gradient-leg inpainting helpers for cmbhalolensing.

Provides `GradientInpainter`, a class that encapsulates pixel-pixel
covariance inpainting on a coarse sub-stamp (60'x60' / 2' by default),
then upgrades and substitutes only the hole pixels back into the
fine-resolution input.  Built once per channel at the first cluster,
called per cluster after that.

Designed to be shared between `recon_sim.py` (sims) and `stack.py`
(data).  For data with per-frequency inpainting before ILC, instantiate
one `GradientInpainter` per frequency with different `beam_fn`.
"""

import numpy as np
from pixell import enmap, utils
from orphics import maps, pixcov


class Inpainter:
    """Pixcov-based inpainting on a coarse sub-stamp.

    One instance binds a fixed (stamp geometry, beam, white noise,
    hole radius).

    Parameters
    ----------
    fine_shape, fine_wcs
        Geometry of the fine-resolution stamp that will be passed to
        ``inpaint()``.  Used to build the fine-grid hole mask and the
        cutout selector.
    hole_radius_arcmin
        Radius of the circular hole.
    theory
        Object providing ``theory.lCl('TT', ell)``.
    beam_fn
        Callable ``beam_fn(ell) -> B(ell)``.  For Gaussian beams,
        ``lambda x: maps.gauss_beam(fwhm, x)``.  For per-frequency
        tabulated beams (data), pass the loader's callable directly.
    noise_uK_arcmin
        White-noise level for the diagonal noise covariance.
    context_width_arcmin
        Total width of the coarse sub-stamp.  Default 60'.  Must
        comfortably exceed 2 * hole_radius_arcmin.
    pcov_res_arcmin
        Resolution of the coarse pcov grid.  Default 2'.
        ``context_width_arcmin / pcov_res_arcmin`` must be an integer
        multiple of the fine pixel size.
    eigval_floor
        Floor on the conditional-covariance eigenvalues, relative to
        the largest.  Only used when ``inpaint(add_noise=True)``;
        harmless otherwise.
    """

    def __init__(
        self,
        fine_shape,
        fine_wcs,
        hole_radius_arcmin,
        theory,
        beam_fn,
        noise_uK_arcmin,
        context_width_arcmin=60.0,
        pcov_res_arcmin=2.0,
        eigval_floor=1e-6,
    ):
        fine_res = np.abs(fine_wcs.wcs.cdelt[1]) * 60.0  # arcmin
        self._dfac = int(round(pcov_res_arcmin / fine_res))
        self._fine_crop = int(round(context_width_arcmin / fine_res))
        coarse_n = self._fine_crop // self._dfac
        if coarse_n * self._dfac != self._fine_crop:
            raise ValueError(
                "context_width_arcmin must be an integer multiple of "
                "pcov_res_arcmin AND of fine_res "
                f"(fine_res={fine_res}', context={context_width_arcmin}', "
                f"pcov_res={pcov_res_arcmin}')"
            )
        if 2 * hole_radius_arcmin >= context_width_arcmin:
            raise ValueError(
                f"context_width_arcmin ({context_width_arcmin}) must exceed "
                f"2 * hole_radius_arcmin ({2 * hole_radius_arcmin})"
            )

        # Coarse pcov geometry built from a dummy fine map so its WCS
        # stays in the thumbnail's TAN projection.
        dummy = enmap.zeros(fine_shape, fine_wcs)
        cut0 = maps.crop_center(
            dummy, cropy=self._fine_crop, cropx=self._fine_crop, sel=False
        )
        coarse0 = enmap.downgrade(cut0, self._dfac)
        pshape, pwcs = coarse0.shape, coarse0.wcs

        # Diagonal white-noise + theory + beam -> pixcov on coarse grid
        ipsizemap = enmap.pixsizemap(pshape, pwcs)
        pivar = maps.ivar(pshape, pwcs, noise_uK_arcmin, ipsizemap=ipsizemap)
        pcov = pixcov.tpcov_from_ivar(coarse_n, pivar, theory.lCl, beam_fn)

        hole_radius = hole_radius_arcmin * utils.arcmin
        self._geo = pixcov.make_geometry(
            pshape,
            pwcs,
            hole_radius,
            n=coarse_n,
            deproject=True,
            iau=False,
            res=None,
            pcov=pcov,
            eigval_floor=eigval_floor,
        )

        self._cutout_sel = maps.crop_center(
            dummy, cropy=self._fine_crop, cropx=self._fine_crop, sel=True
        )
        modrmap = enmap.modrmap(fine_shape, fine_wcs)
        self._fine_hole_mask = modrmap < hole_radius
        self._box_hole_mask = self._fine_hole_mask[self._cutout_sel]
        self._fine_shape = tuple(fine_shape)

    def inpaint(self, fine_stamp, add_noise=False, rng=None):
        """Return a copy of ``fine_stamp`` with the central hole filled.

        Only the hole pixels are modified; surrounding pixels are left
        exactly as in the input (no downgrade/upgrade smearing).

        Parameters
        ----------
        fine_stamp : enmap
            Fine-resolution stamp.  Must match the (shape, wcs) the
            inpainter was built with.
        add_noise : bool
            If True, add a constrained-realisation random draw on top of
            the conditional mean.  Default False = mean-only.
        rng : np.random.Generator, optional
            RNG for the random draw.  Ignored when ``add_noise=False``.
        """
        if tuple(fine_stamp.shape) != self._fine_shape:
            raise ValueError(
                f"fine_stamp shape {tuple(fine_stamp.shape)} != "
                f"geometry shape {self._fine_shape}"
            )

        out = fine_stamp.copy()
        fine_cutout = maps.crop_center(
            out, cropy=self._fine_crop, cropx=self._fine_crop, sel=False
        ).copy()
        coarse = enmap.downgrade(fine_cutout, self._dfac)
        coarse_inp = pixcov.inpaint_stamp(
            coarse, self._geo, add_noise=add_noise, rng=rng
        )
        fine_inp = enmap.upgrade(coarse_inp, self._dfac)
        fine_cutout[self._box_hole_mask] = fine_inp[self._box_hole_mask]
        out[self._cutout_sel] = fine_cutout
        return out
