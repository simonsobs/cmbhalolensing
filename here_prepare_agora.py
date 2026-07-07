import numpy as np
from pixell import enmap, utils, bunch
import utils as cutils
from orphics import io, catalogs
from astropy.io import fits


paths = cutils.paths
sim_paths = bunch.Bunch(io.config_from_yaml("input/sim_data.yml"))

sim_version = "nemo-sim-kit61"
sim_type = "agora"

output_path = f"{sim_paths.cat_path}/{sim_version}/"
io.mkdir(output_path)
print(" ::: saving to", output_path)



def load_cat(file_name):
    print(" ::: input tSZ cat:", file_name)
    with fits.open(file_name) as hdu:
        data = hdu[1].data

    catalog = {
        "zs": data["redshift"],
        "ras": data["RADeg"],
        "decs": data["decDeg"],
        "true_M200c": data["true_M200c"],
        "true_M500c": data["true_M500c"],
        "M200c": data["M200c"],
        "snrs": data["SNR"],
    }

    io.save_cols(
        output_path + f"{sim_type}_tsz_true.txt",
        (catalog["ras"], catalog["decs"], catalog["zs"], catalog["true_M200c"]),
    )

    print(f"\nN={len(catalog['zs'])} [min SNR={catalog['snrs'].min():.1f}]")
    print(f"mean true M200c={catalog['true_M200c'].mean():.2f}")
    print(f"mean true M500c={catalog['true_M500c'].mean():.2f}")
    print(f"mean true z={catalog['zs'].mean():.2f}\n")

    return catalog



def generate_random_cat(n_per_obj, ras, choice="halo"):

    seed = 100
    np.random.seed(seed)

    N = n_per_obj * len(ras)
    print(f" ::: total number of random objects = {n_per_obj} x {len(ras)} = {N} [{choice}]")

    if choice == "halo":

        dec_min=-np.pi/2
        dec_max=np.pi/2
        ra_min=0
        ra_max=2.*np.pi

        coords = np.zeros((2,N))
        dmin = np.cos(np.pi/2 - dec_min)
        dmax = np.cos(np.pi/2 - dec_max)
        coords[0,:] = (np.pi/2 - np.arccos(np.random.uniform(dmin, dmax, N))) / utils.degree
        coords[1,:] = np.random.uniform(ra_min, ra_max, N) / utils.degree

    elif choice == "tsz":

        mask_path = f"{paths.data}mask/websky_act_mask.fits"
        mask = enmap.read_map(mask_path)
        shape, wcs = mask.shape, mask.wcs

        valid_pix = mask > 0
        Npix = valid_pix.sum()

        inds = np.random.choice(Npix, size=N, replace=False)

        pixs = enmap.pixmap(shape, wcs)
        coords = mask.pix2sky(pixs[:, valid_pix][:, inds], safe=False) / utils.degree

        # cmapper = catalogs.CatMapper(coords[1], coords[0], shape=shape, wcs=wcs)
        # io.hplot(
        #     enmap.downgrade(cmapper.counts, 16),
        #     f"{paths.data}mask/randcounts_{choice}",
        #     mask=0,
        # )

    io.save_cols(
        output_path + f"{sim_type}_{choice}_randoms.txt",
        (coords[1], coords[0]), # ra, dec
    )



def apply_snr_cut(catalog, threshold, label):

    keep = catalog["snrs"] > threshold

    zs = catalog["zs"][keep]
    ras = catalog["ras"][keep]
    decs = catalog["decs"][keep]
    m200 = catalog["true_M200c"][keep]
    m500 = catalog["true_M500c"][keep]
    snrs = catalog["snrs"][keep]

    print(f"\nSNR > {threshold}")
    print(f"N={len(zs)} [min SNR={snrs.min():.1f}]")
    print(f"mean true M200c={m200.mean():.2f}")
    print(f"mean true M500c={m500.mean():.2f}")
    print(f"mean z={zs.mean():.2f}")

    io.save_cols(
        output_path + f"{sim_type}_tsz_{label}_true.txt",
        (ras, decs, zs, m200),
    )

    return zs, m200


def resample_halo(tsz_zs, tsz_masses, snrcut):

    halo_file = "/home3/eunseong/cmbhalolensing/data/agora_halo_5e13cut.fits"
    hdu = fits.open(halo_file)
    data = hdu[1].data

    ras = data["RADeg"]
    decs = data["decDeg"]
    masses = data["true_M200c"]
    M500c = data["true_M500c"]
    zs = data["redshift"]

    print("\n ::: resampling halo sample!")
    print(" ::: number of clusters in tsz cat =", len(tsz_masses))

    num_bins = 30
    mbins = np.linspace(tsz_masses.min(), tsz_masses.max(), num_bins)
    zbins = np.linspace(tsz_zs.min(), tsz_zs.max(), num_bins)

    hist, mbin_edges, zbin_edges = np.histogram2d(
        tsz_masses, tsz_zs, bins=[mbins, zbins]
    )

    seed = 101
    rng = np.random.default_rng(seed)

    out = {k: [] for k in ["ras", "decs", "masses", "M500c", "zs"]}

    for i in range(len(mbin_edges) - 1):
        for j in range(len(zbin_edges) - 1):
            mmin, mmax = mbin_edges[i], mbin_edges[i + 1]
            zmin, zmax = zbin_edges[j], zbin_edges[j + 1]

            mask = (
                (masses > mmin)
                & (masses <= mmax)
                & (zs > zmin)
                & (zs <= zmax)
            )

            if hist[i, j] == 0 or not np.any(mask):
                continue

            idx_pool = np.where(mask)[0]
            n_target = int(hist[i, j])

            replace = len(idx_pool) < n_target
            inds = rng.choice(idx_pool, size=n_target, replace=replace)

            out["ras"].append(ras[inds])
            out["decs"].append(decs[inds])
            out["masses"].append(masses[inds])
            out["M500c"].append(M500c[inds])
            out["zs"].append(zs[inds])

    ras = np.concatenate(out["ras"])
    decs = np.concatenate(out["decs"])
    masses = np.concatenate(out["masses"])
    M500c = np.concatenate(out["M500c"])
    zs = np.concatenate(out["zs"])

    print(f" ::: z range = {zs.min():.2f}–{zs.max():.2f} (mean={zs.mean():.2f})")
    print(f" ::: M200c range = {masses.min():.2f}–{masses.max():.2f} (mean={masses.mean():.2f})")
    print(f" ::: M500c range = {M500c.min():.2f}–{M500c.max():.2f} (mean={M500c.mean():.2f})")
    print(f" ::: number of halos after resampling = {len(ras)}")

    if snrcut == "snr4":
        io.save_cols(
            output_path + f"{sim_type}_halo_true.txt",
            (ras, decs, zs, masses),       
        )
    else:
        io.save_cols(
            output_path + f"{sim_type}_halo_{snrcut}_true.txt",
            (ras, decs, zs, masses),
        )




file_name = "data/20260320_cmb_tsz_ksz_mass_true.fits" # nsk61
catalog = load_cat(file_name)

generate_random_cat(n_per_obj=100, ras=catalog["ras"], choice="tsz")
generate_random_cat(n_per_obj=100, ras=catalog["ras"], choice="halo")


z_snr5p5, m_snr5p5 = apply_snr_cut(catalog, 5.5, "snr5p5")
z_snr7, m_snr7 = apply_snr_cut(catalog, 7.0, "snr7")


resample_halo(catalog["zs"], catalog["true_M200c"], "snr4")
resample_halo(z_snr5p5, m_snr5p5, "snr5p5")
resample_halo(z_snr7, m_snr7, "snr7")