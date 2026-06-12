import numpy as np
import pandas as pd
from astropy.table import Table


halo_path = "/data5/sims/agora_sims/agora_halo_forNEMO_partial.parquet.gzip"
lens_path = "/data5/sims/agora_sims/agora_lensedhalo_forNEMO.parquet.gzip"
output_path = "data/agora_halo_1e13cut.fits"
# output_path = "data/agora_halo_5e13cut.fits"


H0 = 67.77
min_mass = 1e13  # Msun

print("Reading Agora halo catalog...")

halo_cat = pd.read_parquet(halo_path)
lens_cat = pd.read_parquet(lens_path)


zs = halo_cat["totz"].to_numpy()
RAs = halo_cat["totra"].to_numpy()
decs = halo_cat["totdec"].to_numpy()

M200c = halo_cat["totm200"].to_numpy()  # Msun/h
M500c = halo_cat["totm500"].to_numpy()  # Msun/h

RAs_lens = lens_cat["lenra"].to_numpy()
decs_lens = lens_cat["lendec"].to_numpy()


# Unit conversion (Msun/h -> Msun)
M200c = M200c / (0.01 * H0)
M500c = M500c / (0.01 * H0)

print("Initial sample size:", len(zs))


mask = M200c > min_mass

RAs = RAs[mask]
decs = decs[mask]
zs = zs[mask]
M200c = M200c[mask]
M500c = M500c[mask]

RAs_lens = RAs_lens[mask]
decs_lens = decs_lens[mask]

print("Sample size after mass cut:", len(zs), min_mass/1e14)


out_tab = Table()

out_tab["RADeg_unl"] = RAs
out_tab["decDeg_unl"] = decs
out_tab["RADeg"] = RAs_lens
out_tab["decDeg"] = decs_lens
out_tab["redshift"] = zs
out_tab["true_M200c"] = M200c / 1e14
out_tab["true_M500c"] = M500c / 1e14


print(f"Writing catalogue to {output_path}")
out_tab.write(output_path, overwrite=True)

print("Done.")


