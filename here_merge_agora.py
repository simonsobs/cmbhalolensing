import numpy as np
import pandas as pd
import time as t

start = t.time()

my_path = "/data5/sims/agora_sims/"

halo_path = "/data5/sims/agora_sims/full/halocat/"
lens_path = "/home3/eunseong/lens/"

"""
Slice number ranges from 4-200 and each integer increases by 25 Mpc/h 
In other words:
X=0 -> 0-25 Mpc/h 
X=1 -> 25-100 Mpc/h

With lensed coordinates:
Slice number ranges from 4-177 
columns are lensed RA / lensed DEC / magnification / real / image part of rotation
the length should match with the unlensed halo cat per slice 
so you can extract masses and other info by merging the two
"""

total_num = 174

halo_dfs = []
lens_dfs = []

for slice_num in range(4, 4+total_num):

    print(f"Processing slice {slice_num}", flush=True)

    npz = np.load(halo_path + f"agora_halolc_rot_{slice_num}_v050223.npz")

    halo_df = pd.DataFrame.from_dict(
        {item: npz[item] for item in npz.files}
    )

    halo_dfs.append(
        pd.DataFrame(
            halo_df,
            columns=[
                "totz",
                "totra",
                "totdec",
                "totm200", # x200 critial density [Msun/h]
                "totm500", # x500 critial density [Msun/h]
            ],
            dtype="float32",
        )
    )

    npz.close()


    npy = np.load(lens_path + f"agora_halos_lenra_lendec_mag_rotreal_rotimag_deflectnside16384_{slice_num}.npy")

    lens_dfs.append(
        pd.DataFrame(
            npy[:, 0:2],
            columns=["lenra", "lendec"],
            dtype="float32",
        )
    )

    del npy




print("Concatenating halo catalog...", flush=True)

halo_cat = pd.concat(halo_dfs, ignore_index=True)
print(halo_cat, flush=True)

halo_out = my_path + "agora_halo_forNEMO_partial.parquet.gzip"
halo_cat.to_parquet(halo_out, compression="gzip")

print(f"Saved: {halo_out}", flush=True)


print("Concatenating lensed halo catalog...", flush=True)

lens_cat = pd.concat(lens_dfs, ignore_index=True)
print(lens_cat, flush=True)

lens_out = my_path + "agora_lensedhalo_forNEMO.parquet.gzip"
lens_cat.to_parquet(lens_out, compression="gzip")

print(f"Saved: {lens_out}", flush=True)

elapsed = t.time() - start
print(f"\n::: reading and merging catalogues took {elapsed:.1f} seconds", flush=True)