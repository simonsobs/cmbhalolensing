from astropy.cosmology import FlatLambdaCDM
import numpy as np
import time as t
from itertools import chain
from functools import reduce

z_min = 0.4
z_max = 0.7
M_min = 1.e13 # Msun
M_max = 1.e14 # Msun
slice_width = 25 # Mpc/h

halos_dir = '/data5/sims/agora_sims/halolc/'

cosmo = FlatLambdaCDM(H0=67.77, Om0=0.307) # agora fiducial cosmology

h = cosmo.h
d_min = cosmo.comoving_distance(z_min)*h # Mpc/h
d_max = cosmo.comoving_distance(z_max)*h # Mpc/h

min_slice = int(np.floor(d_min.value/25))
max_slice = int(np.floor(d_max.value/25))

print(d_min, min_slice)
print(d_max, max_slice)



cmass_like = {'ra':[],
              'dec':[],
              'z':[],
              'M200c':[]

}
i=1
# num = 94
# test_slice = f"haloslc_rot_{num}_v050223.npz"
# with np.load(halos_dir + test_slice) as data:
#     print(f"--> zmin: {min(data['totz'])}, zmax:{max(data['totz'])}")
#     print(f"--> luminosity, min: {cosmo.luminosity_distance(min(data['totz']))*h}, max: {cosmo.luminosity_distance(max(data['totz']))*h}")
#     print(f"--> comoving, min: {cosmo.comoving_distance(min(data['totz']))*h}, max: {cosmo.comoving_distance(max(data['totz']))*h}")
#     print(f"--> angular, min: {cosmo.angular_diameter_distance(min(data['totz']))*h}, max: {cosmo.angular_diameter_distance(max(data['totz']))*h}")
#     # print(f"--> proper, min: {cosmo.redshift_distance(min(data['totz']))*h}, max: {cosmo.redshift_distance(max(data['totz']))*h}")

#     print(f"--> labeled, min: {num*25}, max: {(num+1)*25}")
for slice in np.arange(min_slice, max_slice+1):
    slice_path = f"haloslc_rot_{slice}_v050223.npz"
    print(f"loading slice {i} of {max_slice-min_slice+1}")
    start = t.time()
    with np.load(halos_dir + slice_path) as data:
        # dec_mask = np.logical_and(data['totdec']<10, -10 <data['totdec'])
        z_mask = np.logical_and(z_min <= data['totz'], data['totz'] <= z_max)
        mass_mask = np.logical_and(M_min*h <= data['totm200'], data['totm200'] <= M_max*h) # Msun/h
        cmass_mask = np.where(np.logical_and(z_mask, mass_mask))[0]
        print(f"--> zmin:{min(data['totz'])}, zmax:{max(data['totz'])}")
        print(f"--> mask lens {len(np.where(z_mask)[0])}, {len(np.where(mass_mask)[0])}")
    
        cmass_like['ra'].extend(data['totra'][cmass_mask])
        cmass_like['dec'].extend(data['totdec'][cmass_mask])
        cmass_like['z'].extend(data['totz'][cmass_mask])
        cmass_like['M200c'].extend(data['totm200'][cmass_mask])
        end = t.time()
        print(f"--> slice {i} read in {end-start:.2}s, contains {len(data['totra'][cmass_mask])}")
    i+=1




# z_range = np.logical_and(z_min <= zs, zs <= z_max)
# mass_range = np.logical_and(M_min/h <= M200cs, M200cs <= M_max/h)
# cmass_range = np.where(np.logical_and(z_range, mass_range))[0]

for key in cmass_like:
    cmass_like[key] = np.asarray(cmass_like[key])

np.save("agora_cmasslike.npy", cmass_like, allow_pickle=True)
