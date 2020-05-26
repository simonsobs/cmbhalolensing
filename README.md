CMB Halo Lensing Reconstruction and Stacking
============================================

* Copy input/paths.yml to input/paths_local.yml and edit with your local paths. Do not attempt to add the latter file to the git tree.
* **stack.py** : to ILC/combine, reconstruct and stack on either catalogs or randoms.
* **post.py** : to post-process stacks and do simple fits, calculate SNR
* **make_lensed_sims.py** : to make and save lensed sims for sim injection tests
* **sim.py** : simple sim tests (no window or mean-field subtraction)
* **utils.py** : common utilities