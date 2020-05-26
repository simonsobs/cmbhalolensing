CMB Halo Lensing Reconstruction and Stacking
============================================

This is a set of pipeline scripts for reconstructing stacked images
and profiles of the CMB lensing convergence due to galaxies, groups and
clusters. It includes some utilities for quick modeling
and interpretation of the results as well. Interfaces to catalogs of interest
can be added through the function ``utils.catalog_interface``; feel free to
provide a PR for your use case.

Dependencies
------------

There's some work to do to trim out some of these dependencies.

* pixell_
* symlens_
* orphics_ (git clone and install with pip install -e . --user)
* hmvec_ (possbly temporary, for NFW fits ; git clone and install with pip install -e . --user)
* enlib_ (temporary, for benchmarking using enlib.bench; just git clone and add
  to PYTHONPATH)
* healpy, numpy, scipy, matplotlib


Usage
-----

The main scripts stacky.py, post.py and make_lensed_sims.py are partially
documented through their command line arguments. Just invoke them with the ``-h``
flag for more information.

* Copy input/paths.yml to input/paths_local.yml and edit with your local paths. Do not attempt to add the latter file to the git tree.
* **stack.py** : to ILC/combine, reconstruct and stack on either catalogs or randoms.
* **post.py** : to post-process stacks and do simple fits, calculate SNR
* **make_lensed_sims.py** : to make and save lensed sims for sim injection tests
* **sim.py** : simple sim tests (no window or mean-field subtraction)
* **utils.py** : common utilities

.. _pixell: https://github.com/simonsobs/pixell/
.. _symlens: https://github.com/simonsobs/symlens/
.. _hmvec: https://github.com/simonsobs/hmvec/
.. _orphics: https://github.com/msyriac/orphics/
.. _enlib: https://github.com/amaurea/enlib/
