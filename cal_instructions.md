# QE Response Calibration

`cal_response.py` measures a Fourier-space transfer function
`R(L) = <Re[(QE(lensed) - QE(unlensed)) · ktrue*]> / <|ktrue|^2>`
on paired NFW-lensed Gaussian CMB sims.  The same script works for both
`recon_sim.py` (sims) and `stack.py` (data) because all filter
parameters are read from a YAML preset / CLI args.

## 1. Generate a calibration

Two presets have been added:

```bash
# Sim-side calibration (mirrors recon_sim.py constants)
python cal_response.py --config input/cal_recon_sim.yml \
    --output-dir <cal_root>/cal_sim_inp4 \
    --inpaint-radius-arcmin 6.0

# Same, no inpainting (for comparison)
python cal_response.py --config input/cal_recon_sim.yml \
    --output-dir <cal_root>/cal_sim_nohole

# Data-side calibration (mirrors input/defaults.yml + stack.py)
python cal_response.py --config input/cal_stack.yml \
    --output-dir <cal_root>/cal_data_inp4 \
    --inpaint-radius-arcmin 6.0
```

MPI parallel (recommended for ≥500 sims):

```bash
OMP_NUM_THREADS=2 mpirun -n 16 python cal_response.py \
    --config input/cal_recon_sim.yml \
    --output-dir <cal_root>/cal_sim_inp4 \
    --inpaint-radius-arcmin 6.0 --nsims 2000
```

Any preset key can be overridden on the CLI (flag = key with `_` -> `-`,
e.g. `--hres-lmax 4000`).  Common overrides: `--inpaint-radius-arcmin`,
`--mass`, `--nsims`, `--debug` (10 sims).

### Output

```
<cal_root>/cal_*_inp4/
    cal.npz     # ell_edges, ell_centers, R_L, R_L_err, auto_power, cross_mean
    cal.yaml    # full filter parameter set + NFW metadata
```

## 2. Apply the calibration

Pass `--calibration <cal_dir>` to `recon_sim.py` or `stack.py`.  The
loader validates that every parameter in `FILTER_KEYS` matches the
run's actual settings; mismatches raise a `ValueError`.

```bash
# Sim
python recon_sim.py myrun agora halo --cmb-ksz --inpaint 6.0 \
    --calibration <cal_root>/cal_sim_inp4

# Mean-field: SAME calibration must be passed
python recon_sim.py myrun agora halo --cmb-ksz --inpaint 6.0 \
    --calibration <cal_root>/cal_sim_inp4 --is-meanfield

# Data
python stack.py ... --inpaint --calibration <cal_root>/cal_data_inp4
python stack.py ... --inpaint --calibration <cal_root>/cal_data_inp4 \
    --is-meanfield
```

The integration divides `rkmap` by `R_2d` immediately after the symlens
QE reconstruct.  Because this is a linear operation, you can subtract the
mean-field at radial-profile time as before — provided both runs used
the same `--calibration` directory.

## 3. Note

- **`stack.py` and `recon_sim.py` have different settings.**  Do *not*
  reuse the same calibration product across them.  The validator will
  refuse, but it's faster to use the right preset from the start.
- **Foregrounds are intentionally absent in the sims.**  `R(L)` depends
  on the QE filter weights only, not on the data.  Adding foregrounds
  would add scatter without changing the mean response.
- **Mean-field with calibration.**  Always pass `--calibration` to
  *both* the cluster run and the `--is-meanfield` run, or to neither.
- **`R(L)` is mildly mass-dependent.**  Calibrate at the median mass of
  the sample (default `--mass 3.5e14`).
