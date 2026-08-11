# Solution Separation vs. GLR testing — reproduction code

Code accompanying the chapter comparing the **solution separation (SS)** test
with the **generalized likelihood ratio (GLR / `T_q`)** test, in terms of the
probability of positioning failure `P_F` (integrity risk).

The repository contains two independent experiments:

| Directory | Experiment | Where it runs |
|---|---|---|
| `Code_binary_example/` | Binary hypothesis test (one `H0` vs. one `Ha`), `n = 1`, `q = 2`, three `Ca`-matrix geometries | Locally, on a laptop/desktop |
| `Code_GNSS_example/` | GNSS positioning example, `m = 11` satellites, `k = 66` fault hypotheses | Snellius HPC (SLURM), post-processed locally |

Both follow the same two-stage pattern: a **simulation stage** that writes plain
`.txt` result files to disk, and a **plotting stage** that reads those files back
and produces the figures and table numbers. The two stages are deliberately
decoupled so that figures can be restyled without re-running any Monte Carlo.

---

## Requirements

* Python 3.9 or newer
* `numpy`, `scipy`, `matplotlib`
* A working LaTeX installation — several plotting scripts set
  `text.usetex = True` and load `amsmath` / `amssymb`. Set it to `False` if you
  do not have LaTeX available; only the label rendering changes.
* For the GNSS example: a SLURM cluster. The submit scripts target Snellius and
  load the `2023` toolchain modules.

No installation step is needed; every script is standalone and imports nothing
from the repository itself.

---

## Repository layout

```
.
├── LICENSE
├── Code_binary_example/
│   ├── Power_SS_vs_Tq_and_IR_binary_case.py       <- simulation (detection only)
│   ├── IR_SS_vs_Tq_Detection_and_identification.txt  <- DIA variant (see notes)
│   ├── plot_IR_binary_detection_only.py           <- 3-column comparison figure
│   ├── plot_IR_binary_detection_and_ident.py      <- 3-column comparison figure
│   ├── plot_IR_binary_det_only_per_ca.py          <- one 2x2 figure per Ca
│   └── plot_IR_binary_det_and_ident_per_ca.py     <- one 2x2 figure per Ca
│
└── Code_GNSS_example/
    ├── plot_results_GNSS_ex_CH5_from_Snellius.py       <- table + (b1,b2) heat maps
    ├── plot_results_GNSS_ex_CH5_from_Snellius_line.py  <- chapter P_F vs b figures
    │
    ├── Code_1D_analysis/          <- bias swept along a LINE  -> chapter figures
    │   ├── det_only/
    │   │   ├── main_code/gnss_simulate_hypothesis_det_only_line.py
    │   │   ├── submit_GNSS_ex_per_hypt.bash
    │   │   └── submit_jobs.sh
    │   └── det_and_ident/
    │       ├── main_code/gnss_simulate_hypothesis_det_and_ident_line.py
    │       ├── submit_GNSS_ex_per_hypt.bash
    │       └── submit_jobs.sh
    │
    └── Code/                      <- bias swept over a 2-D GRID -> chapter table
        ├── det_only/with_sim_var/
        │   ├── main_code/gnss_simulate_hypothesis_det_only_with_sim_var.py
        │   ├── submit_GNSS_ex_per_hypt.bash
        │   └── submit_jobs.sh
        └── det_and_ident/with_sim_var/
            ├── main_code/gnss_simulate_hypothesis_det_and_ident_with_sim_var.py
            ├── submit_GNSS_ex_per_hypt.bash
            └── submit_jobs.sh
```

---

## 1. Binary hypothesis example (run locally)

A minimal model with `m = 5` observations, `n = 1` unknown, `A = [1,1,1,1,1]`, unit
covariance `Qyy = I`, and a single alternative hypothesis with `q = 2` bias
parameters. The integrity risk is computed over a 2-D grid of bias values
`(b1, b2)`, for both tests, and repeated over several Monte Carlo replications
so that the simulation uncertainty can be reported alongside the mean.

### 1.1 The three `Ca`-matrix geometries

The variable `type_of_ci_matrix` selects the fault geometry. Three are used in
the chapter, each with its own bias grid and alert limit:

| `type_of_ci_matrix` | Description | Bias grid | `Bx` |
|---|---|---|---|
| `"outlier"` | Two independent outliers; `Ca` = two unit vectors | `b1, b2 ∈ [-5, 5]`, 100 × 100 | 1.5 |
| `"linearly_dependent"` | Two almost parallel columns (`c2 = c1 + ε`, `ε = 1e-1`) | `b1 ∈ [-25, 0]`, `b2 ∈ [0, 20]`, 200 × 200 | 1.5 |
| `"almost_in_A"` | Columns nearly inside the column space of `A` | `b1 ∈ [-150, 50]`, `b2 ∈ [0, 300]`, 200 × 200 | 15.0 |



### 1.2 Simulation stage

**`Power_SS_vs_Tq_and_IR_binary_case.py`** — the detection-only generator.

Key parameters, all near the top of the `__main__` block:

```python
np.random.seed(40)
m, n, q   = 5, 1, 2
alpha     = 0.05          # total false-alarm probability
PH0, PHi  = 0.95, 0.05    # prior hypothesis probabilities
nr_sims   = 20            # Monte Carlo replications
nr_samples = int(1e5)     # samples per replication
```

Main routines:

* `stable_inverse`, `plusmat`, `perpmat` — numerically guarded linear algebra
  helpers (Cholesky, falling back to LU and then pseudo-inverse).
* `calc_IR(...)` — the Monte Carlo core. For one bias vector it draws
  `nr_samples` observation vectors, forms both test statistics, and returns the
  integrity risk, the detection power `gamma`, and the risks conditioned on `H0`
  and on `Ha` separately, for each of the `nr_sims` replications.
* The `__main__` block loops `calc_IR` over the flattened `(b1, b2)` grid and
  writes the results.

Run it once per `Ca` geometry by editing `type_of_ci_matrix` and re-running.

### 1.3 Output layout

The base directory is set by `savingDir` in the `__main__` block. Under it:

```
Data_binary_example/
└── det_only/                       (or det_and_ident/)
    ├── 2 Outliers/Data/                       <- "outlier"
    ├── Ci-almost-lin-dependent/Data/          <- "linearly_dependent"
    └── Ci-columns-almost-in-A/AL=15/Data/     <- "almost_in_A"
```

For the detection-and-identification results an extra `Det+Ident/` level is
inserted before `Data/`, e.g. `2 Outliers/Det+Ident/Data/`.

Each `Data/` folder holds comma-delimited text files:

| File | Shape | Contents |
|---|---|---|
| `b1grid.txt`, `b2grid.txt` | `(nb1,)`, `(nb2,)` | Bias axes |
| `IR_grid_SS.txt`, `IR_grid_Tq.txt` | `(nb1*nb2, 2)` | Column 0 = mean `P_F`; column 1 = std across replications |
| `gamma_grid_SS.txt`, `gamma_grid_Tq.txt` | `(nb1*nb2,)` | Detection power |
| `SS_stats.txt`, `Tq_stats.txt` | `(nb1*nb2,)` | Test-statistic diagnostics |

**Important:** column 1 of `IR_grid_*` is the *raw* standard deviation across
the `nr_sims` replications, **not** the standard error of the mean. The
plotting scripts convert it with `bool_convert_sigma = True`, which divides by
`sqrt(nr_sims)`. If you change `nr_sims` in the simulation, change it in the
plotting scripts too — they do not read it from disk.

### 1.4 Plotting stage

Four scripts, all reading the layout above:

| Script | Produces |
|---|---|
| `plot_IR_binary_detection_only.py` | One wide figure with the three `Ca` geometries side by side (SS top row, GLR bottom row), plus zoomed and flattened difference panels |
| `plot_IR_binary_detection_and_ident.py` | Same, for the detection-and-identification case |
| `plot_IR_binary_det_only_per_ca.py` | One 2×2 figure *per* `Ca`: left column the mean `P_F`, right column the relative simulation std `sigma / P_F`; SS on top, GLR below |
| `plot_IR_binary_det_and_ident_per_ca.py` | Same, for detection and identification |

The `per_ca` scripts rebuild the geometry and the thresholds internally (so the
`T_q` contour lines can be overlaid) and then load the simulated grids. Both
`per_ca` scripts write `PPF_mean_and_simstd_grid_<type_of_ci_matrix>.{png,pdf}`
to `figSavingDir`; the combined scripts write `PPF_grid`,
`PPF_sim_var_on_grid`, `grid_zoomed_in_flattened`, `difference_flattened`, and
`difference_flattened_separate`.

Note that `nr_sims = 20` for detection-only and `nr_sims = 5` for detection and
identification. This is intentional and matches the runs that produced the data.

### 1.5 Workflow summary

```
for each Ca in {outlier, linearly_dependent, almost_in_A}:
    edit type_of_ci_matrix, run Power_SS_vs_Tq_and_IR_binary_case.py
then:
    run plot_IR_binary_detection_only.py         (comparison figure)
    run plot_IR_binary_det_only_per_ca.py        (per-Ca figures)
```

---

## 2. GNSS example (run on Snellius)

A dual-constellation GPS + Galileo snapshot, taken at 12:00 UTC in Delft on
1 July 2025. Of the 22 visible satellites a fixed subset of 11 is used
(`keep_idx = [2, 3, 4, 6, 10, 13, 15, 16, 17, 18, 19]`, giving 5 GPS and
6 Galileo). With `n = 5` unknowns (three coordinates plus two clock offsets),
the redundancy is `r = 6` and the hypothesis set is

```
k = m + m(m-1)/2 = 11 + 55 = 66
```

covering all single-satellite (`q = 1`) and satellite-pair (`q = 2`) faults.
Integrity is evaluated on the **vertical (UP) coordinate**, index `n_UP = 2`,
against alert limits `AL_Bx ∈ {1.5, 2.5, 3.0}` m.

### 2.1 The two simulation campaigns

The bias vector under a fault hypothesis can be swept in two ways, and the
chapter uses both:

* **`Code_1D_analysis/` — bias along a line.** For `q = 1` the bias is `b`; for
  `q = 2` it is `[b, b]`. A single scalar `b` therefore parameterizes every
  hypothesis, which is what makes the `P_F` vs. `b` curves comparable across
  hypotheses. **This campaign produces the two chapter figures.**
* **`Code/` — bias over a 2-D grid.** For `q = 2` the full `(b1, b2)` plane is
  swept, so the maximum of `P_F` is found over the whole plane rather than along
  the diagonal. **This campaign produces the max-`P_F` numbers in the table** at
  the end of the GNSS section.

Each campaign is further split into `det_only/` and `det_and_ident/`:

* **Detection only.** A failure occurs when the *global* least-squares estimate
  leaves the safety region `Bx` **and** the test fails to reject `H0`. The
  probability that the estimate is unsafe is computed **analytically** — under
  `Hi(b)` the UP-bias of the global estimator is deterministic — so only the
  detection power `gamma` is Monte-Carlo'd. `AL_Bx` therefore enters only through
  closed-form expressions, and one Monte Carlo run serves all three alert limits.
* **Detection and identification (DIA).** After rejection, the hypothesis with
  the largest statistic is identified and the corresponding *adapted* estimator
  is used. A failure is scored empirically by counting samples for which the
  identified estimate falls outside `Bx`. `AL_Bx` enters through the counting,
  done for all three limits in one pass by broadcasting.

### 2.2 Simulation scripts and their routines

All four simulators share the same skeleton:

| Routine | Purpose |
|---|---|
| `stable_inverse`, `plusmat`, `perpmat` | Guarded linear algebra |
| `sigma_user_n`, `compute_design_matrix`, `get_covariance_matrix` | GNSS error model: elevation-dependent user noise, tropospheric residual, URA, and the carrier-phase-smoothed pseudorange covariance |
| `setup_geometry(...)` | Builds `A`, `Qyy`, whitens so that `Qyy = I`, forms the parity basis `B`, enumerates all 66 hypotheses with their projectors `Pcti`, and computes the prior probabilities `PH0` and `PHi_list` from `psat_GPS = 1e-3`, `psat_GAL = 3e-3`. Returns one dictionary passed to every worker. |
| `compute_test_statistics(...)` | Both statistics for all hypotheses at once, each normalized by its own threshold so that "reject" is simply `> 1` |
| `compute_identified_x(...)` | (DIA only) The adapted estimator under the identified hypothesis |
| `prob_x0_not_in_Bx(...)` | (detection-only) Analytical `Pr(x0 ∉ Bx)` given the deterministic bias |
| `simulate_H0(...)` | `N_sims` replications under the fault-free hypothesis; returns the observed false-alarm rate and the conditional risk |
| `simulate_Hi(...)` | The same under one fault hypothesis, swept over the bias grid |
| `save_H0_results`, `save_Hi_results` | Write the `.txt` files |
| `main()` | Argument parsing and dispatch |

**Parallelism.** The two campaigns parallelize along different axes. The
`Code_1D_analysis/` scripts distribute the `N_sims` replications across workers
(one replication per worker, each sweeping the whole line). The `Code/` scripts
distribute the *bias grid points* across workers, with each worker running all
`N_sims` replications at its point. Both use a `multiprocessing.Pool` with the
geometry passed once through the initializer.

**Seeding.** Every replication draws from
`np.random.default_rng(SeedSequence([base_seed, hypt + 1, i_sim]))`, so streams
are independent across hypotheses and replications and the whole campaign is
reproducible from `--seed` alone. The satellite subset is fixed, not drawn.

### 2.3 Setups

`--setup-nr` selects the false-alarm budget. `factor_convert_alpha` rescales
`alpha_i_Tq` so that both tests attain the same *total* false-alarm rate, which
is what makes the comparison fair:

| `setup_nr` | `alpha` | `factor_convert_alpha` (11 sats) | (all 22 sats) |
|---|---|---|---|
| 1 | 0.01 | 1.795 | 1.68 |
| 2 | 0.001 | 1.65 | 1.65 |
| 3 | 0.1 | 1.790 | 1.85 |

In all three, `psat_GPS = 1e-3` and `psat_GAL = 3e-3`. Individual levels are
`alpha_i_SS = alpha / k` and `alpha_i_Tq = alpha * factor_convert_alpha / k`.

### 2.4 Running on Snellius

Each simulation directory contains the same pair of shell scripts, differing
only in the Python file they invoke and in `--cpus-per-task`:

* **`submit_GNSS_ex_per_hypt.bash`** — the SLURM job for *one* hypothesis. It
  loads the `2023` toolchain (`numba`, `matplotlib`, `SciPy-bundle`), pins every
  BLAS/OpenMP thread count to 1 so that the Python-level `Pool` gets the cores,
  and calls the simulator with the hypothesis index as `$1`.
* **`submit_jobs.sh`** — the driver. It computes `k = m + m(m-1)/2 = 66` and
  submits `sbatch submit_GNSS_ex_per_hypt.bash $nr_hypts` for
  `nr_hypts = 0, 1, ..., 66`, i.e. **67 jobs**: index `0` is `H0` and index `i`
  is alternative `i - 1`.

To launch a campaign, `cd` into the relevant directory and run:

```bash
cd Code_GNSS_example/Code_1D_analysis/det_only
mkdir -p output          # SLURM writes output/output_%j-%a.out
bash submit_jobs.sh
```

The `python` line inside `submit_GNSS_ex_per_hypt.bash` is where `--N-sims`,
`--setup-nr`, and any grid overrides are set. Edit it before submitting. To run
a single hypothesis directly, without SLURM:

```bash
python ./main_code/gnss_simulate_hypothesis_det_only_line.py \
    --hypt 0 --N-sims 10 --setup-nr 1 --nr-samples 100000 \
    --out-base /path/to/output
```

Useful arguments (defaults in brackets):

| Argument | Default | Meaning |
|---|---|---|
| `--hypt` | required | `0` = `H0`, `1..66` = alternative `hypt - 1` |
| `--setup-nr` | 1 | Row of the setup table above |
| `--N-sims` | 10 | Monte Carlo replications |
| `--nr-samples` | `1e5` | Samples per replication |
| `--seed` | 1 | Base seed for the noise streams |
| `--geom-seed` | 1 | Seed reserved for geometry construction |
| `--n-workers` | all cores | Pool size |
| `--out-base` | `/home/bvannoort/CH5/...` | Root of the output tree |
| `--AL-Bx` | `1.5 2.5 3.0` | Alert limits, in metres |
| `--b-min --b-max --b-step` | `0.01 20.0 0.5` | Line sweep (`Code_1D_analysis`), 41 points |
| `--q1-b-*`, `--q2-b1-*`, `--q2-b2-*`, `--b-step` | `b1 ∈ [-20, 20]`, `b2 ∈ [0.01, 20]`, step `0.5` | Grid sweep (`Code`) |

A full campaign is 67 jobs at roughly 45 minutes wall-clock each.

### 2.5 Output layout

Every simulator writes the same tree, one leaf per (alert limit, hypothesis):

```
<out-base>/setup_<setup_nr>/N_sims_<N_sims>/AL_<AL_Bx>/H_<hypt>/
```

with `H_0` for the fault-free case and `H_1 ... H_66` for the alternatives.
Files are comma-delimited:

**`H_0/`** (all four simulators)

| File | Contents |
|---|---|
| `IR_PH0_SS.txt`, `IR_PH0_Tq.txt` | `P_F` given `H0`, averaged over `N_sims` |
| `std_IR_PH0_SS.txt`, `std_IR_PH0_Tq.txt` | Standard error of that mean |
| `alpha_obs_SS.txt`, `alpha_obs_Tq.txt` | Observed false-alarm rate |
| `std_alpha_obs_SS.txt`, `std_alpha_obs_Tq.txt` | Standard error of that mean |

**`H_i/`, `i ≥ 1`**

| File | Contents |
|---|---|
| `IR_SS.txt`, `IR_Tq.txt` | `P_F` given `Hi`, over the bias sweep |
| `std_IR_SS.txt`, `std_IR_Tq.txt` | Standard error of the mean |
| `gamma_SS.txt`, `gamma_Tq.txt` | Detection power (**detection-only campaigns only**) |
| `std_gamma_SS.txt`, `std_gamma_Tq.txt` | Standard error of the mean |
| `b_values.txt` | Bias axis, for a line sweep or a `q = 1` grid hypothesis |
| `b1_grid.txt`, `b2_grid.txt` | Bias axes for a `q = 2` grid hypothesis (`meshgrid(..., indexing='ij')`) |

`gamma` does not depend on `AL_Bx`, but it is duplicated into every `AL_*`
folder so that each leaf is self-contained.

### 2.6 Post-processing

Both post-processing scripts run **locally**, after the results have been copied
back from the cluster. Configure them in the block that starts around line 250:

```python
maindir = r'.../Chapter 5 SS vs GLR/'
scenario_type = 'det_and_ident'   # or 'det_only'
setup_nr = 1
N_sims = 10
```

**`plot_results_GNSS_ex_CH5_from_Snellius_line.py`** — reads the
`Code_1D_analysis/` output and produces the chapter figures. For each alert
limit it draws one figure of `P_F` against `b`, containing:

* blue — total `P_F`, solid for GLR and dashed for SS, with a ±1σ band;
* green — the contribution of `H0` alone, `(P_F | H0) · P(H0)`;
* red — the maximum `P_F` over the bias sweep.

Figures are written as `PF_vs_b.{pdf,png}` under
`Figures/gnss_example/<scenario_type>/setup <setup_nr>/AL=<AL> m/`.

**`plot_results_GNSS_ex_CH5_from_Snellius.py`** — reads the `Code/` output and
prints the max-`P_F` table for both tests across the three alert limits. It also
draws the `(b1, b2)` heat maps for the last `q = 2` hypothesis it loads.

Both scripts assemble the totals the same way. The stored quantities are
**conditional on the hypothesis**, so each is multiplied by its prior exactly
once:

```
P_F(b) = P(H0) · (P_F | H0)  +  Σ_i P(Hi) · (P_F | Hi)(b)
```

with the standard errors combined in quadrature, the hypothesis streams being
independent.

### 2.7 Workflow summary

```
On Snellius:
    cd Code_GNSS_example/Code_1D_analysis/det_only     (then det_and_ident)
    edit --setup-nr / --N-sims in submit_GNSS_ex_per_hypt.bash
    mkdir -p output && bash submit_jobs.sh             (67 jobs)
    repeat for Code/det_only/with_sim_var and Code/det_and_ident/with_sim_var

Locally:
    copy <out-base>/ back, keeping the setup_*/N_sims_*/AL_*/H_* tree
    set maindir, scenario_type, setup_nr in both plot scripts
    run plot_results_GNSS_ex_CH5_from_Snellius_line.py   -> chapter figures
    run plot_results_GNSS_ex_CH5_from_Snellius.py        -> chapter table
```

---

## 3. Conventions worth knowing

* **Everything on disk is conditional on its hypothesis.** No prior probability
  is baked into any saved `P_F` value. The priors are applied once, in the
  post-processing scripts. This holds for the standard deviations too.
* **Normalized statistics.** Both test statistics are divided by their own
  threshold before being saved or compared, so the rejection rule is uniformly
  `statistic > 1`.
* **Uncertainty is the standard error of the mean.** All `std_*.txt` files in
  the GNSS example are `std(..., ddof=0) / sqrt(N_sims)`. The binary example
  stores the raw replication standard deviation instead and divides at plot
  time — a deliberate difference, but one to keep in mind when comparing the two.
* **`AL_Bx` handling.** In detection-only it enters only through closed-form
  expressions; in DIA through empirical counting. Either way, all three alert
  limits come out of a single Monte Carlo campaign.
* **`Prob_H` is applied exactly once.** When editing the post-processing, check
  whether a variable has already been scaled by its prior before multiplying
  again. This is the single easiest thing to get wrong.

---

## 4. Paths to edit before running

Every script uses absolute Windows or Snellius paths from the original machine.
Update these first:

| Script(s) | Variable |
|---|---|
| `Power_SS_vs_Tq_and_IR_binary_case.py` | `savingDir` |
| `plot_IR_binary_*.py` | `savingDir` / `savingDirBase`, `figSavingDir` |
| GNSS simulators | `--out-base` (or the default in `main()`) |
| GNSS plot scripts | `maindir`, `topdir`, `dirN` |

The GNSS plot scripts expect an extra layer relative to the cluster output —
`Data_GNSS_example/Snellius/with_sim_var/[b_sim_1D/]<scenario>/setup_N/N_sims_10/`
— reflecting how the results were filed after being copied back. Either mirror
that when copying, or adjust `topdir`.

Several scripts also reference a personal matplotlib style file
(`mystyle.mplstyle`), which is not part of this repository. Those lines are
commented out in the binary example; comment them out in the GNSS simulators as
well, or supply your own style file.

---

## 5. Notes and known gaps

* The top-level `Code/det_only/` and `Code/det_and_ident/` folders contain
  submit scripts that point at `./main_code/...` paths one level above where the
  simulators actually live. They are earlier wrappers kept for reference; use the
  scripts inside `with_sim_var/` instead.
* `IR_SS_vs_Tq_Detection_and_identification.txt` is a Python script saved with a
  `.txt` extension. It computes the detection-and-identification risk for the
  binary example but contains no `savetxt` calls — it plots inline only. The data
  actually read by `plot_IR_binary_detection_and_ident.py` and
  `plot_IR_binary_det_and_ident_per_ca.py` (from the `Det+Ident/Data/` folders)
  was produced by a saving variant of this script that is not included here.
  Rename the file to `.py` and add the corresponding `savetxt` block if the
  detection-and-identification data needs to be regenerated from scratch.
* The `--geom-seed` argument is accepted by all GNSS simulators but the
  satellite subset is hard-coded, so it currently has no effect.

---

## License

MIT. See `LICENSE`.
