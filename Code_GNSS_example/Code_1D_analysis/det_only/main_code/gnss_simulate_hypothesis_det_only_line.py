"""
gnss_simulate_hypothesis_det_only_line.py

HPC GNSS detection-ONLY simulation, LINE version.

Mirrors gnss_simulate_hypothesis_det_and_ident_line.py but for the
detection-only application. The integrity-risk decomposition is analytical:

    IR | H0  =  (1 - alpha_obs)  * Pr(x0 not in Bx | H0)
    IR | Hi  =  (1 - gamma_i(b)) * Pr(x0 not in Bx | Hi(b))

alpha_obs and gamma_i(b) are estimated by Monte Carlo (averaged over N_sims);
Pr(x0 not in Bx | ...) is computed analytically from the deterministic UP-bias
of the global LSE Aplus @ E[y]. All IR values are CONDITIONAL on the
hypothesis (no prior PH0/PHi multiplied in) -- multiply by the hypothesis
probability in post-processing to obtain the contribution to the total IR.

For q_i = 2 the bias is sampled on a LINE, b = [bi, bi], NOT on a grid;
q_i = 1 is unchanged (b = [bi]). Both produce 1-D IR arrays over a single
b-line.

Parallelization (identical strategy to the det+id line script):
    * One SLURM task per hypothesis (--hypt).
    * Within the task, the N_sims replications are distributed across CPUs
      via multiprocessing.Pool: one worker runs ONE full replication
      (fresh y-vector(s), full b-line sweep). The parent averages the
      per-replication results and reports the across-replication std.

Usage
-----
    python gnss_simulate_hypothesis_det_only_line.py --hypt <int> [opts]

    hypt = 0      → H0 simulation (no bias dependence; one scalar per AL_Bx)
    hypt = 1..k   → alternative hypothesis i with i = hypt - 1

Output
------
    <out-base>/setup_<setup_nr>/N_sims_<N_sims>/AL_<AL_Bx>/H_<hypt>/

    hypt = 0 (H0):
        IR_PH0_SS.txt        scalar  (mean over N_sims, conditional on H0)
        IR_PH0_Tq.txt        scalar
        std_IR_PH0_SS.txt    scalar  (SEM across N_sims)
        std_IR_PH0_Tq.txt    scalar
        alpha_obs_SS.txt     scalar
        alpha_obs_Tq.txt     scalar
        std_alpha_obs_SS.txt scalar
        std_alpha_obs_Tq.txt scalar

    hypt > 0 (q_i = 1 or q_i = 2, both a line):
        IR_SS.txt            1-D array, length len(b_values)
        IR_Tq.txt            1-D array
        std_IR_SS.txt        1-D array (SEM across N_sims)
        std_IR_Tq.txt        1-D array
        gamma_SS.txt         1-D array (mean gamma; AL_Bx-independent)
        gamma_Tq.txt         1-D array
        std_gamma_SS.txt     1-D array (SEM across N_sims)
        std_gamma_Tq.txt     1-D array
        b_values.txt         1-D array (the bi line; for q=2, b = [bi, bi])
"""

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.stats
from scipy.linalg import LinAlgError


# ============================================================
# Linear-algebra helpers
# ============================================================
def stable_inverse(A, rcond=1e-12):
    A = np.asarray(A, dtype=float)
    try:
        if np.allclose(A, A.T, atol=1e-12):
            try:
                L = np.linalg.cholesky(A)
                return np.linalg.solve(L.T, np.linalg.solve(L, np.eye(A.shape[0])))
            except LinAlgError:
                pass
        return np.linalg.solve(A, np.eye(A.shape[0]))
    except LinAlgError:
        return np.linalg.pinv(A, rcond=rcond)


def plusmat(A, Qyy, inverse=False):
    if inverse:
        return stable_inverse(A.T @ Qyy @ A) @ A.T @ Qyy
    return stable_inverse(A.T @ stable_inverse(Qyy) @ A) @ A.T @ stable_inverse(Qyy)


def perpmat(A, Q):
    return np.eye(A.shape[0]) - A @ plusmat(A, Q)


def sigma_user_n(elev_deg, *, out_of_range="clip"):
    ELEV = np.array(
        [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
        dtype=float,
    )
    SIGM = np.array(
        [
            0.4529, 0.3553, 0.3063, 0.2638, 0.2593, 0.2555, 0.2504, 0.2438,
            0.2396, 0.2359, 0.2339, 0.2302, 0.2295, 0.2278, 0.2297, 0.2310,
            0.2274, 0.2277,
        ],
        dtype=float,
    )
    e = np.asarray(elev_deg, dtype=float)
    if out_of_range == "clip":
        return np.interp(e, ELEV, SIGM)
    if out_of_range == "nan":
        out = np.full(e.shape, np.nan)
        mask = (e >= ELEV.min()) & (e <= ELEV.max())
        out[mask] = np.interp(e[mask], ELEV, SIGM)
        return out
    return np.interp(
        e, ELEV, SIGM,
        left=float(out_of_range), right=float(out_of_range),
    )


def compute_design_matrix(elevations, azimuths, sys_arr):
    el = np.radians(np.asarray(elevations))
    az = np.radians(np.asarray(azimuths))
    GPS_clock = (sys_arr == "GPS").astype(float)
    GAL_clock = (sys_arr == "GAL").astype(float)
    return np.column_stack([
        -np.cos(el) * np.sin(az),
        -np.cos(el) * np.cos(az),
        -np.sin(el),
        GPS_clock,
        GAL_clock,
    ])


def get_covariance_matrix(elevations, sys_arr):
    f_L1 = 1575.42e6
    f_L5 = 1176e6
    el_rad = np.deg2rad(elevations)
    el_GPS = elevations[sys_arr == "GPS"]
    el_GAL = elevations[sys_arr == "GAL"]

    sigma_URA_GPS = 0.75
    sigma_URA_GAL = 0.96
    sigma_tropo = 0.12 * (1.001 / np.sqrt(0.002001 + np.sin(el_rad) ** 2))

    sigma_MP_GPS = 0.13 + 0.53 * np.exp(-el_GPS / 10)
    sigma_rec_GPS = 0.15 + 0.43 * np.exp(-el_GPS / 6.9)
    f_iono = np.sqrt((f_L1 ** 4 + f_L5 ** 4) / (f_L1 ** 2 - f_L5 ** 2) ** 2)
    sigma_user_GPS = f_iono * np.sqrt(sigma_MP_GPS ** 2 + sigma_rec_GPS ** 2)
    sigma_user_GAL = sigma_user_n(el_GAL)

    sigma_tot_GPS = sigma_user_GPS ** 2 + sigma_URA_GPS ** 2
    sigma_tot_GAL = sigma_user_GAL ** 2 + sigma_URA_GAL ** 2
    sigma_tot = np.hstack((sigma_tot_GPS, sigma_tot_GAL)) + sigma_tropo ** 2
    return np.diag(sigma_tot)


# ============================================================
# Setup config
# ============================================================
SETUPS = {
    1: dict(psat_GPS=1e-3, psat_GAL=3e-3, alpha=0.01,
            factor_convert_alpha=1.65, factor_convert_alpha_all_sats=1.65),
    2: dict(psat_GPS=1e-3, psat_GAL=3e-3, alpha=0.001,
            factor_convert_alpha=1.65, factor_convert_alpha_all_sats=1.65),
    3: dict(psat_GPS=1e-3, psat_GAL=3e-3, alpha=0.1,
            factor_convert_alpha=1.790, factor_convert_alpha_all_sats=1.85),
}

# Fixed satellite subset (independent of any seed), per user's modification.
KEEP_IDX_11 = np.array([2, 3, 4, 6, 10, 13, 15, 16, 17, 18, 19])


def setup_geometry(setup_nr=1, consider_all_satellites=False, n_keep=11,
                   geom_seed=1):
    """Build the static geometry, hypothesis projectors, and priors.

    The satellite subset is FIXED (KEEP_IDX_11 for 11 sats, or all 22),
    independent of geom_seed, so every hypothesis task uses an identical
    geometry by construction. geom_seed is accepted for interface
    compatibility but no longer affects selection.
    """
    if setup_nr not in SETUPS:
        raise ValueError(f"setup_nr {setup_nr} not in {list(SETUPS)}")
    cfg = SETUPS[setup_nr]

    elevation = np.array([
        82.3, 50.7, 70.3, 18.8, 8.4, 9.5, 44.9, 22.7, 16.4, 14.2, 5.8,
        21.4, 49.7, 17.0, 36.0, 65.5, 32.2, 56.4, 16.4, 17.4, 13.4, 6.2,
    ])
    azimuth = np.array([
        109.6, 121.6, 247.7, 182.8, 170.7, 261.6, 293.9, 314.9, 278.7,
        79.7, 106.3, 43.1, 87.2, 37.5, 161.1, 297.6, 185.5, 103.3,
        260.3, 312.0, 290.4, 108.2,
    ])
    sys_array = np.array([
        "GPS-1", "GPS-2", "GPS-3", "GPS-4", "GPS-8", "GPS-14", "GPS-17",
        "GPS-19", "GPS-22", "GPS-28", "GPS-31", "GPS-32",
        "GAL-3", "GAL-5", "GAL-8", "GAL-13", "GAL-14", "GAL-15", "GAL-21",
        "GAL-23", "GAL-26", "GAL-34",
    ])

    if consider_all_satellites:
        n_keep = 22
        keep_idx = np.arange(n_keep).astype(int)
    else:
        keep_idx = KEEP_IDX_11

    az_used = azimuth[keep_idx]
    el_used = elevation[keep_idx]
    sys_used = sys_array[keep_idx]
    sys_labels = np.array([s.split("-")[0] for s in sys_used])

    A_raw = compute_design_matrix(el_used, az_used, sys_labels)
    Qyy_raw = get_covariance_matrix(el_used, sys_labels)
    Qyy_inv_raw = np.diag(1 / np.diag(Qyy_raw))

    nc_GPS = int(np.sum(sys_labels == "GPS"))
    nc_GAL = int(np.sum(sys_labels == "GAL"))

    # Whiten so Qyy = I after the transformation
    Qyy_sqrt_inv = scipy.linalg.sqrtm(Qyy_inv_raw)
    A = Qyy_sqrt_inv @ A_raw
    m, n = A.shape
    r = m - n
    Qyy = np.eye(m)
    Qyy_inv = np.eye(m)

    B = scipy.linalg.null_space(A.T)
    Qtt = B.T @ Qyy @ B
    Qtt_sqrt_inv = stable_inverse(scipy.linalg.sqrtm(Qtt))
    B = B @ Qtt_sqrt_inv  # so Qtt = I
    Qtt = np.eye(r)
    Qtt_inv = np.eye(r)

    Qx0 = stable_inverse(A.T @ Qyy_inv @ A)
    n_UP = 2
    sigma_squared_UP = float(Qx0[n_UP, n_UP])
    Aplus = plusmat(A, Qyy_inv, inverse=True)

    k = m + m * (m - 1) // 2
    alpha = cfg["alpha"]
    fca = (cfg["factor_convert_alpha_all_sats"]
           if consider_all_satellites else cfg["factor_convert_alpha"])
    alpha_i_SS = alpha / k
    alpha_i_Tq = alpha * fca / k

    # Priors
    psat_GPS = cfg["psat_GPS"]
    psat_GAL = cfg["psat_GAL"]
    PHi_GPS = psat_GPS * (1 - psat_GPS) ** (nc_GPS - 1) * (1 - psat_GAL) ** nc_GAL
    PHi_GAL = psat_GAL * (1 - psat_GAL) ** (nc_GAL - 1) * (1 - psat_GPS) ** nc_GPS
    PHi_GPS_GPS = (psat_GPS ** 2 * (1 - psat_GPS) ** (nc_GPS - 2)
                   * (1 - psat_GAL) ** nc_GAL)
    PHi_GAL_GAL = (psat_GAL ** 2 * (1 - psat_GAL) ** (nc_GAL - 2)
                   * (1 - psat_GPS) ** nc_GPS)
    PHi_GPS_GAL = (psat_GAL * psat_GPS
                   * (1 - psat_GAL) ** (nc_GAL - 1) * (1 - psat_GPS) ** (nc_GPS - 1))

    n_comb_GPS = nc_GPS * (nc_GPS - 1) // 2
    n_comb_GAL = nc_GAL * (nc_GAL - 1) // 2
    n_comb_GPS_GAL = m * (m - 1) // 2 - n_comb_GPS - n_comb_GAL
    PH0 = (1 - nc_GPS * PHi_GPS - nc_GAL * PHi_GAL
           - n_comb_GPS * PHi_GPS_GPS - n_comb_GAL * PHi_GAL_GAL
           - n_comb_GPS_GAL * PHi_GPS_GAL)

    # Build hypotheses (q=1 first, then q=2 with i<j)
    ci_vectors = []
    Pcti_list = []
    cti_plus_list = []
    Qxi_list = []
    Abar_plus_list = []
    PHi_list = []

    for i in range(m):
        ci = Qyy_sqrt_inv @ np.eye(m)[:, i].reshape(-1, 1)
        cti = B.T @ ci
        ci_vectors.append(ci)
        cti_plus = plusmat(cti, Qtt, inverse=False)
        cti_plus_list.append(cti_plus)
        Pcti_list.append(cti @ cti_plus)
        A_ci = np.hstack((A, ci))
        Qxi_bi = stable_inverse(A_ci.T @ Qyy_inv @ A_ci)
        Qxi_list.append(float(Qxi_bi[n_UP, n_UP]))
        Abar_plus_list.append(plusmat(perpmat(ci, Qyy) @ A, Qyy))
        PHi_list.append(PHi_GPS if sys_labels[i] == "GPS" else PHi_GAL)

    for i in range(m):
        for j in range(i + 1, m):
            ci = Qyy_sqrt_inv @ np.eye(m)[:, [i, j]]
            cti = B.T @ ci
            ci_vectors.append(ci)
            cti_plus = plusmat(cti, Qtt, inverse=False)
            cti_plus_list.append(cti_plus)
            Pcti_list.append(cti @ cti_plus)
            A_ci = np.hstack((A, ci))
            Qxi_bi = stable_inverse(A_ci.T @ Qyy_inv @ A_ci)
            Qxi_list.append(float(Qxi_bi[n_UP, n_UP]))
            Abar_plus_list.append(plusmat(perpmat(ci, Qyy) @ A, Qyy))
            idxes, _ = np.where(ci.astype(bool))
            if idxes[0] < nc_GPS and idxes[1] < nc_GPS:
                P_alt = PHi_GPS_GPS
            elif idxes[0] < nc_GPS or idxes[1] < nc_GPS:
                P_alt = PHi_GPS_GAL
            else:
                P_alt = PHi_GAL_GAL
            PHi_list.append(P_alt)

    x_true = np.array([0.5, 1.0, 3.0, 1.0, 2.0]).reshape(-1, 1)
    x3_true = float(x_true[n_UP, 0])

    return dict(
        A=A, B=B, Qyy=Qyy, Qyy_inv=Qyy_inv, Qtt_inv=Qtt_inv, Aplus=Aplus,
        ci_vectors=ci_vectors, cti_plus_list=cti_plus_list,
        Pcti_list=Pcti_list, Qxi_list=Qxi_list,
        Abar_plus_list=Abar_plus_list, PHi_list=PHi_list,
        sys_labels=sys_labels, m=m, n=n, r=r, k=k, n_UP=n_UP,
        sigma_squared_UP=sigma_squared_UP, x_true=x_true, x3_true=x3_true,
        alpha=alpha, alpha_i_SS=alpha_i_SS, alpha_i_Tq=alpha_i_Tq, PH0=PH0,
    )


# ============================================================
# Test statistics (identical formulas to the non-parallel scripts)
# ============================================================
def compute_test_statistics(t_samples, geom):
    """Returns (SS_tests, Tq_tests), shape (k, nr_samples); >1 means reject."""
    Aplus = geom['Aplus']
    n_UP = geom['n_UP']
    sigma_squared_UP = geom['sigma_squared_UP']
    Qxi_list = geom['Qxi_list']
    alpha_i_SS = geom['alpha_i_SS']
    alpha_i_Tq = geom['alpha_i_Tq']
    ci_vectors = geom['ci_vectors']
    cti_plus_list = geom['cti_plus_list']
    Pcti_list = geom['Pcti_list']
    k = geom['k']
    nr_samples = t_samples.shape[1]

    Tq_tests = np.empty((k, nr_samples))
    SS_tests = np.empty((k, nr_samples))

    for i, (ci, cti_plus, Pcti) in enumerate(
        zip(ci_vectors, cti_plus_list, Pcti_list)
    ):
        Pcti_t = Pcti @ t_samples
        thr_Tq = scipy.stats.chi2.isf(alpha_i_Tq, df=ci.shape[1])
        Tqi = np.einsum("ij,ji->i", Pcti_t.T, Pcti_t)
        Tq_tests[i] = Tqi / thr_Tq

        SS_i = Aplus @ ci @ cti_plus @ t_samples
        denom = scipy.stats.norm.isf(
            alpha_i_SS / 2, scale=np.sqrt(Qxi_list[i] - sigma_squared_UP)
        )
        SS_tests[i] = np.abs(SS_i[n_UP, :]) / denom

    return SS_tests, Tq_tests


def prob_x0_not_in_Bx(AL_Bx, bias, sigma_UP):
    """Analytical Pr(x0 - x_true not in [-AL, AL]) for Gaussian x0 with
    mean = x_true + bias and std = sigma_UP (two-sided exclusion)."""
    return (scipy.stats.norm.sf(AL_Bx, loc=bias, scale=sigma_UP)
            + scipy.stats.norm.cdf(-AL_Bx, loc=bias, scale=sigma_UP))


# ============================================================
# Worker globals (populated once via the Pool initializer)
# ============================================================
_GEOM = None
_HYPT = None          # 0 = H0, else 1..k
_HYPT_LOCAL = None    # hypt - 1 (0-indexed into ci_vectors), or None for H0
_AL_BX_ARR = None
_B_VALUES = None
_NR_SAMPLES = None
_BASE_SEED = None


def _init_worker(geom, hypt, al_bx_arr, b_values, nr_samples, base_seed):
    global _GEOM, _HYPT, _HYPT_LOCAL, _AL_BX_ARR, _B_VALUES, _NR_SAMPLES, _BASE_SEED
    _GEOM = geom
    _HYPT = hypt
    _HYPT_LOCAL = None if hypt == 0 else hypt - 1
    _AL_BX_ARR = al_bx_arr
    _B_VALUES = b_values
    _NR_SAMPLES = nr_samples
    _BASE_SEED = base_seed


def _worker_run_sim_H0(i_sim):
    """One full H0 replication: fresh y, returns alpha and conditional IR|H0
    per AL_Bx for this replication.

    Detection-only IR|H0 = (1 - alpha) * Pr(x0 not in Bx | H0), with bias = 0.
    Px0_H0 is deterministic; only alpha is random, so IR uses this rep's alpha.
    """
    geom = _GEOM
    AL_Bx_arr = _AL_BX_ARR
    nr_samples = _NR_SAMPLES
    base_seed = _BASE_SEED

    A = geom['A']
    B = geom['B']
    x_true = geom['x_true']
    sigma_UP = float(np.sqrt(geom['sigma_squared_UP']))

    y_mean_H0 = (A @ x_true).flatten()

    rng = np.random.default_rng(np.random.SeedSequence([base_seed, 0, i_sim]))
    Z = rng.standard_normal((nr_samples, A.shape[0]))
    y_samples_H0 = (Z + y_mean_H0).T
    t_samples_H0 = B.T @ y_samples_H0

    SS_tests, Tq_tests = compute_test_statistics(t_samples_H0, geom)
    alpha_SS = float(np.mean(SS_tests.max(axis=0) > 1))
    alpha_Tq = float(np.mean(Tq_tests.max(axis=0) > 1))

    # Conditional IR|H0 (no PH0): (1 - alpha) * Pr(x0 not in Bx | H0)
    Px0_H0 = 2 * scipy.stats.norm.sf(AL_Bx_arr, scale=sigma_UP)  # (n_AL,)
    IR_SS = (1 - alpha_SS) * Px0_H0
    IR_Tq = (1 - alpha_Tq) * Px0_H0
    return alpha_SS, alpha_Tq, IR_SS, IR_Tq  # IR_*: (n_AL,)


def _worker_run_sim_Hi(i_sim):
    """One full Hi replication: sweeps the whole b-line, drawing fresh noise
    per b-value. Returns, for every b on the line:
        gamma_SS, gamma_Tq (AL_Bx-independent detection rates), and
        conditional IR|Hi(b) per AL_Bx = (1 - gamma) * Pr(x0 not in Bx | Hi(b)),
    for this replication.
    """
    geom = _GEOM
    hypt = _HYPT_LOCAL
    AL_Bx_arr = _AL_BX_ARR
    b_values = _B_VALUES
    nr_samples = _NR_SAMPLES
    base_seed = _BASE_SEED
    n_AL = len(AL_Bx_arr)
    n_b = len(b_values)

    A = geom['A']
    B = geom['B']
    Aplus = geom['Aplus']
    n_UP = geom['n_UP']
    ci_true = geom['ci_vectors'][hypt]
    qi = ci_true.shape[1]
    x_true = geom['x_true']
    x3_true = geom['x3_true']
    sigma_UP = float(np.sqrt(geom['sigma_squared_UP']))

    rng = np.random.default_rng(
        np.random.SeedSequence([base_seed, hypt + 1, i_sim])
    )

    IR_SS = np.zeros((n_AL, n_b))
    IR_Tq = np.zeros((n_AL, n_b))
    gamma_SS = np.zeros(n_b)
    gamma_Tq = np.zeros(n_b)

    for idx_b, bi in enumerate(b_values):
        # b on a LINE: [bi] for q=1, [bi, bi] for q=2
        if qi == 1:
            b_vector = np.array([[bi]])
        else:
            b_vector = np.array([[bi], [bi]])

        y_mean = (A @ x_true + ci_true @ b_vector).flatten()

        # Deterministic UP-bias of the global LSE
        x_mean = Aplus @ y_mean
        bias = float(x_mean[n_UP] - x3_true)

        # Fresh noise for this (i_sim, b)
        Z = rng.standard_normal((nr_samples, A.shape[0]))
        y_samples_Hi = (Z + y_mean).T
        t_samples_Hi = B.T @ y_samples_Hi

        SS_tests, Tq_tests = compute_test_statistics(t_samples_Hi, geom)
        g_SS = float(np.mean(SS_tests.max(axis=0) > 1))
        g_Tq = float(np.mean(Tq_tests.max(axis=0) > 1))
        gamma_SS[idx_b] = g_SS
        gamma_Tq[idx_b] = g_Tq

        # Conditional IR|Hi(b) (no PHi): (1 - gamma) * Pr(x0 not in Bx | Hi(b))
        Px0 = prob_x0_not_in_Bx(AL_Bx_arr, bias, sigma_UP)  # (n_AL,)
        IR_SS[:, idx_b] = (1 - g_SS) * Px0
        IR_Tq[:, idx_b] = (1 - g_Tq) * Px0

    return IR_SS, IR_Tq, gamma_SS, gamma_Tq


# ============================================================
# Drivers (parallel over the N_sims replications)
# ============================================================
def simulate_H0(geom, AL_Bx_list, N_sims, nr_samples, base_seed, n_workers):
    AL_Bx_arr = np.array(AL_Bx_list)
    n_AL = len(AL_Bx_list)

    alpha_SS = np.zeros(N_sims)
    alpha_Tq = np.zeros(N_sims)
    IR_SS = np.zeros((N_sims, n_AL))
    IR_Tq = np.zeros((N_sims, n_AL))

    print(f"[H0] N_sims={N_sims}, nr_samples={nr_samples}, n_workers={n_workers}",
          flush=True)
    t0 = time.time()
    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(geom, 0, AL_Bx_arr, np.empty(0), nr_samples, base_seed),
    ) as pool:
        for i_sim, (a_SS, a_Tq, ir_SS, ir_Tq) in enumerate(
            pool.imap(_worker_run_sim_H0, range(N_sims))
        ):
            alpha_SS[i_sim] = a_SS
            alpha_Tq[i_sim] = a_Tq
            IR_SS[i_sim, :] = ir_SS
            IR_Tq[i_sim, :] = ir_Tq
            print(f"  sim {i_sim + 1}/{N_sims} done "
                  f"(elapsed {time.time() - t0:.1f}s)", flush=True)

    sqrtN = np.sqrt(N_sims)
    return dict(
        alpha_obs_SS=float(alpha_SS.mean()),
        alpha_obs_Tq=float(alpha_Tq.mean()),
        std_alpha_obs_SS=float(alpha_SS.std(ddof=0)) / sqrtN if N_sims > 1 else 0.0,
        std_alpha_obs_Tq=float(alpha_Tq.std(ddof=0)) / sqrtN if N_sims > 1 else 0.0,
        IR_PH0_SS_arr=IR_SS.mean(axis=0),
        IR_PH0_Tq_arr=IR_Tq.mean(axis=0),
        std_IR_PH0_SS_arr=(IR_SS.std(axis=0, ddof=0) / sqrtN
                           if N_sims > 1 else np.zeros(n_AL)),
        std_IR_PH0_Tq_arr=(IR_Tq.std(axis=0, ddof=0) / sqrtN
                           if N_sims > 1 else np.zeros(n_AL)),
        AL_Bx_list=AL_Bx_list,
    )


def simulate_Hi(geom, hypt_local, AL_Bx_list, N_sims, nr_samples, base_seed,
                n_workers, b_values):
    AL_Bx_arr = np.array(AL_Bx_list)
    n_AL = len(AL_Bx_list)
    n_b = len(b_values)
    qi = geom['ci_vectors'][hypt_local].shape[1]

    IR_SS_per_sim = np.zeros((N_sims, n_AL, n_b))
    IR_Tq_per_sim = np.zeros((N_sims, n_AL, n_b))
    gamma_SS_per_sim = np.zeros((N_sims, n_b))
    gamma_Tq_per_sim = np.zeros((N_sims, n_b))

    print(f"[hypt={hypt_local + 1}, qi={qi}] line with {n_b} b-values, "
          f"N_sims={N_sims}, nr_samples={nr_samples}, n_workers={n_workers}",
          flush=True)
    t0 = time.time()
    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(geom, hypt_local + 1, AL_Bx_arr, b_values, nr_samples, base_seed),
    ) as pool:
        for i_sim, (ir_SS, ir_Tq, g_SS, g_Tq) in enumerate(
            pool.imap(_worker_run_sim_Hi, range(N_sims))
        ):
            IR_SS_per_sim[i_sim] = ir_SS
            IR_Tq_per_sim[i_sim] = ir_Tq
            gamma_SS_per_sim[i_sim] = g_SS
            gamma_Tq_per_sim[i_sim] = g_Tq
            print(f"  sim {i_sim + 1}/{N_sims} done "
                  f"(elapsed {time.time() - t0:.1f}s)", flush=True)

    sqrtN = np.sqrt(N_sims)
    IR_grid_SS = IR_SS_per_sim.mean(axis=0)  # (n_AL, n_b)
    IR_grid_Tq = IR_Tq_per_sim.mean(axis=0)
    gamma_grid_SS = gamma_SS_per_sim.mean(axis=0)  # (n_b,)
    gamma_grid_Tq = gamma_Tq_per_sim.mean(axis=0)
    if N_sims > 1:
        std_IR_grid_SS = IR_SS_per_sim.std(axis=0, ddof=0) / sqrtN
        std_IR_grid_Tq = IR_Tq_per_sim.std(axis=0, ddof=0) / sqrtN
        std_gamma_grid_SS = gamma_SS_per_sim.std(axis=0, ddof=0) / sqrtN
        std_gamma_grid_Tq = gamma_Tq_per_sim.std(axis=0, ddof=0) / sqrtN
    else:
        std_IR_grid_SS = np.zeros((n_AL, n_b))
        std_IR_grid_Tq = np.zeros((n_AL, n_b))
        std_gamma_grid_SS = np.zeros(n_b)
        std_gamma_grid_Tq = np.zeros(n_b)

    return dict(
        qi=qi,
        IR_grid_SS=IR_grid_SS,
        IR_grid_Tq=IR_grid_Tq,
        std_IR_grid_SS=std_IR_grid_SS,
        std_IR_grid_Tq=std_IR_grid_Tq,
        gamma_grid_SS=gamma_grid_SS,
        gamma_grid_Tq=gamma_grid_Tq,
        std_gamma_grid_SS=std_gamma_grid_SS,
        std_gamma_grid_Tq=std_gamma_grid_Tq,
        b_values=b_values,
        AL_Bx_list=AL_Bx_list,
    )


# ============================================================
# Output saving
# ============================================================
def _hypt_dir(out_base, setup_nr, N_sims, AL_Bx, hypt_label):
    return (Path(out_base)
            / f'setup_{setup_nr}'
            / f'N_sims_{N_sims}'
            / f'AL_{AL_Bx}'
            / f'H_{hypt_label}')


def save_H0_results(results, out_base, setup_nr, N_sims):
    AL_Bx_list = results['AL_Bx_list']
    for ial, AL_Bx in enumerate(AL_Bx_list):
        d = _hypt_dir(out_base, setup_nr, N_sims, AL_Bx, 0)
        d.mkdir(parents=True, exist_ok=True)
        np.savetxt(d / 'IR_PH0_SS.txt', [results['IR_PH0_SS_arr'][ial]])
        np.savetxt(d / 'IR_PH0_Tq.txt', [results['IR_PH0_Tq_arr'][ial]])
        np.savetxt(d / 'std_IR_PH0_SS.txt', [results['std_IR_PH0_SS_arr'][ial]])
        np.savetxt(d / 'std_IR_PH0_Tq.txt', [results['std_IR_PH0_Tq_arr'][ial]])
        np.savetxt(d / 'alpha_obs_SS.txt', [results['alpha_obs_SS']])
        np.savetxt(d / 'alpha_obs_Tq.txt', [results['alpha_obs_Tq']])
        np.savetxt(d / 'std_alpha_obs_SS.txt', [results['std_alpha_obs_SS']])
        np.savetxt(d / 'std_alpha_obs_Tq.txt', [results['std_alpha_obs_Tq']])


def save_Hi_results(results, hypt_label, out_base, setup_nr, N_sims):
    AL_Bx_list = results['AL_Bx_list']
    for ial, AL_Bx in enumerate(AL_Bx_list):
        d = _hypt_dir(out_base, setup_nr, N_sims, AL_Bx, hypt_label)
        d.mkdir(parents=True, exist_ok=True)
        np.savetxt(d / 'IR_SS.txt', results['IR_grid_SS'][ial], delimiter=',')
        np.savetxt(d / 'IR_Tq.txt', results['IR_grid_Tq'][ial], delimiter=',')
        np.savetxt(d / 'std_IR_SS.txt', results['std_IR_grid_SS'][ial], delimiter=',')
        np.savetxt(d / 'std_IR_Tq.txt', results['std_IR_grid_Tq'][ial], delimiter=',')
        # gamma is AL_Bx-independent but duplicated into each AL dir so each
        # H_<hypt> folder is self-contained.
        np.savetxt(d / 'gamma_SS.txt', results['gamma_grid_SS'], delimiter=',')
        np.savetxt(d / 'gamma_Tq.txt', results['gamma_grid_Tq'], delimiter=',')
        np.savetxt(d / 'std_gamma_SS.txt', results['std_gamma_grid_SS'], delimiter=',')
        np.savetxt(d / 'std_gamma_Tq.txt', results['std_gamma_grid_Tq'], delimiter=',')
        # Single b-line for both q=1 and q=2 (for q=2, b = [bi, bi]).
        np.savetxt(d / 'b_values.txt', results['b_values'], delimiter=',')


# ============================================================
# Entry point
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hypt', type=int, required=True,
                   help='0 = H0, 1..k = alternative i (i = hypt - 1)')
    p.add_argument('--setup-nr', type=int, default=1)
    p.add_argument('--N-sims', type=int, default=10)
    p.add_argument('--nr-samples', type=int, default=int(1e5))
    p.add_argument('--seed', type=int, default=1, help='Base seed for noise RNG.')
    p.add_argument('--geom-seed', type=int, default=1,
                   help='Accepted for interface compatibility; the satellite '
                        'subset is fixed (KEEP_IDX_11), so this has no effect.')
    p.add_argument('--n-workers', type=int, default=None,
                   help='Defaults to SLURM_CPUS_PER_TASK or os.cpu_count(). '
                        'Capped at N_sims (one worker per replication).')
    p.add_argument('--out-base', type=str,
                   default='/home/bvannoort/CH5/Data_GNSS_example/b_sim_1D/det_only')
    p.add_argument('--AL-Bx', type=float, nargs='+', default=[1.5, 2.5, 3.0])
    p.add_argument('--consider-all-satellites', action='store_true')
    p.add_argument('--n-keep', type=int, default=11)
    # Single b-line (used for q=1 and q=2 alike)
    p.add_argument('--b-min', type=float, default=0.01)
    p.add_argument('--b-max', type=float, default=20.0)
    p.add_argument('--b-step', type=float, default=0.5)
    args = p.parse_args()

    avail = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    if args.n_workers is None:
        args.n_workers = avail
    # One worker per replication; never more workers than replications.
    args.n_workers = max(1, min(args.n_workers, args.N_sims))

    print(f"[gnss_simulate_hypothesis_det_only_line] hypt={args.hypt} "
          f"setup_nr={args.setup_nr} N_sims={args.N_sims} "
          f"nr_samples={args.nr_samples} n_workers={args.n_workers}", flush=True)

    geom = setup_geometry(
        setup_nr=args.setup_nr,
        consider_all_satellites=args.consider_all_satellites,
        n_keep=args.n_keep,
        geom_seed=args.geom_seed,
    )
    k = geom['k']
    print(f"  m={geom['m']}, n={geom['n']}, r={geom['r']}, k={k}", flush=True)

    if args.hypt == 0:
        results = simulate_H0(
            geom, args.AL_Bx, args.N_sims, args.nr_samples, args.seed,
            args.n_workers,
        )
        save_H0_results(results, args.out_base, args.setup_nr, args.N_sims)
        print(f"  H0 done. alpha_obs_SS={results['alpha_obs_SS']:.4f}, "
              f"alpha_obs_Tq={results['alpha_obs_Tq']:.4f}", flush=True)
    elif 1 <= args.hypt <= k:
        b_values = np.arange(args.b_min, args.b_max + args.b_step / 2, args.b_step)
        hypt_local = args.hypt - 1
        results = simulate_Hi(
            geom, hypt_local, args.AL_Bx, args.N_sims, args.nr_samples,
            args.seed, args.n_workers, b_values,
        )
        save_Hi_results(results, args.hypt, args.out_base, args.setup_nr,
                        args.N_sims)
        print(f"  hypt {args.hypt} done ({len(b_values)} b-values).", flush=True)
    else:
        raise ValueError(f"hypt={args.hypt} out of range; must be in [0, {k}]")


if __name__ == '__main__':
    main()
