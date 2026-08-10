"""
gnss_simulate_hypothesis.py

HPC-parallelized GNSS detection-and-identification simulation.
One SLURM task per hypothesis; within a task, the b-grid is parallelized
across CPUs via multiprocessing.Pool.

Usage
-----
    python gnss_simulate_hypothesis.py --hypt <int> [options]

    hypt = 0      → H0 simulation (no bias dependence; one scalar per AL_Bx)
    hypt = 1..k   → alternative hypothesis i with i = hypt - 1 (0-indexed
                    into ci_vectors; matches the H_<hypt> labels used in
                    the original non-parallel scripts where H_1..H_k were
                    the alternatives)

Output
------
    <out-base>/setup_<setup_nr>/N_sims_<N_sims>/AL_<AL_Bx>/H_<hypt>/

    For hypt = 0 (H0):
        IR_PH0_SS.txt       scalar  (PH0 * P(unsafe | H0), averaged over N_sims)
        IR_PH0_Tq.txt       scalar
        alpha_obs_SS.txt    scalar  (averaged over N_sims)
        alpha_obs_Tq.txt    scalar

    For hypt > 0 with q_i = 1:
        IR_SS.txt           1-D array, length len(b_values)
        IR_Tq.txt           1-D array, length len(b_values)
        b_values.txt        1-D array, same length

    For hypt > 0 with q_i = 2:
        IR_SS.txt           2-D array, shape (len(b1), len(b2))
        IR_Tq.txt           2-D array, same shape
        b1_grid.txt         2-D array, same shape (meshgrid with indexing='ij')
        b2_grid.txt         2-D array, same shape

The IR values include the prior factor (PH0 or PHi[hypt-1]) so that summing
all H_<hypt> files reconstructs the total integrity risk for that AL_Bx,
matching the convention of PIR_*_per_hypt in the non-parallel scripts.
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
            factor_convert_alpha=1.795, factor_convert_alpha_all_sats=1.68),
    2: dict(psat_GPS=1e-3, psat_GAL=3e-3, alpha=0.001,
            factor_convert_alpha=1.65, factor_convert_alpha_all_sats=1.65),
    3: dict(psat_GPS=1e-3, psat_GAL=3e-3, alpha=0.1,
            factor_convert_alpha=1.790, factor_convert_alpha_all_sats=1.85),
}


def setup_geometry(setup_nr=1, consider_all_satellites=False, n_keep=11,
                   geom_seed=1):
    """Build the static geometry, hypothesis projectors, and priors. All
    derived quantities returned in a dict that the workers receive once via
    the Pool initializer."""
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

    rng_geom = np.random.default_rng(geom_seed)
    if consider_all_satellites:
        n_keep = 22
        keep_idx = np.arange(n_keep).astype(int)
    else:
        keep_idx = np.array([ 2,  3,  4,  6, 10, 13, 15, 16, 17, 18, 19]) 

    az_used = azimuth[keep_idx]
    el_used = elevation[keep_idx]
    sys_used = sys_array[keep_idx]
    sys_labels = np.array([s.split("-")[0] for s in sys_used])

    A_raw = compute_design_matrix(el_used, az_used, sys_labels)
    Qyy_raw = get_covariance_matrix(el_used, sys_labels)
    Qyy_inv_raw = np.diag(1 / np.diag(Qyy_raw))

    nc_GPS = int(np.sum(sys_labels == "GPS"))
    nc_GAL = int(np.sum(sys_labels == "GAL"))

    # Whiten so Qyy = I after the transformation (matches non-parallel scripts)
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

    # Build hypotheses (same ordering as non-parallel scripts: q=1 first,
    # then q=2, with i<j inside the q=2 block)
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
    """Returns (SS_tests, Tq_tests) with shape (k, nr_samples). A value > 1
    means rejection at the chosen alpha_i."""
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


def compute_identified_x(y_samples, argmax, max_test, geom):
    """For each sample: identified x at the n_UP component.
    Detection (max_test >= 1) → constrained estimator with Abar_plus_list[argmax].
    No detection → global LSE Aplus."""
    Aplus = geom['Aplus']
    n_UP = geom['n_UP']
    Abar_plus_list = geom['Abar_plus_list']
    k = geom['k']
    nr_samples = y_samples.shape[1]

    x_id = np.empty(nr_samples)
    detected = max_test >= 1
    for i in range(k):
        idx = (argmax == i) & detected
        if idx.any():
            x_id[idx] = (Abar_plus_list[i] @ y_samples[:, idx])[n_UP, :]
    if (~detected).any():
        x_id[~detected] = (Aplus @ y_samples[:, ~detected])[n_UP, :]
    return x_id


# ============================================================
# H0 simulation: serial across N_sims, no b dependence
# ============================================================
def simulate_H0(geom, AL_Bx_list, N_sims, nr_samples, base_seed):
    AL_Bx_sq = np.array(AL_Bx_list) ** 2
    n_AL = len(AL_Bx_list)
    A = geom['A']
    B = geom['B']
    PH0 = geom['PH0']
    x_true = geom['x_true']
    x3_true = geom['x3_true']

    y_mean_H0 = (A @ x_true).flatten()

    alpha_SS = np.zeros(N_sims)
    alpha_Tq = np.zeros(N_sims)
    IR_SS = np.zeros((N_sims, n_AL))
    IR_Tq = np.zeros((N_sims, n_AL))

    for i_sim in range(N_sims):
        ss = np.random.SeedSequence(entropy=[base_seed, 0, i_sim])
        rng = np.random.default_rng(ss)
        Z = rng.standard_normal((nr_samples, A.shape[0]))
        y_samples_H0 = (Z + y_mean_H0).T
        t_samples_H0 = B.T @ y_samples_H0

        SS_tests, Tq_tests = compute_test_statistics(t_samples_H0, geom)
        max_SS = SS_tests.max(axis=0)
        max_Tq = Tq_tests.max(axis=0)
        argmax_SS = SS_tests.argmax(axis=0)
        argmax_Tq = Tq_tests.argmax(axis=0)

        alpha_SS[i_sim] = float(np.mean(max_SS > 1))
        alpha_Tq[i_sim] = float(np.mean(max_Tq > 1))

        x_id_SS = compute_identified_x(y_samples_H0, argmax_SS, max_SS, geom)
        x_id_Tq = compute_identified_x(y_samples_H0, argmax_Tq, max_Tq, geom)
        sq_SS = (x_id_SS - x3_true) ** 2
        sq_Tq = (x_id_Tq - x3_true) ** 2
        IR_SS[i_sim, :] = np.sum(
            sq_SS[None, :] > AL_Bx_sq[:, None], axis=1
        ) / nr_samples
        IR_Tq[i_sim, :] = np.sum(
            sq_Tq[None, :] > AL_Bx_sq[:, None], axis=1
        ) / nr_samples
        
    # Standard error of the mean across the N_sims replications.
    std_alpha_obs_SS = float(np.std(alpha_SS, ddof=0)) / np.sqrt(N_sims) 
    std_alpha_obs_Tq = float(np.std(alpha_Tq, ddof=0)) / np.sqrt(N_sims) 
    std_IR_PH0_SS_arr = np.std(IR_SS, axis=0, ddof=0) / np.sqrt(N_sims) 
    std_IR_PH0_Tq_arr = np.std(IR_Tq, axis=0, ddof=0) / np.sqrt(N_sims) 
    
    return dict(
        alpha_obs_SS=float(np.mean(alpha_SS)),
        alpha_obs_Tq=float(np.mean(alpha_Tq)),
        IR_PH0_SS_arr=IR_SS.mean(axis=0),
        IR_PH0_Tq_arr=IR_Tq.mean(axis=0),
        std_alpha_obs_SS=std_alpha_obs_SS,
        std_alpha_obs_Tq=std_alpha_obs_Tq,
        std_IR_PH0_SS_arr=std_IR_PH0_SS_arr,
        std_IR_PH0_Tq_arr=std_IR_PH0_Tq_arr,
        AL_Bx_list=AL_Bx_list,
    )


# ============================================================
# Hi simulation: parallel over the b-grid
# ============================================================
# Worker globals, populated once via the Pool initializer.
_GEOM = None
_HYPT_LOCAL = None
_AL_BX_SQ = None
_N_SIMS = None
_NR_SAMPLES = None
_BASE_SEED = None


def _init_worker(geom, hypt_local, AL_Bx_sq, N_sims, nr_samples, base_seed):
    global _GEOM, _HYPT_LOCAL, _AL_BX_SQ, _N_SIMS, _NR_SAMPLES, _BASE_SEED
    _GEOM = geom
    _HYPT_LOCAL = hypt_local
    _AL_BX_SQ = AL_Bx_sq
    _N_SIMS = N_sims
    _NR_SAMPLES = nr_samples
    _BASE_SEED = base_seed


def _worker_simulate_b(work_item):
    """Run N_sims replications at one b-vector. Returns averaged IR
    contribution (PHi[hypt] * P(unsafe | Hi(b))) for each AL_Bx."""
    b_seq_idx, b_save_idx, b_vector = work_item
    geom = _GEOM
    hypt = _HYPT_LOCAL
    AL_Bx_sq = _AL_BX_SQ
    N_sims = _N_SIMS
    nr_samples = _NR_SAMPLES
    base_seed = _BASE_SEED
    n_AL = len(AL_Bx_sq)

    A = geom['A']
    B = geom['B']
    PHi = geom['PHi_list'][hypt]
    ci_true = geom['ci_vectors'][hypt]
    x_true = geom['x_true']
    x3_true = geom['x3_true']

    y_mean = (A @ x_true + ci_true @ b_vector).flatten()

    counts_SS = np.zeros((N_sims, n_AL))
    counts_Tq = np.zeros((N_sims, n_AL))

    for i_sim in range(N_sims):
        # Per-(hypt, b_seq_idx, i_sim) deterministic seed: SeedSequence with
        # an entropy list combines the components in a hash-like manner.
        ss = np.random.SeedSequence(
            entropy=[base_seed, hypt + 1, b_seq_idx, i_sim]
        )
        rng = np.random.default_rng(ss)
        Z = rng.standard_normal((nr_samples, A.shape[0]))
        y_samples_Hi = (Z + y_mean).T
        t_samples_Hi = B.T @ y_samples_Hi

        SS_tests, Tq_tests = compute_test_statistics(t_samples_Hi, geom)
        max_SS = SS_tests.max(axis=0)
        max_Tq = Tq_tests.max(axis=0)
        argmax_SS = SS_tests.argmax(axis=0)
        argmax_Tq = Tq_tests.argmax(axis=0)

        x_id_SS = compute_identified_x(y_samples_Hi, argmax_SS, max_SS, geom)
        x_id_Tq = compute_identified_x(y_samples_Hi, argmax_Tq, max_Tq, geom)
        sq_SS = (x_id_SS - x3_true) ** 2
        sq_Tq = (x_id_Tq - x3_true) ** 2
        counts_SS[i_sim] = np.sum(
            sq_SS[None, :] > AL_Bx_sq[:, None], axis=1
        )
        counts_Tq[i_sim] = np.sum(
            sq_Tq[None, :] > AL_Bx_sq[:, None], axis=1
        )

    IR_per_sim_SS = counts_SS / nr_samples          # (N_sims, n_AL)
    IR_per_sim_Tq = counts_Tq / nr_samples
    IR_SS = IR_per_sim_SS.mean(axis=0)              # (n_AL,)
    IR_Tq = IR_per_sim_Tq.mean(axis=0)

    # Standard error of the mean across N_sims, per AL_Bx.
    std_IR_SS = IR_per_sim_SS.std(axis=0, ddof=0) / np.sqrt(N_sims)
    std_IR_Tq = IR_per_sim_Tq.std(axis=0, ddof=0) / np.sqrt(N_sims)

    return b_save_idx, IR_SS, IR_Tq, std_IR_SS, std_IR_Tq


def simulate_Hi(geom, hypt_local, AL_Bx_list, N_sims, nr_samples, base_seed,
                n_workers, b_grid_params):
    """Build the b-grid for the given hypothesis's q_i, parallelize across
    the b-values via multiprocessing.Pool, return IR grids and the b-grid."""
    AL_Bx_sq = np.array(AL_Bx_list) ** 2
    n_AL = len(AL_Bx_list)
    qi = geom['ci_vectors'][hypt_local].shape[1]
    step = b_grid_params['step']

    if qi == 1:
        b_values = np.arange(b_grid_params['q1_min'],
                             b_grid_params['q1_max'] + step / 2, step)
        n_b = len(b_values)
        work_items = [
            (seq, seq, np.array([[b]])) for seq, b in enumerate(b_values)
        ]
        IR_grid_SS = np.zeros((n_AL, n_b))
        IR_grid_Tq = np.zeros((n_AL, n_b))
        std_IR_grid_SS = np.zeros((n_AL, n_b))
        std_IR_grid_Tq = np.zeros((n_AL, n_b))
        b_grid_save = b_values
    elif qi == 2:
        b1_values = np.arange(b_grid_params['q2_b1_min'],
                              b_grid_params['q2_b1_max'] + step / 2, step)
        b2_values = np.arange(b_grid_params['q2_b2_min'],
                              b_grid_params['q2_b2_max'] + step / 2, step)
        B1_grid, B2_grid = np.meshgrid(b1_values, b2_values, indexing='ij')
        n_b1, n_b2 = B1_grid.shape
        work_items = []
        seq = 0
        for i in range(n_b1):
            for j in range(n_b2):
                bv = np.array([[B1_grid[i, j]], [B2_grid[i, j]]])
                work_items.append((seq, (i, j), bv))
                seq += 1
        IR_grid_SS = np.zeros((n_AL, n_b1, n_b2))
        IR_grid_Tq = np.zeros((n_AL, n_b1, n_b2))
        std_IR_grid_SS = np.zeros((n_AL, n_b1, n_b2))
        std_IR_grid_Tq = np.zeros((n_AL, n_b1, n_b2))
        b_grid_save = (B1_grid, B2_grid)
    else:
        raise ValueError(f"Unsupported q_i={qi}")

    n_total = len(work_items)
    print(f"[hypt={hypt_local + 1}, qi={qi}] {n_total} b-values, "
          f"N_sims={N_sims}, nr_samples={nr_samples}, n_workers={n_workers}",
          flush=True)

    t0 = time.time()
    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(geom, hypt_local, AL_Bx_sq, N_sims, nr_samples, base_seed),
    ) as pool:
        for n_done, (b_idx, IR_SS, IR_Tq, std_IR_SS, std_IR_Tq) in enumerate(
            pool.imap_unordered(_worker_simulate_b, work_items, chunksize=4)
        ):
            if qi == 1:
                IR_grid_SS[:, b_idx] = IR_SS
                IR_grid_Tq[:, b_idx] = IR_Tq
                std_IR_grid_SS[:, b_idx] = std_IR_SS
                std_IR_grid_Tq[:, b_idx] = std_IR_Tq
            else:
                i, j = b_idx
                IR_grid_SS[:, i, j] = IR_SS
                IR_grid_Tq[:, i, j] = IR_Tq
                std_IR_grid_SS[:, i, j] = std_IR_SS
                std_IR_grid_Tq[:, i, j] = std_IR_Tq
            done = n_done + 1
            if done % 100 == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_total - done) / rate if rate > 0 else 0
                print(f"  done {done}/{n_total}  "
                      f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s", flush=True)

    return dict(
        qi=qi,
        IR_grid_SS=IR_grid_SS,
        IR_grid_Tq=IR_grid_Tq,
        std_IR_grid_SS=std_IR_grid_SS,
        std_IR_grid_Tq=std_IR_grid_Tq,
        b_grid=b_grid_save,
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
        np.savetxt(d / 'alpha_obs_SS.txt', [results['alpha_obs_SS']])
        np.savetxt(d / 'alpha_obs_Tq.txt', [results['alpha_obs_Tq']])
        
        np.savetxt(d / 'std_IR_PH0_SS.txt', [results['std_IR_PH0_SS_arr'][ial]])
        np.savetxt(d / 'std_IR_PH0_Tq.txt', [results['std_IR_PH0_Tq_arr'][ial]])
        np.savetxt(d / 'std_alpha_obs_SS.txt', [results['std_alpha_obs_SS']])
        np.savetxt(d / 'std_alpha_obs_Tq.txt', [results['std_alpha_obs_Tq']])

def save_Hi_results(results, hypt_label, out_base, setup_nr, N_sims):
    AL_Bx_list = results['AL_Bx_list']
    qi = results['qi']
    for ial, AL_Bx in enumerate(AL_Bx_list):
        d = _hypt_dir(out_base, setup_nr, N_sims, AL_Bx, hypt_label)
        d.mkdir(parents=True, exist_ok=True)
        np.savetxt(d / 'IR_SS.txt', results['IR_grid_SS'][ial], delimiter=',')
        np.savetxt(d / 'IR_Tq.txt', results['IR_grid_Tq'][ial], delimiter=',')
        
        # Standard error of the mean across N_sims, per b-value, as a
        # significance/quality metric for the simulation.
        np.savetxt(d / 'std_IR_SS.txt', results['std_IR_grid_SS'][ial], delimiter=',')
        np.savetxt(d / 'std_IR_Tq.txt', results['std_IR_grid_Tq'][ial], delimiter=',')
        if qi == 1:
            np.savetxt(d / 'b_values.txt', results['b_grid'], delimiter=',')
        else:
            B1, B2 = results['b_grid']
            np.savetxt(d / 'b1_grid.txt', B1, delimiter=',')
            np.savetxt(d / 'b2_grid.txt', B2, delimiter=',')


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
    p.add_argument('--seed', type=int, default=1,
                   help='Base seed for noise RNG.')
    p.add_argument('--geom-seed', type=int, default=1,
                   help='Seed for the satellite-selection step (must match '
                        'across all hypothesis tasks for a given run).')
    p.add_argument('--n-workers', type=int, default=None,
                   help='Defaults to SLURM_CPUS_PER_TASK or os.cpu_count().')
    p.add_argument('--out-base', type=str,
                   default='/home/bvannoort/CH5/Data_GNSS_example/det_and_ident')
    p.add_argument('--AL-Bx', type=float, nargs='+', default=[1.5, 2.5, 3.0])
    p.add_argument('--consider-all-satellites', action='store_true')
    p.add_argument('--n-keep', type=int, default=11)
    p.add_argument('--q1-b-min', type=float, default=0.01)
    p.add_argument('--q1-b-max', type=float, default=20.0)
    p.add_argument('--q2-b1-min', type=float, default=-20.0)
    p.add_argument('--q2-b1-max', type=float, default=20.0)
    p.add_argument('--q2-b2-min', type=float, default=0.01)
    p.add_argument('--q2-b2-max', type=float, default=20.0)
    p.add_argument('--b-step', type=float, default=0.5)
    args = p.parse_args()

    if args.n_workers is None:
        args.n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK',
                                            os.cpu_count() or 1))

    print(f"[gnss_simulate_hypothesis] hypt={args.hypt} setup_nr={args.setup_nr} "
          f"N_sims={args.N_sims} nr_samples={args.nr_samples} "
          f"n_workers={args.n_workers}", flush=True)

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
        )
        save_H0_results(results, args.out_base, args.setup_nr, args.N_sims)
        print(f"  H0 done. alpha_obs_SS={results['alpha_obs_SS']:.4f}, "
              f"alpha_obs_Tq={results['alpha_obs_Tq']:.4f}", flush=True)
    elif 1 <= args.hypt <= k:
        b_grid_params = dict(
            q1_min=args.q1_b_min, q1_max=args.q1_b_max,
            q2_b1_min=args.q2_b1_min, q2_b1_max=args.q2_b1_max,
            q2_b2_min=args.q2_b2_min, q2_b2_max=args.q2_b2_max,
            step=args.b_step,
        )
        hypt_local = args.hypt - 1
        results = simulate_Hi(
            geom, hypt_local, args.AL_Bx, args.N_sims,
            args.nr_samples, args.seed, args.n_workers, b_grid_params,
        )
        save_Hi_results(
            results, args.hypt, args.out_base, args.setup_nr, args.N_sims,
        )
    else:
        raise ValueError(f"hypt={args.hypt} out of range; must be in [0, {k}]")


if __name__ == '__main__':
    main()
