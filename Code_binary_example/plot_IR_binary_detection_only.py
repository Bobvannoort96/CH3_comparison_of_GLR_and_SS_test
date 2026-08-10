# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 09:12:19 2025

@author: bgvannoort
"""

import numpy as np
from numpy.linalg import inv
import scipy.stats
import matplotlib.pyplot as plt
import sys
from scipy.linalg import LinAlgError
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
import os

# plt.style.use(r'C:/Users/bgvannoort/Documents/IDS/Code/MatplotlibStyle/mystyle.mplstyle')

# plt.rcParams.update({
#     "text.usetex": True,
#     # This line tells Matplotlib to load the amsmath package
#     "text.latex.preamble": r"\usepackage{amsmath}",
#     "font.family": "serif",
# })

plt.rcParams.update({"font.size": 16})


# Compute the std of the simulation on the difference
def sigma_diff(sigma_a, sigma_b):
    """
    1-sigma uncertainty of  diff = a - b , assuming a, b independent:
        sigma_diff = sqrt(sigma_a^2 + sigma_b^2)
    """
    return np.sqrt(sigma_a**2 + sigma_b**2)

# Compute the std of the relative difference from the simulation.
def sigma_rel_diff(a, b, sigma_a, sigma_b):
    """
    1-sigma uncertainty of  diff_rel = (a - b)/a = 1 - b/a ,
    first-order Taylor (delta method), assuming a, b independent:
        d(diff_rel)/da =  b / a^2 ,   d(diff_rel)/db = -1 / a
        sigma_rel = (1/a^2) * sqrt( b^2 sigma_a^2 + a^2 sigma_b^2 )
    """
    return np.sqrt((b**2 / a**4) * sigma_a**2 + (1.0 / a**2) * sigma_b**2)


def shift_colorbars_left(fig, colorbars, x_mm):
    """
    Shift all colorbars to the left by x_mm millimeters.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the colorbars.
    colorbars : list of [Colorbar, Axes]
        List where each entry is [cb, cax] as stored in the script.
    x_mm : float
        Amount to shift to the left, in millimeters. Positive values
        move the colorbars left; negative values move them right.
    """
    # Figure width in inches -> mm
    fig_width_in = fig.get_size_inches()[0]
    fig_width_mm = fig_width_in * 25.4

    # Convert the requested shift (mm) to a fraction of the figure width
    dx_frac = x_mm / fig_width_mm

    for _, cax in colorbars:
        pos = cax.get_position()
        cax.set_position([pos.x0 - dx_frac, pos.y0, pos.width, pos.height])

    fig.canvas.draw_idle()

def stable_inverse(A, rcond=1e-12):
    """
    Compute a numerically stable inverse of matrix A.

    Parameters
    ----------
    A : ndarray
        Square matrix to invert.
    rcond : float, optional
        Cutoff for small singular values in pseudo-inverse (default 1e-12).

    Returns
    -------
    A_inv : ndarray
        Inverse (or pseudo-inverse if A is ill-conditioned).
    """
    A = np.asarray(A, dtype=float)

    try:
        # Try Cholesky if symmetric positive definite
        # print(A, A.T)
        if np.allclose(A, A.T, atol=1e-12):
            try:
                L = np.linalg.cholesky(A)
                invA = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(A.shape[0])))
                return invA
            except LinAlgError:
                # Not positive definite, fall back to SVD
                print("matrix not positive definite: that is an issue.")
                pass

        # Try LU decomposition (solve system instead of explicit inverse)
        I = np.eye(A.shape[0])
        invA = np.linalg.solve(A, I)
        return invA

    except LinAlgError:
        # Fall back to pseudo-inverse if singular or near-singular
        print("Returning Pseudoinverse...")
        return np.linalg.pinv(A, rcond=rcond)


def plusmat(A, Qyy, inverse=False):
    if inverse:
        return stable_inverse(A.T @ Qyy @ A) @ A.T @ Qyy
    else:
        return stable_inverse(A.T @ stable_inverse(Qyy) @ A) @ A.T @ stable_inverse(Qyy)


def perpmat(A, Q):
    return np.eye(A.shape[0]) - A @ plusmat(A, Q)


if __name__ == "__main__":
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb, amsmath}'
    np.random.seed(40)
    PH0 = 0.95
    PHi = 0.05

    m = 5
    n = 1
    r = m - n
    q = 2

    Bx = 1.5

    A = np.ones((m, 1))
    BT = scipy.linalg.null_space(A.T).T

    sigma = 1.0
    Qyy = np.eye(m) * sigma**2
    Qyy_inv = 1 / (sigma**2) * np.eye(m)

    Qtt = BT @ Qyy @ BT.T
    Qtt_inv = stable_inverse(Qtt)

    # alpha = 1e-4
    alpha = 0.05
    BT = scipy.linalg.null_space(A.T).T
    Ci = np.zeros((m, 2))
    Qtt = BT @ Qyy @ BT.T

    # for the grid of b-values
    fig = plt.figure(figsize=(16.6, 9))
    gs = GridSpec(
        2, 6, figure=fig, width_ratios=[20, 1, 20, 1 , 20, 1], height_ratios=[1, 1]
    )
    colorbars = []

    fig_sigma = plt.figure(figsize=(16.6, 8.99))
    gs_sigma = GridSpec(
        2, 6, figure=fig_sigma, width_ratios=[20, 1, 20, 1, 20, 1], height_ratios=[1, 1]
    )
    colorbars_sigma = []

    # --- Create subplots --- for the flattened array
    fig_flat, ax_zoom = plt.subplots(3, 1, figsize=(16, 13), sharex=False)

    plot_type = (
        "IR"  # can be IR or gamma, plotting either the power or the IR on the grid.
    )
    all_diffs = []
    all_diff_perc = []
    all_sd_diff = []
    all_sd_perc = []
    
    IRS_all_SS = []
    IRS_all_GLR = []
    types_of_ci_matrices = ["outlier", "linearly_dependent", "almost_in_A"]
    for i_col, type_of_ci_matrix in enumerate(types_of_ci_matrices):
        A = np.ones((m, n))
        if type_of_ci_matrix == "default":
            Ci = np.zeros((m, q))
            Ci[:, 0] = np.array([1, 0, 2, 0, 0])
            Ci[:, 1] = np.array([2, 1, 5, 1, -2])
            # define bias grid
            b1 = np.linspace(-5, 5, 100)
            b2 = np.linspace(-5, 5, 100)
            B1, B2 = np.meshgrid(b1, b2)
        elif type_of_ci_matrix == "perpendicular":
            # #Randomly generate a Ci.
            # Ci = np.random.randint(-10, 10, size=(m,q))/4

            # generate random matrix
            M = np.random.randn(m, q) * -5 + 10

            # orthogonalize columns using QR decomposition
            Q, _ = np.linalg.qr(M)

            # keep only first q columns
            Ci = Q[:, :q] * 3

            # print("Ci =\n", Ci)

            # # check orthogonality
            # print("Ci^T Ci =\n", Ci.T @ Ci)
            # define bias grid
            b1 = np.linspace(-5, 5, 100)
            b2 = np.linspace(-5, 5, 100)
            B1, B2 = np.meshgrid(b1, b2)

        elif type_of_ci_matrix == "linearly_dependent":
            ## instead here we want almost linearly dependent columns.
            # start with one random column
            c1 = np.random.randn(m, 1)

            # make second column almost parallel to c1, plus a small perturbation
            epsilon = 1e-1  # controls how close they are
            perturb = epsilon * np.random.randn(m, 1)

            c2 = c1 + perturb

            Ci = np.hstack((c1, c2))

            # define bias grid
            b1 = np.linspace(-25, 0, 200)
            b2 = np.linspace(0, 20, 200)

            B1, B2 = np.meshgrid(b1, b2)

        elif type_of_ci_matrix == "outlier":
            Ci = np.zeros((m, q))
            Ci[:, 0] = np.array([1, 0, 0, 0, 0])
            Ci[:, 1] = np.array([0, 1, 0, 0, 0])

            b1 = np.linspace(-5, 5, 100)
            b2 = np.linspace(-5, 5, 100)
            B1, B2 = np.meshgrid(b1, b2)
        elif type_of_ci_matrix == "almost_in_A":
            epsilon = 1e-2
            perturb = epsilon * np.random.randn(m, q)

            # take random linear combinations of rows of A
            coeffs = np.random.randn(n, q)
            Ci = A @ coeffs
            Ci[:, 0] = 1.5 * Ci[:, 0] / np.linalg.norm(Ci[:, 0])
            Ci[:, 1] = 2 * Ci[:, 1] / np.linalg.norm(Ci[:, 1])

            # add a small disturbance so Ci is not entirely in the column space of A
            Ci += perturb

            b1 = np.linspace(-150, 50, 200)
            b2 = np.linspace(0, 300, 200)
            B1, B2 = np.meshgrid(b1, b2)

            Bx = 15.0

        b_val = np.array([[2], [2]])
        x_true = np.array([0]).reshape(-1, 1)

        Pa = A @ plusmat(A, Qyy)
        Pa_perp = np.eye(m) - Pa
        Ci_bar = Pa_perp @ Ci

        A_Ci = np.hstack((A, Ci))
        Qxa = stable_inverse(A_Ci.T @ Qyy_inv @ A_Ci)[:n, :n]
        Qx0 = stable_inverse(A.T @ Qyy_inv @ A)

        Cti = BT @ Ci
        Ctiplus = plusmat(Cti, Qtt, inverse=False)

        UMPI_middle_cti = (
            Qtt_inv @ Cti @ stable_inverse(Cti.T @ Qtt_inv @ Cti) @ Cti.T @ Qtt_inv
        )

        # store results
        Tss = np.zeros_like(B1)
        UMPI = np.zeros_like(B1)
        Aplus = plusmat(A, Qyy)
        # compute Tss(b) and UMPI(b)
        for i in range(B1.shape[0]):
            for j in range(B1.shape[1]):
                b = np.array([[B1[i, j]], [B2[i, j]]])
                E_t = Cti @ b

                # Here I assume C_i_bar^+ is implemented as plusmat
                Tss[i, j] = (Aplus @ Ci @ Ctiplus @ E_t).squeeze()

                # UMPI test statistic (projection of bias in constraint space)
                UMPI[i, j] = float(E_t.T @ UMPI_middle_cti @ E_t)

        # ---- thresholds (you fill in actual values) ----
        threshold_SS = scipy.stats.norm.isf(alpha / 2) * np.sqrt(Qxa - Qx0)
        threshold_Tq = scipy.stats.chi2.isf(alpha, df=q)

        # ---- Quadratic form contours ----
        # you fill in Q yourself
        Q = Cti.T @ UMPI_middle_cti @ Cti  #
        quad = Q[0, 0] * B1**2 + 2 * Q[0, 1] * B1 * B2 + Q[1, 1] * B2**2
        c_vals = [scipy.stats.chi2.isf(alpha, df=m - n - q), 10, 20, 50]
        c_vals = np.sort(c_vals)

        # Load the data
        # savingDir = (
        #     r"C:/Users/bgvannoort/Documents/IDS/Code/SS_vs_Tq/Results_of_some_examples"
        # )
        
        # Set to true if needs to convert to extra fraction for the sqrt(N_sims)
        bool_convert_sigma = True
        nr_sims = 20 
        savingDir = (
            r"C:\Users\bgvannoort\Documents\Dissertation\Chapter 5 SS vs GLR\Data_binary_example\det_only"
        )
        
        figSavingDir = r'C:\Users\bgvannoort\Documents\Dissertation\Chapter 5 SS vs GLR\Figures\binary_example\det_only' 

        
        if type_of_ci_matrix == "outlier":
            savingDir = os.path.join(savingDir, "2 Outliers", "Data")
        elif type_of_ci_matrix == "linearly_dependent":
            savingDir = os.path.join(savingDir, "Ci-almost-lin-dependent", "Data")
        elif type_of_ci_matrix == "almost_in_A":
            savingDir = os.path.join(
                savingDir, "Ci-columns-almost-in-A", "AL=15", "Data"
            )

        # Directory where the files are saved
        loadingDir = savingDir  # or specify another directory

        # Define filenames and variable names
        file_vars = [
            ("b1grid", "b1grid.txt"),
            ("b2grid", "b2grid.txt"),
            ("IR_grid_SS", "IR_grid_SS.txt"),
            ("IR_grid_Tq", "IR_grid_Tq.txt"),
            ("gamma_grid_SS", "gamma_grid_SS.txt"),
            ("gamma_grid_Tq", "gamma_grid_Tq.txt"),
            ("Tq_stats", "Tq_stats.txt"),
            ("SS_stats", "SS_stats.txt"),
        ]

        # Load data dynamically
        for var_name, filename in file_vars:
            filepath = os.path.join(loadingDir, filename)
            globals()[var_name] = np.loadtxt(filepath, delimiter=",")
            
        if bool_convert_sigma:
            IR_grid_SS[:,1] = IR_grid_SS[:, 1] / np.sqrt(nr_sims)
            IR_grid_Tq[:,1] = IR_grid_Tq[:,1]/np.sqrt(nr_sims)
        
        if plot_type == "IR":
            combined_max = np.max([IR_grid_SS.flatten(), IR_grid_Tq.flatten()])
        elif plot_type == "gamma":
            combined_max = 1.0
        # Axes for SS (top row) and Tq (bottom row)
        ax_SS = fig.add_subplot(gs[0, 2 * i_col])  # top plot
        ax_Tq = fig.add_subplot(gs[1, 2 * i_col])  # bottom plot
        cax = fig.add_subplot(gs[:, 2 * i_col + 1])  # colorbar spans both rows

        ax_SS.set_box_aspect(1)
        ax_Tq.set_box_aspect(1)

        # for the std simulation figure. 
        ax_SS_sigma = fig_sigma.add_subplot(gs_sigma[0, 2 * i_col])  # top plot
        ax_Tq_sigma = fig_sigma.add_subplot(gs_sigma[1, 2 * i_col])  # bottom plot
        cax_sigma = fig_sigma.add_subplot(gs_sigma[:, 2 * i_col + 1])  # colorbar spans both rows

        ax_SS_sigma.set_box_aspect(1)
        ax_Tq_sigma.set_box_aspect(1)






        # --- SS test ---
        if plot_type == "IR":
            pcm_SS = ax_SS.pcolormesh(
                b1grid,
                b2grid,
                IR_grid_SS[:, 0].reshape(len(b1grid), len(b2grid)),
                vmin=0,
                vmax=combined_max,
            )
            
            # --- SS test --- also for the sigma standard deviation simulation
            pcm_SS_sigma = ax_SS_sigma.pcolormesh(
                b1grid,
                b2grid,
                (IR_grid_SS[:, 1] / IR_grid_SS[:,0]).reshape(len(b1grid), len(b2grid)),
                vmin=0,
                vmax=0.05,
            )
            # colorbars.append(pcm_SS)
            ax_SS_sigma.contour(
                b1grid,
                b2grid,
                np.abs(SS_stats.reshape(len(b1grid), len(b2grid))),
                levels=[threshold_SS],
                colors="red",
                linewidths=2,
            )
            
        elif plot_type == "gamma":
            pcm_SS = ax_SS.pcolormesh(
                b1grid,
                b2grid,
                gamma_grid_SS.reshape(len(b1grid), len(b2grid)),
                vmin=0,
                vmax=combined_max,
            )

        # colorbars.append(pcm_SS)
        ax_SS.contour(
            b1grid,
            b2grid,
            np.abs(SS_stats.reshape(len(b1grid), len(b2grid))),
            levels=[threshold_SS],
            colors="red",
            linewidths=2,
        )
        # ax_SS.set_title(f"SS test – Config {col+1}")
        # ax_SS.set_xlabel('b1 [m]')
        if i_col == 0:
            ax_SS.set_ylabel(r"$b_2$ [m]", fontsize=18)
            ax_SS_sigma.set_ylabel(r"$b_2$ [m]", fontsize=18)
        # ax_SS.set_aspect(1)
        ax_SS.grid(True, linestyle="--", alpha=0.6)
        ax_SS.tick_params(labelsize=16)
        ax_SS_sigma.grid(True, linestyle="--", alpha=0.6)
        
        # --- Tq test ---
        if plot_type == "IR":
            pcm_Tq = ax_Tq.pcolormesh(
                b1grid,
                b2grid,
                IR_grid_Tq[:, 0].reshape(len(b1grid), len(b2grid)),
                vmin=0,
                vmax=combined_max,
            )
            # also for the std of the simulation
            pcm_Tq_sigma = ax_Tq_sigma.pcolormesh(
                b1grid,
                b2grid,
                (IR_grid_Tq[:, 1] / IR_grid_Tq[:, 0]).reshape(len(b1grid), len(b2grid)),
                vmin=0,
                vmax=0.05,
            )
            ax_Tq_sigma.contour(
                b1grid,
                b2grid,
                np.abs(SS_stats.reshape(len(b1grid), len(b2grid))),
                levels=[threshold_SS],
                colors="red",
                linewidths=2,
            )
        elif plot_type == "gamma":
            pcm_Tq = ax_Tq.pcolormesh(
                b1grid,
                b2grid,
                gamma_grid_Tq.reshape(len(b1grid), len(b2grid)),
                vmin=0,
                vmax=combined_max,
            )
        ax_Tq.contour(
            b1grid,
            b2grid,
            np.abs(SS_stats.reshape(len(b1grid), len(b2grid))),
            levels=[threshold_SS],
            colors="red",
            linewidths=2,
        )

        # finer contour for Tq
        b1grid_fine, b2grid_fine = np.meshgrid(
            np.linspace(np.min(b1), np.max(b1), 100),
            np.linspace(np.min(b2), np.max(b2), 100),
        )
        for cc, (i, j) in enumerate(zip(b1grid_fine.flatten(), b2grid_fine.flatten())):
            b_v = np.vstack((i, j))
            Tq = (
                (Cti @ b_v).T
                @ Qtt_inv
                @ Cti
                @ stable_inverse(Cti.T @ Qtt_inv @ Cti)
                @ Cti.T
                @ Qtt_inv
                @ (Cti @ b_v)
            )
            Tq_stats[cc] = Tq
        ax_Tq.contour(
            b1grid_fine,
            b2grid_fine,
            np.abs(Tq_stats.reshape(len(b1grid_fine), len(b2grid_fine))),
            levels=[threshold_Tq],
            colors="black",
            linewidths=2,
        )
        ax_Tq_sigma.contour(
            b1grid_fine,
            b2grid_fine,
            np.abs(Tq_stats.reshape(len(b1grid_fine), len(b2grid_fine))),
            levels=[threshold_Tq],
            colors="black",
            linewidths=2,
        )
        # ax_Tq.set_title(f"Tq test – Config {col+1}")

        ax_Tq.set_xlabel(r"$b_1$ [m]", fontsize=18)
        ax_Tq_sigma.set_xlabel(r"$b_1$ [m]", fontsize=18)
        if i_col == 0:
            ax_Tq.set_ylabel(r"$b_2$ [m]", fontsize=18)
            ax_Tq_sigma.set_ylabel(r"$b_2$ [m]", fontsize=18)
        # ax_Tq.set_aspect(1)
        ax_Tq.grid(True, linestyle="--", alpha=0.6)
        ax_Tq.tick_params(labelsize=16)
        ax_Tq_sigma.grid(True, linestyle="--", alpha=0.6)
        
        # --- Shared colorbar per column ---
        cb = fig.colorbar(pcm_SS, cax=cax)
        pos = cax.get_position()
        cax.set_position([pos.x0 - 0.015, pos.y0, pos.width, pos.height])
        colorbars.append([cb, cax])
        # --- colorbar (fig): label + ticks ---
        # fig  (all three colorbars)
        cb.set_label(r"$\mathbb{P}_{\mathcal{F}}$", fontsize=18, rotation=270, labelpad=15)

        
        cb.ax.tick_params(labelsize=16)
        
        # --- factor out a common power of ten on the fig colorbar ---
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))          # always pull out a power of ten
        cb.formatter = fmt
        cb.update_ticks()
        cb.ax.yaxis.get_offset_text().set_fontsize(18)  # size of the "x10^-4" at top
        
       
        fig.canvas.draw()
        s = cb.ax.yaxis.get_offset_text().get_text()
        cb.ax.yaxis.get_offset_text().set_visible(False)
        cb.ax.text(1.35, 0.96, s, transform=cb.ax.transAxes,
                   ha="left", va="top", fontsize=16)

        
        cb_sigma = fig_sigma.colorbar(pcm_SS_sigma, cax=cax_sigma)
        pos_sigma = cax_sigma.get_position()
        cax_sigma.set_position([pos_sigma.x0 - 0.015, pos_sigma.y0, pos_sigma.width, pos_sigma.height])
        colorbars_sigma.append([cb_sigma, cax_sigma])
        
        # --- colorbar (fig_sigma): label + ticks ---
        # fig_sigma  (all three colorbars)
        cb_sigma.set_label(r"$\sigma_{\mathbb{P}_{\mathcal{F}}} / \mathbb{P}_{\mathcal{F}}$", 
                           fontsize=18, rotation=270, labelpad=5)
        cb_sigma.ax.tick_params(labelsize=16)
        
        # # --- same for the fig_sigma colorbar ---
        fig_sigma.canvas.draw()
        s_sigma = cb_sigma.ax.yaxis.get_offset_text().get_text()
        cb_sigma.ax.yaxis.get_offset_text().set_visible(False)
        cb_sigma.ax.text(1.35, 0.96, s_sigma, transform=cb_sigma.ax.transAxes,
                         ha='left', va='top', fontsize=16)

        # %% Schrijf hier een functie die de flattened array plot

        # # --- Main plot ---
        # ax_main.plot(IR_grid_SS[:, 0],  color='red')
        # ax_main.plot(IR_grid_Tq[:, 0],  color='blue')

        diff = IR_grid_SS[:, 0] - IR_grid_Tq[:, 0]
        all_diffs.append(diff)
        
        IRS_all_SS.append(IR_grid_SS[:,0])
        IRS_all_GLR.append(IR_grid_Tq[:,0])
        
        
        # ax_main.plot(diff,  color='black')

        # ax_main.set_xlabel('Grid point')
        # ax_main.set_ylabel('Probability')
        # ax_main.set_title('IR for SS and Tq test for different values of $b_a$, Ci {} columns'.format(type_of_ci_matrix))
        # ax_main.grid(True, which="both", linestyle="--", alpha=0.6)
        # ax_main.set_xlim(0, len(IR_grid_SS[:, 0]) - 1)

        # --- Highlight zoom area with gray rectangle ---
        if type_of_ci_matrix == "outlier":
            x1, x2, y1, y2 = 200, 700, -0.001, 0.003
        elif type_of_ci_matrix == "linearly_dependent":
            x1, x2, y1, y2 = 1400, 1900, -0.0017, 0.0065
        elif type_of_ci_matrix == "almost_in_A":
            # x1, x2, y1, y2 =  2300, 2500, -0.005, 0.05
            x1, x2, y1, y2 = 200, 700, -0.006, 0.05
        else:
            x1, x2, y1, y2 = 2300, 2500, -0.005, 0.05
        # rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
        #                          linewidth=1.5, edgecolor='gray', facecolor='none', alpha=0.6)
        # ax_main.add_patch(rect)
        # ax_main.set_ylim(1.5*y1, 1.5*y2)

        # --- Zoomed-in plot ---
        ax_zoom[i_col].plot(
            IR_grid_SS[:, 0],
            color="red",
            label=r"$\mathbb{P}_{\mathcal{F}}^{\text{SS}}$",
            linestyle="-",
            linewidth=2.5,
        )
        ax_zoom[i_col].plot(
            IR_grid_Tq[:, 0], color="blue", label=r"$\mathbb{P}_{\mathcal{F}}^{\text{GLR}}$", linestyle="-",
            linewidth=2.5,
        )
        ax_zoom[i_col].plot(
            diff,
            color="black",
            label=r"$\mathbb{P}_{\mathcal{F}}^{\text{SS}} - \mathbb{P}_{\mathcal{F}}^{\text{GLR}}$",
            linestyle="-",
            linewidth=2.5,
        )

        # --- Secondary y-axis for percentage difference ---
        ax_zoom2 = ax_zoom[i_col].twinx()
        perc_diff = 100 * (diff / IR_grid_SS[:, 0])
        all_diff_perc.append(perc_diff)
        ax_zoom2.plot(perc_diff, color="green", linestyle="-", label="Rel. diff \% of SS", linewidth=2.5)
        ax_zoom2.set_ylabel("Difference [\%]", color="green", fontsize=18)
        ax_zoom2.tick_params(axis="y", labelcolor="green", labelsize=16)
        

        ax_zoom[i_col].set_xlim(x1, x2)

        ax_zoom[i_col].set_ylim(y1, y2)
        
        # --- error bars on diff and on the relative (%) difference ---
        a_ir, b_ir = IR_grid_SS[:, 0], IR_grid_Tq[:, 0]
        s_a, s_b   = IR_grid_SS[:, 1], IR_grid_Tq[:, 1]

        sd_diff = sigma_diff(s_a, s_b)                    # error bar for diff
        sd_perc = 100.0 * sigma_rel_diff(a_ir, b_ir, s_a, s_b)  # error bar for %-diff

        x = np.arange(len(diff))
        step = 40  # one error bar every 'step' grid points, to avoid clutter

        ax_zoom[i_col].errorbar(
            x[::step], diff[::step], yerr=sd_diff[::step],
            fmt="none", ecolor="black", elinewidth=0.8, capsize=2, alpha=0.7,
        )
        ax_zoom2.errorbar(
            x[::step], perc_diff[::step], yerr=sd_perc[::step],
            fmt="none", ecolor="green", elinewidth=0.8, capsize=2, alpha=0.7,
        )
        
        # store for the combined (sorted) fig_all plot
        all_sd_diff.append(sd_diff)
        all_sd_perc.append(sd_perc)
        

        ax_zoom[i_col].set_ylabel("P[-]", fontsize=18)
        ax_zoom[i_col].grid(True, linestyle="--", alpha=0.6)
        ax_zoom[i_col].tick_params(labelsize=16)
        if i_col == 0:
            handles1, labels1 = ax_zoom[i_col].get_legend_handles_labels()
            handles2, labels2 = ax_zoom2.get_legend_handles_labels()
            ax_zoom[i_col].legend(
                handles1 + handles2,
                labels1 + labels2,
                loc="upper left",
                bbox_to_anchor=(0.0, 1.2),  # outside top-right corner
                framealpha=1.0,
                borderaxespad=0.0,
                ncols=4, fontsize=18,
            )
        if i_col == 2:
            ax_zoom[i_col].set_xlabel("Grid point (not sorted)", fontsize=18)

    fig_flat.tight_layout()
    fig.tight_layout(w_pad=3.0)
    fig_sigma.tight_layout(w_pad=3.0)
    plt.show()

    # %% Combined diff plot (new figure)
    fig_all, ax_all = plt.subplots(figsize=(16.0, 8.025))
    ax_all_perc = ax_all.twinx()
    zz = ["red", "blue", "black"]
    label_names = [r'$C_a^{\text{outliers}}$', r'$C_a^{\text{lin.dep.}}$', r'$C_a^{\text{rangeA}}$']
    
    
    
    step = 40
    for i, diff_curve in enumerate(all_diffs):
        x = np.arange(len(diff_curve))

        # --- absolute difference (sorted) ---
        order = np.argsort(diff_curve)
        ax_all.plot(x, diff_curve[order], color=zz[i], label=label_names[i])
        ax_all.errorbar(
            x[::step], diff_curve[order][::step], yerr=all_sd_diff[i][order][::step],
            fmt="none", ecolor=zz[i], elinewidth=0.8, capsize=2, alpha=0.5,
        )
        # ax_all.fill_between(x, diff - all_sd_diff[i][order], diff + all_sd_diff[i][order], color="black", alpha=0.4, linewidth=0.4)

        # --- relative (%) difference (sorted independently) ---
        perc_curve = all_diff_perc[i]
        order_p = np.argsort(perc_curve)
        ax_all_perc.plot(x, perc_curve[order_p], color=zz[i], linestyle="--")
        ax_all_perc.errorbar(
            x[::step], perc_curve[order_p][::step], yerr=all_sd_perc[i][order_p][::step],
            fmt="none", ecolor=zz[i], elinewidth=0.6, capsize=2, alpha=0.4,
        )
    
    for i, diff_curve in enumerate(all_diffs):
        # ax_all.plot(diff_curve, label=f'Config {i+1}', color=zz[i])
        ax_all.plot(np.sort(diff_curve), color=zz[i], label=label_names[i], linewidth=2.5)
        # ax_all.plot(np.sort(IRS_all_GLR[i]), color=zz[i], linestyle='-.')
        # ax_all.plot(np.sort(IRS_all_SS[i]), color=zz[i], linestyle='dotted')

        ax_all_perc.plot(np.sort(all_diff_perc[i]), color=zz[i], linestyle="--", linewidth=2.5)


    # ax_all.set_yscale('log')
    ax_all_perc.set_ylabel("Rel diff [\%]", fontsize=18)
    # ax_all.set_title(r"D-only $\mathbb{P}_{\mathcal{F}}$ and relative differences as a function of $b$")
    ax_all.set_xlabel("Grid point (sorted)", fontsize=18)
    ax_all.set_ylabel("Diff [-]", fontsize=18)
    
    ax_all.grid(True, linestyle="--", alpha=0.6)
    ax_all.legend()
    ax_all.set_xlim(0, len(diff_curve))
    ax_all.tick_params(labelsize=16)
    ax_all_perc.tick_params(labelsize=16)
    fig_all.tight_layout()

    # Modify grid figure.
    fig.subplots_adjust(top=0.990, bottom=0.065, left=0.000, right=0.98, hspace=0.096, wspace=0.000)
    
    fig_sigma.subplots_adjust(top=0.990, bottom=0.065, left=0.000, right=0.98, hspace=0.096, wspace=0.000)
    # fig.subplots_adjust(top=0.990, bottom=0.065, left=0.000, right=1.000, hspace=0.096, wspace=0.000)
    shift_colorbars_left(fig, colorbars, 10)
    
    # Modify grid figure.
    # fig_sigma.subplots_adjust(top=0.990, bottom=0.065, left=0.000, right=1.000, hspace=0.096, wspace=0.000)
    shift_colorbars_left(fig_sigma, colorbars_sigma, 10)

# %% Combined diff plot — absolute (left) and relative (right)
    fig_all_separate, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(19.2,10.48))
    zz = ["red", "blue", "black"]
    label_names = [r'$C_a^{\text{outliers}}$', r'$C_a^{\text{lin.dep.}}$', r'$C_a^{\text{rangeA}}$']
    
    step = 40
    for i, diff_curve in enumerate(all_diffs):
        x = np.arange(len(diff_curve))

        # --- left: absolute difference (sorted) ---
        order = np.argsort(diff_curve)
        diff_sorted = diff_curve[order]
        sd_diff_sorted = all_sd_diff[i][order]          # std sorted with the SAME order
        ax_abs.plot(x, diff_sorted, color=zz[i], label=label_names[i], linewidth=2.5)
        # ax_abs.fill_between(
        #     x,
        #     diff_sorted - sd_diff_sorted,
        #     diff_sorted + sd_diff_sorted,
        #     color=zz[i], alpha=0.2, linewidth=0,
        # )
        
        ax_abs.errorbar(
            x[::step], diff_sorted[::step], yerr=sd_diff_sorted[::step],
            fmt="none", ecolor=zz[i], elinewidth=0.8, capsize=2, alpha=0.5,
        )
        # ax_abs.set_yscale('symlog')
        # ax_abs.set_ylim(-1e-3, 0.05)

        # --- right: relative (%) difference (sorted independently) ---
        perc_curve = all_diff_perc[i]
        order_p = np.argsort(perc_curve)
        perc_sorted = perc_curve[order_p]
        sd_perc_sorted = all_sd_perc[i][order_p]        # std sorted with the SAME order_p
        ax_rel.plot(x, perc_sorted, color=zz[i], label=label_names[i], linewidth=2.5)
        # ax_rel.fill_between(
        #     x,
        #     perc_sorted - sd_perc_sorted,
        #     perc_sorted + sd_perc_sorted,
        #     color=zz[i], alpha=0.2, linewidth=0,
        # )
        ax_rel.errorbar(
            x[::step], perc_curve[order_p][::step], yerr=sd_perc_sorted[::step],
            fmt="none", ecolor=zz[i], elinewidth=0.6, capsize=2, alpha=0.5,
        )
        ax_rel.set_ylim(-100, 100)

    # --- left subplot styling (absolute) ---
    ax_abs.set_xlabel("Grid point (sorted)", fontsize=18)
    ax_abs.set_ylabel("Diff [-]", fontsize=18)
    ax_abs.grid(True, linestyle="--", alpha=0.6)
    ax_abs.legend(fontsize=18)
    ax_abs.set_xlim(0, len(diff_curve))
    ax_abs.tick_params(labelsize=16)

    # --- right subplot styling (relative) ---
    ax_rel.set_xlabel("Grid point (sorted)", fontsize=18)
    ax_rel.set_ylabel("Rel. diff [\%]", fontsize=18)
    ax_rel.grid(True, linestyle="--", alpha=0.6)
    ax_rel.legend(fontsize=18)
    ax_rel.set_xlim(0, len(diff_curve))
    ax_rel.tick_params(labelsize=16)

    fig_all_separate.tight_layout()
    
    #%% Save all the figures 

    fig.savefig(os.path.join(figSavingDir, 'PPF_grid.png'), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(figSavingDir, 'PPF_grid.pdf'), bbox_inches="tight")
    
    fig_sigma.savefig(os.path.join(figSavingDir, 'PPF_sim_var_on_grid.png'), dpi=300, bbox_inches="tight")
    fig_sigma.savefig(os.path.join(figSavingDir, 'PPF_sim_var_on_grid.pdf'), bbox_inches="tight")
    
    fig_flat.savefig(os.path.join(figSavingDir, 'grid_zoomed_in_flattened.png'), dpi=300, bbox_inches="tight")
    fig_flat.savefig(os.path.join(figSavingDir, 'grid_zoomed_in_flattened.pdf'), bbox_inches="tight")
    
    fig_all.savefig(os.path.join(figSavingDir, 'difference_flattened.png'), dpi=300, bbox_inches="tight")
    fig_all.savefig(os.path.join(figSavingDir, 'difference_flattened.pdf'), dpi=300, bbox_inches="tight")
    
    fig_all_separate.savefig(os.path.join(figSavingDir, 'difference_flattened_separate.png'), dpi=300, bbox_inches='tight')
    fig_all_separate.savefig(os.path.join(figSavingDir, 'difference_flattened_separate.pdf'), bbox_inches='tight')
    