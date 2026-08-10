# -*- coding: utf-8 -*-
"""
Per-Ca figures for the binary hypothesis example.

For each Ca-matrix structure this makes ONE figure with a 2x2 panel grid:

        |   mean (sim.)        |   std of the mean (sim.)
  ------+----------------------+--------------------------
  SS    |  ax_SS_mean   [cb1]  |  ax_SS_std    [cb2]
  GLR   |  ax_GLR_mean  [cb1]  |  ax_GLR_std   [cb2]

Left column  = mean of the simulated IR on the bias grid.
Right column = simulation std of the mean (here: relative std, sigma/IR),
               matching the previous combined std figure.
Top row = SS test, bottom row = GLR test. Each data column has its own
colorbar (shared between its SS and GLR panels).
"""

import os
import numpy as np
import scipy.stats
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from scipy.linalg import LinAlgError
import sys

# ============================================================
# Helpers (carried over from the original script)
# ============================================================
def stable_inverse(A, rcond=1e-12):
    A = np.asarray(A, dtype=float)
    try:
        if np.allclose(A, A.T, atol=1e-12):
            try:
                L = np.linalg.cholesky(A)
                return np.linalg.solve(L.T, np.linalg.solve(L, np.eye(A.shape[0])))
            except LinAlgError:
                print("matrix not positive definite: that is an issue.")
                pass
        return np.linalg.solve(A, np.eye(A.shape[0]))
    except LinAlgError:
        print("Returning Pseudoinverse...")
        return np.linalg.pinv(A, rcond=rcond)


def plusmat(A, Qyy, inverse=False):
    if inverse:
        return stable_inverse(A.T @ Qyy @ A) @ A.T @ Qyy
    return stable_inverse(A.T @ stable_inverse(Qyy) @ A) @ A.T @ stable_inverse(Qyy)


def perpmat(A, Q):
    return np.eye(A.shape[0]) - A @ plusmat(A, Q)


def shift_colorbars_left(fig, colorbars, x_mm):
    """Shift all colorbars left by x_mm millimeters."""
    fig_width_mm = fig.get_size_inches()[0] * 25.4
    dx_frac = x_mm / fig_width_mm
    for _, cax in colorbars:
        pos = cax.get_position()
        cax.set_position([pos.x0 - dx_frac, pos.y0, pos.width, pos.height])
    fig.canvas.draw_idle()


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 16})
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amssymb, amsmath}"

    np.random.seed(40)

    # ---- fixed problem dimensions / parameters ----
    m = 5
    n = 1
    r = m - n
    q = 2
    alpha = 0.05

    sigma = 1.0
    Qyy = np.eye(m) * sigma ** 2
    Qyy_inv = 1 / (sigma ** 2) * np.eye(m)

    BT = scipy.linalg.null_space(np.ones((m, 1)).T).T
    Qtt = BT @ Qyy @ BT.T
    Qtt_inv = stable_inverse(Qtt)

    # left column shows the mean of: "IR" or "gamma"; right column always
    # shows the relative simulation std (sigma/IR), so it requires IR data.
    plot_type = "IR"

    # Convert the stored per-run std to std-of-the-mean (divide by sqrt(nr_sims))
    bool_convert_sigma = True
    nr_sims = 20

    # vmax for the relative-std (right column) colorbar
    vmax_relstd = 0.05

    # ---- directories (adjust to your machine) ----
    savingDirBase = (
        r"C:\Users\bgvannoort\Documents\Dissertation\Chapter 5 SS vs GLR\Data_binary_example\det_only"
    )
    figSavingDir = (
        r"C:\Users\bgvannoort\Documents\Dissertation\Chapter 5 SS vs GLR\Figures\binary_example\det_only"
    )

    types_of_ci_matrices = ["outlier", "linearly_dependent", "almost_in_A"]

    for type_of_ci_matrix in types_of_ci_matrices:
        # =====================================================
        # 1) Geometry for this Ca-matrix (needed for thresholds + Tq contour)
        # =====================================================
        A = np.ones((m, n))
        Bx = 1.5

        if type_of_ci_matrix == "linearly_dependent":
            c1 = np.random.randn(m, 1)
            epsilon = 1e-1
            c2 = c1 + epsilon * np.random.randn(m, 1)
            Ci = np.hstack((c1, c2))
            b1 = np.linspace(-25, 0, 200)
            b2 = np.linspace(0, 20, 200)
        elif type_of_ci_matrix == "outlier":
            Ci = np.zeros((m, q))
            Ci[:, 0] = np.array([1, 0, 0, 0, 0])
            Ci[:, 1] = np.array([0, 1, 0, 0, 0])
            b1 = np.linspace(-5, 5, 100)
            b2 = np.linspace(-5, 5, 100)
        elif type_of_ci_matrix == "almost_in_A":
            epsilon = 1e-2
            perturb = epsilon * np.random.randn(m, q)
            coeffs = np.random.randn(n, q)
            Ci = A @ coeffs
            Ci[:, 0] = 1.5 * Ci[:, 0] / np.linalg.norm(Ci[:, 0])
            Ci[:, 1] = 2 * Ci[:, 1] / np.linalg.norm(Ci[:, 1])
            Ci += perturb
            b1 = np.linspace(-150, 50, 200)
            b2 = np.linspace(0, 300, 200)
            Bx = 15.0
        else:
            raise NotImplementedError(type_of_ci_matrix)

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

        threshold_SS = float(scipy.stats.norm.isf(alpha / 2) * np.sqrt(Qxa - Qx0))
        threshold_Tq = float(scipy.stats.chi2.isf(alpha, df=q))

        # =====================================================
        # 2) Load the simulated grids for this Ca-matrix
        # =====================================================
        if type_of_ci_matrix == "outlier":
            loadingDir = os.path.join(savingDirBase, "2 Outliers", "Data")
        elif type_of_ci_matrix == "linearly_dependent":
            loadingDir = os.path.join(savingDirBase, "Ci-almost-lin-dependent", "Data")
        elif type_of_ci_matrix == "almost_in_A":
            loadingDir = os.path.join(
                savingDirBase, "Ci-columns-almost-in-A", "AL=15", "Data"
            )

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
        data = {}
        for var_name, filename in file_vars:
            data[var_name] = np.loadtxt(os.path.join(loadingDir, filename), delimiter=",")

        b1grid = data["b1grid"]
        b2grid = data["b2grid"]
        IR_grid_SS = data["IR_grid_SS"]
        IR_grid_Tq = data["IR_grid_Tq"]
        gamma_grid_SS = data["gamma_grid_SS"]
        gamma_grid_Tq = data["gamma_grid_Tq"]
        SS_stats = data["SS_stats"]

        if bool_convert_sigma:
            IR_grid_SS[:, 1] = IR_grid_SS[:, 1] / np.sqrt(nr_sims)
            IR_grid_Tq[:, 1] = IR_grid_Tq[:, 1] / np.sqrt(nr_sims)

        nb1, nb2 = len(b1grid), len(b2grid)

        # ---- mean fields (left column) ----
        if plot_type == "IR":
            mean_SS = IR_grid_SS[:, 0].reshape(nb1, nb2)
            mean_Tq = IR_grid_Tq[:, 0].reshape(nb1, nb2)
        elif plot_type == "gamma":
            mean_SS = gamma_grid_SS.reshape(nb1, nb2)
            mean_Tq = gamma_grid_Tq.reshape(nb1, nb2)
        combined_max = float(np.max([mean_SS, mean_Tq]))

        # ---- relative-std fields (right column): sigma / IR ----
        with np.errstate(divide="ignore", invalid="ignore"):
            relstd_SS = (IR_grid_SS[:, 1] / IR_grid_SS[:, 0]).reshape(nb1, nb2)
            relstd_Tq = (IR_grid_Tq[:, 1] / IR_grid_Tq[:, 0]).reshape(nb1, nb2)

        # ---- fine grid for the black GLR (Tq) acceptance contour ----
        b1f, b2f = np.meshgrid(
            np.linspace(np.min(b1), np.max(b1), 100),
            np.linspace(np.min(b2), np.max(b2), 100),
        )
        Tq_fine = np.zeros(b1f.size)
        for cc, (bi1, bi2) in enumerate(zip(b1f.ravel(), b2f.ravel())):
            b_v = np.vstack((bi1, bi2))
            E_t = Cti @ b_v
            Tq_fine[cc] = float(E_t.T @ UMPI_middle_cti @ E_t)
        Tq_fine = Tq_fine.reshape(b1f.shape)

        SS_abs = np.abs(SS_stats.reshape(nb1, nb2))

        # =====================================================
        # 3) Build the 2x2 (+ 2 colorbars) figure
        # =====================================================
        fig = plt.figure(figsize=(12, 9.0))
        gs = GridSpec(
            2, 4, figure=fig,
            width_ratios=[20, 1, 20, 1], height_ratios=[1, 1],
        )

        ax_SS_mean = fig.add_subplot(gs[0, 0])
        ax_GLR_mean = fig.add_subplot(gs[1, 0])
        cax_mean = fig.add_subplot(gs[:, 1])      # colorbar spans both rows

        ax_SS_std = fig.add_subplot(gs[0, 2])
        ax_GLR_std = fig.add_subplot(gs[1, 2])
        cax_std = fig.add_subplot(gs[:, 3])       # colorbar spans both rows

        for ax in (ax_SS_mean, ax_GLR_mean, ax_SS_std, ax_GLR_std):
            ax.set_box_aspect(1)

        # ---- left column: mean ----
        pcm_SS_mean = ax_SS_mean.pcolormesh(
            b1grid, b2grid, mean_SS, vmin=0, vmax=combined_max
        )
        ax_GLR_mean.pcolormesh(
            b1grid, b2grid, mean_Tq, vmin=0, vmax=combined_max
        )

        # ---- right column: relative std ----
        pcm_SS_std = ax_SS_std.pcolormesh(
            b1grid, b2grid, relstd_SS, vmin=0, vmax=vmax_relstd
        )
        ax_GLR_std.pcolormesh(
            b1grid, b2grid, relstd_Tq, vmin=0, vmax=vmax_relstd
        )

        # ---- acceptance-region contours ----
        # SS panels (top): red SS contour only.
        for ax in (ax_SS_mean, ax_SS_std):
            ax.contour(b1grid, b2grid, SS_abs, levels=[threshold_SS],
                       colors="red", linewidths=2)
        # GLR panels (bottom): red SS contour + black Tq contour.
        for ax in (ax_GLR_mean, ax_GLR_std):
            ax.contour(b1grid, b2grid, SS_abs, levels=[threshold_SS],
                       colors="red", linewidths=2)
            ax.contour(b1f, b2f, np.abs(Tq_fine), levels=[threshold_Tq],
                       colors="black", linewidths=2)

        # ---- axes labels / ticks / grid ----
        # y-labels on the left (mean) column only; x-labels on the bottom row.
        ax_SS_mean.set_ylabel(r"$b_2$ [m]", fontsize=18)
        ax_GLR_mean.set_ylabel(r"$b_2$ [m]", fontsize=18)
        ax_GLR_mean.set_xlabel(r"$b_1$ [m]", fontsize=18)
        ax_GLR_std.set_xlabel(r"$b_1$ [m]", fontsize=18)

        # hide x tick labels on the top row to reduce clutter
        ax_SS_mean.tick_params(labelbottom=False)
        ax_SS_std.tick_params(labelbottom=False)

        for ax in (ax_SS_mean, ax_GLR_mean, ax_SS_std, ax_GLR_std):
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.tick_params(labelsize=16)

        # =====================================================
        # 4) Colorbars (one per data column)
        # =====================================================
        colorbars = []

        # --- mean colorbar (left), with a factored-out power of ten ---
        cb_mean = fig.colorbar(pcm_SS_mean, cax=cax_mean)
        pos = cax_mean.get_position()
        cax_mean.set_position([pos.x0 - 0.015, pos.y0, pos.width, pos.height])
        colorbars.append([cb_mean, cax_mean])
        cb_mean.set_label(r"$\mathbb{P}_{\mathcal{F}}$ [-]",
                          fontsize=18, rotation=270, labelpad=18)
        cb_mean.ax.tick_params(labelsize=16)

        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        cb_mean.formatter = fmt
        cb_mean.update_ticks()
        cb_mean.ax.yaxis.get_offset_text().set_fontsize(18)
        fig.canvas.draw()
        s = cb_mean.ax.yaxis.get_offset_text().get_text()
        cb_mean.ax.yaxis.get_offset_text().set_visible(False)
        cb_mean.ax.text(1.35, 0.96, s, transform=cb_mean.ax.transAxes,
                        ha="left", va="top", fontsize=16)

        # --- relative-std colorbar (right) ---
        cb_std = fig.colorbar(pcm_SS_std, cax=cax_std)
        pos = cax_std.get_position()
        cax_std.set_position([pos.x0 - 0.015, pos.y0, pos.width, pos.height])
        colorbars.append([cb_std, cax_std])
        cb_std.set_label(r"$\sigma_{\mathbb{P}_{\mathcal{F}}} / \mathbb{P}_{\mathcal{F}}$ [-]",
                         fontsize=18, rotation=270, labelpad=18)
        cb_std.ax.tick_params(labelsize=16)

        # =====================================================
        # 5) Whitespace control (tweak these like the previous figures)
        # =====================================================
        
        fig.subplots_adjust(
            top=0.985,
            bottom=0.075,
            left=0.000,
            right=0.965,
            hspace=0.075,
            wspace=0.0
        )
        shift_colorbars_left(fig, colorbars, 15)
        # =====================================================
        # 6) Save
        # =====================================================
        os.makedirs(figSavingDir, exist_ok=True)
        fname = f"PPF_mean_and_simstd_grid_{type_of_ci_matrix}"
        fig.savefig(os.path.join(figSavingDir, fname + ".png"),
                    dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(figSavingDir, fname + ".pdf"),
                    bbox_inches="tight")
        # sys.exit()

    plt.show()