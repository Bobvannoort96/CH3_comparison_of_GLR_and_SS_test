"""
Description of code:
    We do here an experiment, as in the other file "Power_SS_vs_Tq_and_IR_binary_case". 
    We consider again a binary case, one H0 vs one Hi 
    We would like to know, for an arbitrary example with n=1, q_i=2, what would be 
    the IR, integrity risk, from the detection and identification point of view? 

    
Author: 
    Bob van Noort
Date: 
    ${DATE}

"""

import numpy as np
from numpy.linalg import inv
import scipy.stats
import matplotlib.pyplot as plt
import sys
from scipy.linalg import LinAlgError 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import os 


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
        print('Returning Pseudoinverse...')
        return np.linalg.pinv(A, rcond=rcond)

def plusmat(A, Qyy, inverse=False):
    if inverse:
        return stable_inverse(A.T @ Qyy @ A) @ A.T @ Qyy 
    else:
        return stable_inverse(A.T @ stable_inverse(Qyy) @ A) @ A.T @ stable_inverse(Qyy) 


def perpmat(A,Q):
    return np.eye(A.shape[0]) - A @ plusmat(A,Q)



def calc_IR_with_identification(A, Qyy, Ci, x_true, b_val, alpha, 
            nr_samples, threshold_SS, threshold_Tq, nr_sims=5, Bx=1.5):
    m, n = A.shape
    store_IR = [] # columns are respectively SS test and Tq test
    store_gamma = []
    store_conditional_IRs_SS = [] # store under H0 and Hi (columns) for all simulations (rows)
    store_conditional_IRs_Tq = [] # store under H0 and Hi (columns) for all simulations (rows)
    
    alpha_measured = []
    A_Cy = np.hstack((A, Ci))
    Aplus = plusmat(A,Qyy) 
    Aci_plus = plusmat(A_Cy, Qyy)
    
    Pcti = BT @ Ci @ plusmat(BT @ Ci, Qtt)
    

    for sim in range(nr_sims):
        # under H0 
        y = np.random.multivariate_normal( (A @ x_true).flatten(), Qyy, size=nr_samples).T
        x_0 = (Aplus @ y).flatten()
        x_i =  Aci_plus @ y
        # print('x_i.shape', x_i.shape)
        # print('x_0.shape', x_0.shape)
        print("Warning: we flatten the x_i array, because we assume n=1")
        x_i = x_i[:n,:].flatten() # warning, only possible when n=1
        t = BT @ y
        
    
        
        accept_SS = np.abs(x_0-x_i) < threshold_SS
        Tq_vals = np.einsum('ij,jm,mi->i', (Pcti @ t).T, stable_inverse(Qtt), Pcti @ t)
        accept_Tq = (Tq_vals) < threshold_Tq
        
         
        alpha_SS = 1-np.sum(accept_SS)/nr_samples
        alpha_Tq = 1-np.sum(accept_Tq)/nr_samples
        alpha_measured.append([alpha_SS, alpha_Tq])
        
        
        IR_H0_no_detection = (1-alpha)*np.sum((x_0-x_true)**2 > Bx**2) / nr_samples # the same for both
        # print("IR_H0_no_detection", IR_H0_no_detection)
        
        # IR given false alarm for SS 
        x_i_selected_SS = x_i[~accept_SS]
        IR_H0_ident_SS = alpha*np.sum( (x_i_selected_SS-x_true)**2 > Bx**2 ) / len(x_i_selected_SS)
        
        
        # IR given false alarm Tq 
        x_i_selected_Tq = x_i[~accept_Tq].flatten()
        # print("x_i_selected_SS.shape, x_i_selected_Tq.shape", x_i_selected_SS.shape, x_i_selected_Tq.shape)
        IR_H0_ident_Tq = alpha*np.sum((x_i_selected_Tq-x_true)**2 > Bx**2 ) / len(x_i_selected_Tq)
        # print("IR_H0_ident_SS, IR_H0_ident_Tq", IR_H0_ident_SS, IR_H0_ident_Tq)
        
        # Under Hi
        y_Hi = np.random.multivariate_normal( (A_Cy @ np.vstack((x_true,b_val))).flatten(), Qyy, size=nr_samples).T
        x_0 = (Aplus @ y_Hi).flatten()
        x_i = Aci_plus @ y_Hi # this is actually [xhat, bhat]^T
        x_i = x_i[:n,:].flatten()
        # print("x_i.shape", x_i.shape)
        t_Hi = BT @ y_Hi
        
        
        
        IR_Hi_no_detection = np.sum( (x_0-x_true)**2 > Bx**2) / nr_samples
        # print("IR_Hi_no_detection", IR_Hi_no_detection)
        accept_SS = np.abs(x_0-x_i) < threshold_SS
        Tq_vals = np.einsum('ij,jm,mi->i', (Pcti @ t_Hi).T, stable_inverse(Qtt), Pcti @ t_Hi)
        accept_Tq = (Tq_vals) < threshold_Tq
        
        
        gamma_SS = 1 - np.sum(accept_SS) / nr_samples
        gamma_Tq = 1 - np.sum(accept_Tq) / nr_samples
        
        # The no_detection of SS and Tq
        IR_Hi_no_det_SS = IR_Hi_no_detection * (1-gamma_SS)
        IR_Hi_no_det_Tq = IR_Hi_no_detection * (1-gamma_Tq)
        # print("IR_HI_no_det SS, Tq", IR_Hi_no_det_SS, IR_Hi_no_det_Tq)
        
        
        ## Calculate the det + identification, i.e. if xi is chosen 
        x_i_selected_Hi_SS = x_i[~accept_SS]  
        x_i_selected_Hi_Tq = x_i[~accept_Tq]  
        
        IR_Hi_ident_SS = (gamma_SS) * (np.sum( (x_i_selected_Hi_SS-x_true) **2 > Bx**2) / len(x_i_selected_Hi_SS))
        IR_Hi_ident_Tq = gamma_Tq * (np.sum( (x_i_selected_Hi_Tq-x_true) **2 > Bx**2) / len(x_i_selected_Hi_Tq))
        # print("IR_Hi_identification SS, Tq", IR_Hi_ident_SS, IR_Hi_ident_Tq)
        
        ### calculate total IR per simulation
        
        IR_SS = (IR_H0_no_detection + IR_H0_ident_SS)*PH0 + PHi * (IR_Hi_no_det_SS + IR_Hi_ident_SS)
        IR_Tq =  (IR_H0_no_detection + IR_H0_ident_Tq)*PH0 + PHi * (IR_Hi_no_det_Tq + IR_Hi_ident_Tq)
        store_IR.append([IR_SS, IR_Tq])
        store_gamma.append([gamma_SS, gamma_Tq])
        store_conditional_IRs_SS.append([IR_H0_no_detection + IR_H0_ident_SS, IR_Hi_no_det_SS + IR_Hi_ident_SS])
        store_conditional_IRs_Tq.append([IR_H0_no_detection + IR_H0_ident_Tq, IR_Hi_no_det_Tq + IR_Hi_ident_Tq])
    # print('alpha measured', alpha_measured)    
    store_IR = np.array(store_IR)
    store_gamma = np.array(store_gamma)
    store_conditional_IRs_Tq = np.array(store_conditional_IRs_Tq)
    store_conditional_IRs_SS = np.array(store_conditional_IRs_SS)
    return store_IR, store_gamma, store_conditional_IRs_Tq, store_conditional_IRs_SS


if __name__ == '__main__':
    np.random.seed(40)
    PH0 = 0.95
    PHi=0.05
    
    m=5
    n = 1
    r = m-n
    q=2
    
    A = np.ones((m,1))
    BT = scipy.linalg.null_space(A.T).T
    
    
    
    sigma= 1.0 
    Qyy = np.eye(m)*sigma**2
    Qyy_inv = 1/(sigma**2) * np.eye(m)
    
    Qtt = BT @ Qyy @ BT.T
    Qtt_inv = stable_inverse(Qtt)
    
    alpha = 0.05
    BT = scipy.linalg.null_space(A.T).T
    Ci = np.zeros((m,2))
    Qtt = BT @ Qyy @ BT.T
    
    
    type_of_ci_matrix = 'outlier'
    
    A = np.ones((m,n))
    if type_of_ci_matrix == 'default':
    
        Ci = np.zeros((m,q))
        Ci[:,0] = np.array([1,0,2,0,0])
        Ci[:,1] = np.array([2,1,5,1,-2])
        # define bias grid
        b1 = np.linspace(-5, 5, 100)
        b2 = np.linspace(-5, 5, 100)
        B1, B2 = np.meshgrid(b1, b2)
    elif type_of_ci_matrix == 'perpendicular':
        # #Randomly generate a Ci. 
        # Ci = np.random.randint(-10, 10, size=(m,q))/4
        
        # generate random matrix
        M = np.random.randn(m, q)*-5 + 10
        
        # orthogonalize columns using QR decomposition
        Q, _ = np.linalg.qr(M)
        
        # keep only first q columns
        Ci = Q[:, :q]*3
        
        # print("Ci =\n", Ci)
        
        # # check orthogonality
        # print("Ci^T Ci =\n", Ci.T @ Ci)
        # define bias grid
        b1 = np.linspace(-2, 0, 100)
        b2 = np.linspace(0, 2, 100)
        B1, B2 = np.meshgrid(b1, b2)
    
    elif type_of_ci_matrix == 'linearly_dependent':
        ## instead here we want almost linearly dependent columns. 
        # start with one random column
        c1 = np.random.randn(m, 1)
        
        # make second column almost parallel to c1, plus a small perturbation
        epsilon = 1e-1  # controls how close they are
        perturb = epsilon * np.random.randn(m, 1)
        
        c2 = c1 + perturb
        
        Ci = np.hstack((c1, c2))
        
        # define bias grid
        b1 = np.linspace(-25, 25, 200)
        b2 = np.linspace(0, 25, 200)
         
        B1, B2 = np.meshgrid(b1, b2)
    elif type_of_ci_matrix == 'outlier':
        Ci = np.zeros((m,q))
        Ci[:,0] = np.array([1,0,0,0,0])
        Ci[:,1] = np.array([0,1,0,0,0])
        
        b1 = np.linspace(0, 4, 100)
        b2 = np.linspace(0, 4, 100)
        B1, B2 = np.meshgrid(b1, b2)
     
    b_val = np.array([[1],[1]])
    x_true = np.array([0]).reshape(-1,1)
    nr_samples = int(1e6) 
    nr_sims = 10
        
    Aplus = plusmat(A, Qyy)
    Pa = A @ Aplus
    Pa_perp = np.eye(m) - Pa
    Ci_bar = Pa_perp @ Ci
    
    Cti = BT @ Ci 
    Ctiplus = plusmat(Cti, Qtt)
    
    A_Ci = np.hstack((A, Ci))
    Qxa = stable_inverse(A_Ci.T @ Qyy_inv @ A_Ci)[:n, :n]
    Qx0 = stable_inverse(A.T @ Qyy_inv @ A)
    
    
    
    # ---- thresholds (you fill in actual values) ----
    threshold_SS = scipy.stats.norm.isf(alpha/2) *np.sqrt(Qxa - Qx0)[0,0]
    threshold_Tq = scipy.stats.chi2.isf(alpha, df=q)
    
    
    
    store_IR, store_gamma, store_conditional_IRs_Tq, store_conditional_IRs_SS = calc_IR_with_identification(A, 
                                                                               Qyy, 
                                                                               Ci, 
                                                                               x_true, 
                                                                               b_val, 
                                                                               alpha,  
                                                                               nr_samples, 
                                                                               threshold_SS=threshold_SS, 
                                                                               threshold_Tq=threshold_Tq, 
                                                                               nr_sims= nr_sims)


    fig, ax = plt.subplots()
    ax.set_title('Probabilities of hypothesis testing det+identification \n(1H0 and 1Ha) for SS and Tq test \n with a $b_a=[2,3]^T$, Ci {} columns'.format(type_of_ci_matrix))
    ax.set_xlabel('sim nr')
    ax.set_ylabel('Probability')
    ax.set_yscale('log')
    #plot IR
    ax.plot(np.arange(nr_sims), store_IR[:,0], color='red', label='IR SS test')
    ax.plot(np.arange(nr_sims), store_IR[:, 1], color='blue', label='IR Tq test')
    
    #plot gamma
    ax.plot(np.arange(nr_sims), store_gamma[:,0], '--', color='red', label='power SS test')
    ax.plot(np.arange(nr_sims), store_gamma[:,1], '--', color='blue', label='power Tq test')
    ax.grid('on', alpha=0.6, linestyle='--')
    ax.legend()
    
    
    
    #%% Plot as a function of the 2D bias, q=2
    
    b1_new, b2_new = np.linspace(np.min(b1), np.max(b1), 20), np.linspace(np.min(b2), np.max(b2), 20)
    
    b1grid, b2grid = np.meshgrid(b1_new, b2_new)
    ## make a 3D map of the IR as a function of the b_vals 
    IR_grid_SS = np.zeros((len(b1grid.flatten()), 2)) # first the IR and then the variance
    IR_grid_Tq = np.zeros((len(b1grid.flatten()), 2))
    SS_stats = np.zeros(len(b1grid.flatten()))
    
    gamma_grid_SS = np.zeros(len(b1grid.flatten()))
    gamma_grid_Tq = np.zeros(len(b1grid.flatten()))
    
    conditional_IRs_Tq = np.zeros((len(b1grid.flatten()), 2)) # first col is H0 second col is Hi
    conditional_IRs_SS = np.zeros((len(b1grid.flatten()), 2))
    counter = 0
    for b1_, b2_ in zip(b1grid.flatten(), b2grid.flatten()):
        b_val = np.array([[b1_],[b2_]])
        irs, gammas_, conditional_irs_Tq, conditional_irs_SS = calc_IR_with_identification(A, 
                                                Qyy, 
                                                Ci, 
                                                x_true, 
                                                b_val, 
                                                alpha, 
                                                nr_samples, 
                                                threshold_SS=threshold_SS, 
                                                threshold_Tq=threshold_Tq, 
                                                nr_sims= 5)
        ir_ss, sigma_ir_ss = np.mean(irs[:, 0]), np.std(irs[:,0], ddof=1)
        ir_tq, sigma_ir_tq = np.mean(irs[:,1]), np.std(irs[:,1], ddof=1)
        IR_grid_SS[counter, :] = [ir_ss, sigma_ir_ss]
        IR_grid_Tq[counter, :] = [ir_tq, sigma_ir_tq]
        
        # Store powers of the test
        gamma_grid_SS[counter] = np.mean(gammas_[:, 0])
        gamma_grid_Tq[counter] = np.mean(gammas_[:, 1])
        
        # store conditional IRs 
        conditional_IRs_SS[counter, :] = np.mean(conditional_irs_SS, axis=0)
        conditional_IRs_Tq[counter, :] = np.mean(conditional_irs_Tq, axis=0)
        
        SS_test = Aplus @ Ci @ Ctiplus @ Cti @ b_val 
        
        SS_stats[counter] = np.abs(SS_test)
        
        Tq = (Cti @ b_val).T @ Qtt_inv @ Cti @ stable_inverse(Cti.T @ Qtt_inv @ Cti) @ Cti.T @ Qtt_inv @ (Cti @ b_val)
    
        print('at counter ' + str(counter))
        counter +=1
    #%%
    
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
    combined_max = np.max(np.hstack((IR_grid_SS, IR_grid_Tq)))
    levels = np.linspace(0, combined_max, 10)
    
    fig = plt.figure(figsize=(11.33,  9.95))
    # 2 rows × 2 columns grid → last column is narrow for colorbar
    gs = GridSpec(2, 2, figure=fig, width_ratios=[20, 1], height_ratios=[1, 1])
    
    # First subplot (top-left)
    ax0 = fig.add_subplot(gs[0, 0])
    pcm1 = ax0.pcolormesh(
        b1grid, b2grid, IR_grid_SS[:, 0].reshape(len(b1grid), len(b2grid)),
        vmin=0, vmax=combined_max
    )
    ax0.set_xlabel('b1 value [m]')
    ax0.set_ylabel('b2 value [m]')
    ax0.set_title('IR for the SS test vs different b values, Ci {} columns'.format(type_of_ci_matrix))
    ax0.contour(b1grid, b2grid, np.abs(SS_stats.reshape(len(b1grid), len(b2grid))),
                levels=[threshold_SS], colors='red', linewidths=1)
    ax0.set_aspect(1)
    ax0.grid(True, which="both", linestyle="--", alpha=0.6)
    
    # Second subplot (bottom-left)
    ax1 = fig.add_subplot(gs[1, 0])
    pcm2 = ax1.pcolormesh(
        b1grid, b2grid, IR_grid_Tq[:, 0].reshape(len(b1grid), len(b2grid)),
        vmin=0, vmax=combined_max
    )
    ax1.set_xlabel('b1 value [m]')
    ax1.set_ylabel('b2 value [m]')
    ax1.set_title('IR for the Tq test vs different b values, Ci {} columns'.format(type_of_ci_matrix))
    ax1.contour(b1grid, b2grid, np.abs(SS_stats.reshape(len(b1grid), len(b2grid))),
                levels=[threshold_SS], colors='red', linewidths=1)
    
    # finer contour for Tq
    b1grid_fine, b2grid_fine = np.meshgrid(
        np.linspace(np.min(b1), np.max(b1), 100),
        np.linspace(np.min(b2), np.max(b2), 100)
    )
    Tq_stats = np.zeros(len(b1grid_fine.flatten()))
    for cc, (i, j) in enumerate(zip(b1grid_fine.flatten(), b2grid_fine.flatten())):
        b_v = np.vstack((i, j))
        Tq = (Cti @ b_v).T @ Qtt_inv @ Cti @ stable_inverse(Cti.T @ Qtt_inv @ Cti) @ Cti.T @ Qtt_inv @ (Cti @ b_v)
        Tq_stats[cc] = Tq
    ax1.contour(b1grid_fine, b2grid_fine,
                np.abs(Tq_stats.reshape(len(b1grid_fine), len(b2grid_fine))),
                levels=[threshold_Tq], colors='black', linewidths=1)
    
    ax1.set_aspect(1)
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)
    
    # Colorbar axis spanning both rows (right column)
    cax = fig.add_subplot(gs[:, 1])
    cb = fig.colorbar(pcm2, cax=cax)
    cb.ax.tick_params(labelsize=10)
    
    fig.tight_layout(rect=[0, 0, 0.95, 1])  # leave 7% space on the right
    plt.show()
    
    #%% Plot the flattened array
    import matplotlib.patches as patches

    # --- Create subplots ---
    fig, (ax_main, ax_zoom) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    
    # --- Main plot ---
    ax_main.plot(IR_grid_SS[:, 0],  color='red')
    ax_main.plot(IR_grid_Tq[:, 0],  color='blue')
    
    diff = IR_grid_SS[:, 0] - IR_grid_Tq[:, 0]
    ax_main.plot(diff,  color='black')
    
    
    ax_main.set_xlabel('Grid point')
    ax_main.set_ylabel('Probability')
    ax_main.set_title('IR for SS and Tq test for different values of $b_a$, Ci {} columns'.format(type_of_ci_matrix))
    ax_main.grid(True, which="both", linestyle="--", alpha=0.6)
    ax_main.set_xlim(0, len(IR_grid_SS[:, 0]) - 1)
    
    # --- Highlight zoom area with gray rectangle ---
    if type_of_ci_matrix == 'default':
        x1, x2, y1, y2 = 1090, 1290, -0.006, 0.0022
    elif type_of_ci_matrix == 'perpendicular':
        x1, x2, y1, y2 = 150, 200, 0.003, 0.085
    else:
        x1, x2, y1, y2 = 150, 200, 0.002, 0.016
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                             linewidth=1.5, edgecolor='gray', facecolor='none', alpha=0.6)
    ax_main.add_patch(rect)
    
   
    
    # --- Zoomed-in plot ---
    ax_zoom.plot(IR_grid_SS[:, 0], color='red', label='IR for SS test', linestyle='-')
    ax_zoom.plot(IR_grid_Tq[:, 0], color='blue', label='IR for Tq test', linestyle='-')
    ax_zoom.plot(diff, color='black', label='Difference: SS - Tq', linestyle='-')
    
    
    # --- Secondary y-axis for percentage difference ---
    ax_zoom2 = ax_zoom.twinx()
    perc_diff = 100 * (diff / IR_grid_SS[:, 0])
    ax_zoom2.plot(perc_diff, color='green', linestyle=':', label='Diff % of SS')
    ax_zoom2.set_ylabel('Difference [%]', color='green')
    ax_zoom2.tick_params(axis='y', labelcolor='green')
    
    ax_zoom.set_xlim(x1, x2)
    
    ax_zoom.set_ylim(y1, y2)
    ax_zoom.set_xlabel('Grid point')
    ax_zoom.set_ylabel('Probability [-]')
    ax_zoom.grid(True, linestyle="--", alpha=0.6)
    ax_zoom.set_title('Zoomed region')
    
    # Collect handles and labels from both axes
    handles1, labels1 = ax_zoom.get_legend_handles_labels()
    handles2, labels2 = ax_zoom2.get_legend_handles_labels()
    
    # Combine them
    ax_zoom.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    
    fig.tight_layout()
    plt.show()
    
    #%% Plot as a function of P(Hi)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.grid('on', alpha=0.6, linestyle='--')
    
    PHi_range = np.arange(0.01, 0.06, 0.01)
    PIR_tot_SS = np.zeros((len(gamma_grid_SS), len(PHi_range)))
    PIR_tot_Tq = np.zeros((len(gamma_grid_Tq), len(PHi_range)))
    colors = plt.rcParams["axes.prop_cycle"].by_key()['color']
    
    for idx, PHi in enumerate(PHi_range): 
        PH0 = 1 - PHi 
        
        IRH0_SS = PH0 * conditional_IRs_SS[:, 0]
        IRH0_Tq = PH0 * conditional_IRs_Tq[:, 0]
        
        IR_SS = IRH0_SS + PHi   * conditional_IRs_SS[:, 1]
        IR_Tq = IRH0_Tq + PHi   * conditional_IRs_Tq[:, 1]
    
        argsorted_SS = np.argsort(IR_SS)
        argsorted_Tq = np.argsort(IR_Tq)
    
        PIR_tot_SS[:, idx] = IR_SS
        PIR_tot_Tq[:, idx] = IR_Tq
        
        print("Max of IR(SS):", np.max(IR_SS))
        print("Max of IR(Tq):", np.max(IR_Tq))
        
        
        ax.plot(IR_SS[argsorted_SS], linestyle='--', color=colors[idx])
        ax.plot(IR_Tq[argsorted_Tq], 
                label=r'$P(H_i)={}$'.format(np.round(PHi, 3)), 
                linestyle='-', color=colors[idx])
    
    ax.set_xlabel('Grid point nr')
    ax.set_ylabel('Total IR')
    ax.set_title(r'Total IR for Tq and SS as function of $P(H_i)$')
    ax.set_xlim(0, len(IR_SS))
    ax.legend() 
    
    # # --- add zoomed inset ---
    # axins = inset_axes(ax, width="65%", height="30%", loc="center left",
    #                    bbox_to_anchor=(0.1, 0.2, 1, 1),
    #                    bbox_transform=ax.transAxes)
    
    # for idx, PHi in enumerate(PHi_range): 
    #     axins.plot(PIR_tot_SS[:, idx][np.argsort(PIR_tot_SS[:, idx])], linestyle='--', color=colors[idx])
    #     axins.plot(PIR_tot_Tq[:, idx][np.argsort(PIR_tot_Tq[:, idx])], linestyle='-', color=colors[idx])
    
    # axins.set_xlim(0, 350)
    # # axins.set_ylim(0.003, 0.004)
    # # axins.set_ylim(0.03, 0.065)
    # axins.set_ylim(0.008, 0.013)
    # axins.grid(True, linestyle="--", alpha=0.6)
    
    # # highlight zoom area on the main plot
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", lw=1)
    
    plt.tight_layout()
    plt.show()
    