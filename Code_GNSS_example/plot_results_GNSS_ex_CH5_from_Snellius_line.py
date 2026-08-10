# -*- coding: utf-8 -*-
"""
Plots and computes the probability of positioning failure (IR) from the data / 
simulation obtained from the Snellius HPC. 

The directory structure of the data is 

<maindir>/Data_GNSS_example/<det_only or det_and_ident>/setup_<setup_nr>/N_sims_<N_sims>/

Within this directory, the structure is one more 

AL_<AL_radii>/H_<hypt>/

The filenames are 
- gamma_SS.txt (only for det_only) 
- gamma_Tq.txt (only for det_only) 
- IR_SS.txt 
- IR_Tq.txt 

Note that the IRs are stored as conditioned on the hypothesis. I.e. to get the total IR, we need to 
multiply by the probability of the hypothesis. 

"""

import sys
import os
import numpy as np 

import matplotlib.pyplot as plt 

if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind': lambda x: f"{x:.3e}"})
    np.random.seed(1)
     

    # Values of visible satellites at 12.00 UTC in Delft on July 1 2025
    elevation = np.array(
        [
            82.3,
            50.7,
            70.3,
            18.8,
            8.4,
            9.5,
            44.9,
            22.7,
            16.4,
            14.2,
            5.8,
            21.4,
            49.7,
            17.0,
            36.0,
            65.5,
            32.2,
            56.4,
            16.4,
            17.4,
            13.4,
            6.2,
        ]
    )

    azimuth = np.array(
        [
            109.6,
            121.6,
            247.7,
            182.8,
            170.7,
            261.6,
            293.9,
            314.9,
            278.7,
            79.7,
            106.3,
            43.1,
            87.2,
            37.5,
            161.1,
            297.6,
            185.5,
            103.3,
            260.3,
            312.0,
            290.4,
            108.2,
        ]
    )

    sys_array = np.array(
        [
            "GPS-1",
            "GPS-2",
            "GPS-3",
            "GPS-4",
            "GPS-8",
            "GPS-14",
            "GPS-17",
            "GPS-19",
            "GPS-22",
            "GPS-28",
            "GPS-31",
            "GPS-32",
            "GAL-3",
            "GAL-5",
            "GAL-8",
            "GAL-13",
            "GAL-14",
            "GAL-15",
            "GAL-21",
            "GAL-23",
            "GAL-26",
            "GAL-34",
        ]
    )

    consider_all_satellites = False  # set to true if all satellites should be chosen

    if not consider_all_satellites:
        n_keep = 11
    else:
        n_keep = 22  # try with all satellites
        print("Warning: now, we do not delete any of the m=22 satellites")
        print("Hence also factor_to_multiply should alter")

    # Indices to keep
    keep_idx = np.sort(np.random.choice(len(sys_array), size=n_keep, replace=False))
  
    # Resulting arrays
    az_used = azimuth[keep_idx]
    el_used = elevation[keep_idx]
    sys_used = sys_array[keep_idx]

    # --- Step 2: Extract only GPS/GAL system labels ---
    sys_labels = np.array([s.split("-")[0] for s in sys_used])

    m = len(el_used) 
    n = 5 # hardcoded, three coordaintes and two clock offsets 
     

    nc_GPS = np.sum(sys_labels == "GPS")  # nr of GPS sats
    nc_GAL = np.sum(sys_labels == "GAL")  # nr of GAL sats
 
 

    n_UP = 2

    

    # statistical parameters -- for the simulations
    k = int(m + m * (m - 1) / 2)  # total number of fault hypotheses
    # AL_Bx = 5.0 # meters. 
    AL_Bx = 2.5 # meters.
    # AL_Bx = 1.5

    AL_Bx_list = [1.5, 2.5, 3.0]
    ## Setup nr is based on the one provided here
    setup_nr = 2
    if setup_nr == 1:
        psat_GPS = 1e-3
        psat_GAL = 3e-3
        alpha = 0.01
        if not consider_all_satellites:
            factor_convert_alpha = 1.795
        else:
            factor_convert_alpha = 1.68
    elif setup_nr == 2:
        psat_GPS = 1e-3
        psat_GAL = 3e-3
        alpha = 0.001
        
        if not consider_all_satellites:
            factor_convert_alpha = 1.65
        else:
            factor_convert_alpha = 1.65   
    elif setup_nr == 3:
        psat_GPS = 1e-3
        psat_GAL = 3e-3
        alpha = 0.1
        if not consider_all_satellites:
            factor_convert_alpha = 1.790
        else:
            factor_convert_alpha = 1.85  
    else:
        raise NotImplementedError(f"Setup nr {setup_nr} is not provided")

     

    PHi_GPS = psat_GPS * (1 - psat_GPS) ** (nc_GPS - 1) * (1 - psat_GAL) ** nc_GAL
    PHi_GAL = psat_GAL * (1 - psat_GAL) ** (nc_GAL - 1) * (1 - psat_GPS) ** nc_GPS

    PHi_GPS_GPS = (
        psat_GPS**2 * (1 - psat_GPS) ** (nc_GPS - 2) * (1 - psat_GAL) ** nc_GAL
    )
    PHi_GAL_GAL = (
        psat_GAL**2 * (1 - psat_GAL) ** (nc_GAL - 2) * (1 - psat_GPS) ** nc_GPS
    )
    PHi_GPS_GAL = (
        psat_GAL
        * psat_GPS
        * (1 - psat_GAL) ** (nc_GAL - 1)
        * (1 - psat_GPS) ** (nc_GPS - 1)
    )

    n_comb_GPS = int(nc_GPS * (nc_GPS - 1) / 2)
    n_comb_GAL = int(nc_GAL * (nc_GAL - 1) / 2)
    n_comb_GPS_GAL = int(m * (m - 1) / 2) - n_comb_GPS - n_comb_GAL

    PH0 = (
        1
        - nc_GPS * PHi_GPS
        - nc_GAL * PHi_GAL
        - n_comb_GPS * PHi_GPS_GPS
        - n_comb_GAL * PHi_GAL_GAL
        - n_comb_GPS_GAL * PHi_GPS_GAL
    )

    PHi_list = []
    ci_vectors = []
    Pcti_list = []
    cti_plus_list = []
    Qxi_list = []
    Abar_plus_list = []
    for i in np.arange(m):
         
        if sys_labels[i] == "GPS":
            P_of_alt = PHi_GPS
        else:
            P_of_alt = PHi_GAL
        PHi_list.append(P_of_alt)

    for i in np.arange(m):
        for j in np.arange(i + 1, m):
            # All the q=1 hypotheses
            ci = np.eye(m)[:, [i, j]]
            
            idxes, _ = np.where(ci.astype(bool))
            if idxes[0] < nc_GPS and idxes[1] < nc_GPS:
                P_of_alt = PHi_GPS_GPS
            elif idxes[0] < nc_GPS or idxes[1] < nc_GPS:
                P_of_alt = PHi_GPS_GAL
            else:
                P_of_alt = PHi_GAL_GAL
            PHi_list.append(P_of_alt)

    
    ##########################################
    # Load the data and insert the relevant paramters 
    ############################
    maindir = r'C:/Users/bgvannoort/Documents/Dissertation/Chapter 5 SS vs GLR/'
    
    scenario_type = 'det_and_ident' # can be det_only or det_and_ident

    setup_nr = 1
    N_sims=10
    
 
    topdir = rf"Data_GNSS_example/Snellius/with_sim_var/b_sim_1D/{scenario_type}/setup_{setup_nr}/N_sims_{N_sims}/"
 
        
    
    
    store_max_IR_SS = np.zeros((len(AL_Bx_list), k+1)) 
    store_max_IR_Tq = np.zeros((len(AL_Bx_list), k+1) )
    
    store_std_max_IR_SS = np.zeros((len(AL_Bx_list), k+1))
    store_std_max_IR_Tq = np.zeros((len(AL_Bx_list), k+1))
    
    
    # Per-AL total-IR curves vs b, plus the H0 and max scalars, for plotting.
    # Filled inside the loop below; b_grid captured from the first hypt > 0.
    b_grid = None
    n_b = None
    
    PIR_tot_SS_curve = {}      # AL -> array over b (sum_i PHi * IR_i(b)) + IR|H0
    PIR_tot_Tq_curve = {}
    std_PIR_tot_SS_curve = {}  # AL -> array over b (propagated std of the total)
    std_PIR_tot_Tq_curve = {}
    
    IR_PH0_SS_store = {}       # AL -> scalar (already conditional * PH0)
    IR_PH0_Tq_store = {}
    std_IR_PH0_SS_store = {}
    std_IR_PH0_Tq_store = {}
    
    max_IR_SS_store = {}       # AL -> scalar (sum over hypts of max_b PHi*IR_i)
    max_IR_Tq_store = {}
    std_max_IR_SS_store = {}
    std_max_IR_Tq_store = {}
    
    for idx_al, AL in enumerate(AL_Bx_list):
        
        for hypt in np.arange(0, k+1):
            
            resDir =rf'AL_{AL}/H_{hypt}'
            fullresDir = os.path.join(maindir, topdir, resDir)
            
            
            
            if hypt == 0: 
                # No bvalues present
                Prob_H = PH0
                PIR_SS_H = np.loadtxt(os.path.join(fullresDir, 'IR_PH0_SS.txt'), delimiter=',')
                PIR_Tq_H = np.loadtxt(os.path.join(fullresDir, 'IR_PH0_Tq.txt'), delimiter=',')
                
                
                # There is only one value, since not dependent on b-value, so immediately name it the max_std
                sigma_max_IR_SS = np.loadtxt(os.path.join(fullresDir, 'std_IR_PH0_SS.txt'), delimiter=',')
                sigma_max_IR_Tq = np.loadtxt(os.path.join(fullresDir, 'std_IR_PH0_Tq.txt'), delimiter=',')
                
                alpha_obs_SS = np.loadtxt(os.path.join(fullresDir, 'alpha_obs_SS.txt'), delimiter=',')
                alpha_obs_Tq = np.loadtxt(os.path.join(fullresDir, 'alpha_obs_Tq.txt'), delimiter=',')
                
                # There is only one value, since not dependent on b-value, so immediately name it the max_std
                std_alpha_obs_SS = np.loadtxt(os.path.join(fullresDir, 'std_alpha_obs_SS.txt'), delimiter=',')
                std_alpha_obs_Tq = np.loadtxt(os.path.join(fullresDir, 'std_alpha_obs_Tq.txt'), delimiter=',')
                
                IR_PH0_SS_store[AL] = PIR_SS_H * Prob_H
                IR_PH0_Tq_store[AL] = PIR_Tq_H * Prob_H
                std_IR_PH0_SS_store[AL] = sigma_max_IR_SS * Prob_H
                std_IR_PH0_Tq_store[AL] = sigma_max_IR_Tq * Prob_H

                # H0 seeds the running total (constant in b; broadcast later)
                run_tot_SS = PIR_SS_H * Prob_H
                run_tot_Tq = PIR_Tq_H * Prob_H
                run_var_SS = (sigma_max_IR_SS * Prob_H) ** 2
                run_var_Tq = (sigma_max_IR_Tq * Prob_H) ** 2
                
            else:
                # Bias is array
                bval1 = np.loadtxt(os.path.join(fullresDir, 'b_values.txt'), delimiter=',')
                Prob_H = PHi_list[hypt-1]
                
                PIR_SS_H = np.loadtxt(os.path.join(fullresDir, 'IR_SS.txt'), delimiter=',')
                PIR_Tq_H = np.loadtxt(os.path.join(fullresDir, 'IR_Tq.txt'), delimiter=',')
                
                std_PIR_SS_H = np.loadtxt(os.path.join(fullresDir, 'std_IR_SS.txt'), delimiter=',')
                std_PIR_Tq_H = np.loadtxt(os.path.join(fullresDir, 'std_IR_Tq.txt'), delimiter=',')
                
                
                idx_0_max_SS = np.argmax(PIR_SS_H)
                idx_0_max_Tq = np.argmax(PIR_Tq_H)
                
                sigma_max_IR_SS = std_PIR_SS_H[idx_0_max_SS]
                sigma_max_IR_Tq = std_PIR_Tq_H[idx_0_max_Tq] 
                
                if b_grid is None:
                    b_grid = bval1
                    n_b = len(b_grid)
                    # promote the H0 seeds (scalars) to b-length arrays
                    run_tot_SS = run_tot_SS * np.ones(n_b)
                    run_tot_Tq = run_tot_Tq * np.ones(n_b)
                    run_var_SS = run_var_SS * np.ones(n_b)
                    run_var_Tq = run_var_Tq * np.ones(n_b)
                run_tot_SS += Prob_H * PIR_SS_H
                run_tot_Tq += Prob_H * PIR_Tq_H
                run_var_SS += (Prob_H * std_PIR_SS_H) ** 2
                run_var_Tq += (Prob_H * std_PIR_Tq_H) ** 2

                
            max_IR_SS = np.max(PIR_SS_H) * Prob_H 
            max_IR_Tq = np.max(PIR_Tq_H) * Prob_H 
            
            
            
            store_max_IR_SS[idx_al, hypt] = max_IR_SS 
            store_max_IR_Tq[idx_al, hypt] = max_IR_Tq 
            
            store_std_max_IR_SS[idx_al, hypt] = sigma_max_IR_SS*Prob_H
            store_std_max_IR_Tq[idx_al, hypt] = sigma_max_IR_Tq*Prob_H
            
        PIR_tot_SS_curve[AL] = run_tot_SS
        PIR_tot_Tq_curve[AL] = run_tot_Tq
        std_PIR_tot_SS_curve[AL] = np.sqrt(run_var_SS)
        std_PIR_tot_Tq_curve[AL] = np.sqrt(run_var_Tq)

        # max IR = H0 contribution + sum over hypts of max_b (PHi * IR_i(b))
        max_IR_SS_store[AL] = np.sum(store_max_IR_SS[idx_al, :])
        max_IR_Tq_store[AL] = np.sum(store_max_IR_Tq[idx_al, :])
        std_max_IR_SS_store[AL] = np.sqrt(np.sum(store_std_max_IR_SS[idx_al, :] ** 2))
        std_max_IR_Tq_store[AL] = np.sqrt(np.sum(store_std_max_IR_Tq[idx_al, :] ** 2))
    
    print("Max PPF values for the SS test ")
    print("For AL values of ", *AL_Bx_list)
    print(np.sum(store_max_IR_SS, axis=1)) 
    print(r"$\pm$", np.sum(store_std_max_IR_SS, axis=1))
    
    print(4*"-------------")
    print(4*"-------------")
    
    print("Max PPF values for the Tq test ")
    print("For AL values of ", *AL_Bx_list)
    print(np.sum(store_max_IR_Tq, axis=1)) 
    print(r"$\pm$", np.sum(store_std_max_IR_Tq, axis=1))
    
    
#%%
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amssymb, amsmath}",
        "font.family": "serif",
        "axes.labelsize": 18,    # x/y axis labels
        "xtick.labelsize": 16,   # x-tick numbers
        "ytick.labelsize": 16,   # y-tick numbers
    })
    k_sigma = 1.0   # band half-width in std units (1 = ±1 std), as on the blue line
    band_alpha = 0.15

    for idx_al, AL in enumerate(AL_Bx_list):
        b = b_grid
        figIR, axIR = plt.subplots(figsize=(16, 10))

        # ---- total IR vs b (blue): solid = GLR, dashed = SS, shaded band ----
        axIR.plot(b, PIR_tot_Tq_curve[AL], color="blue",
                  label=r"$\mathbb{P}_{\mathcal{F}}^{\text{GLR}}(\mathbf{b})$", linewidth=2.5)
        axIR.fill_between(
            b,
            PIR_tot_Tq_curve[AL] - k_sigma * std_PIR_tot_Tq_curve[AL],
            PIR_tot_Tq_curve[AL] + k_sigma * std_PIR_tot_Tq_curve[AL],
            color="blue", alpha=band_alpha,
        )
        axIR.plot(b, PIR_tot_SS_curve[AL], color="blue", linestyle="--",
                  label=r"$\mathbb{P}_{\mathcal{F}}^{\text{SS}}(\mathbf{b})$", linewidth=2.5)
        axIR.fill_between(
            b,
            PIR_tot_SS_curve[AL] - k_sigma * std_PIR_tot_SS_curve[AL],
            PIR_tot_SS_curve[AL] + k_sigma * std_PIR_tot_SS_curve[AL],
            color="blue", alpha=band_alpha,
        )
        err_step = 3   # show an error bar every 5 points

        # axIR.errorbar(
        #     b,
        #     PIR_tot_Tq_curve[AL],
        #     yerr=k_sigma * std_PIR_tot_Tq_curve[AL],
        #     color="blue",
        #     linestyle="-",
        #     linewidth=2.5,
        #     capsize=4,
        #     errorevery=err_step,
        #     label=r"$\mathbb{P}_{\mathcal{F}}^{\text{GLR}}(\mathbf{b})$"
        # )
        
        # axIR.errorbar(
        #     b,
        #     PIR_tot_SS_curve[AL],
        #     yerr=k_sigma * std_PIR_tot_SS_curve[AL],
        #     color="blue",
        #     linestyle="--",
        #     linewidth=2.5,
        #     capsize=4,
        #     errorevery=err_step+1,
        #     label=r"$\mathbb{P}_{\mathcal{F}}^{\text{SS}}(\mathbf{b})$"
        # )

        # ---- IR under H0 (green): flat line + shaded band ----
        axIR.axhline(IR_PH0_Tq_store[AL], color="green", linestyle="-",
                     label=r"$(\mathbb{P}_{\mathcal{F}}^{\text{GLR}} | \mathcal{H}_0)\, P(\mathcal{H}_0)$", linewidth=2.5)
        axIR.axhspan(IR_PH0_Tq_store[AL] - k_sigma * std_IR_PH0_Tq_store[AL],
                     IR_PH0_Tq_store[AL] + k_sigma * std_IR_PH0_Tq_store[AL],
                     color="green", alpha=band_alpha)
        axIR.axhline(IR_PH0_SS_store[AL], color="green", linestyle="--",
                     label=r"$(\mathbb{P}_{\mathcal{F}}^{\text{SS}} | \mathcal{H}_0)\, P(\mathcal{H}_0)$", linewidth=2.5)
        axIR.axhspan(IR_PH0_SS_store[AL] - k_sigma * std_IR_PH0_SS_store[AL],
                     IR_PH0_SS_store[AL] + k_sigma * std_IR_PH0_SS_store[AL],
                     color="green", alpha=band_alpha)

        # # H0 contribution
        # y_glr_h0 = np.full_like(b, IR_PH0_Tq_store[AL], dtype=float)
        # y_ss_h0  = np.full_like(b, IR_PH0_SS_store[AL], dtype=float)
        
        # axIR.errorbar(
        #     b,
        #     y_glr_h0,
        #     yerr=k_sigma * std_IR_PH0_Tq_store[AL],
        #     color="green",
        #     linestyle="-",
        #     linewidth=2.5,
        #     capsize=4,
        #     errorevery=err_step,
        #     label=r"$(\mathbb{P}_{\mathcal{F}}^{\text{GLR}} | \mathcal{H}_0)\, P(\mathcal{H}_0)$"
        # )
        
        # axIR.errorbar(
        #     b,
        #     y_ss_h0,
        #     yerr=k_sigma * std_IR_PH0_SS_store[AL],
        #     color="green",
        #     linestyle="--",
        #     linewidth=2.5,
        #     capsize=4,
        #     errorevery=err_step+1,
        #     label=r"$(\mathbb{P}_{\mathcal{F}}^{\text{SS}} | \mathcal{H}_0)\, P(\mathcal{H}_0)$"
        # )

        # ---- max IR (red): flat line + shaded band ----
        axIR.axhline(max_IR_Tq_store[AL], color="red", linestyle="-",
                     label=r"$\max_{\mathbf{b}}\, \mathbb{P}_{\mathcal{F}}^{\text{GLR}}(\mathbf{b})$", linewidth=2.5)
        axIR.axhspan(max_IR_Tq_store[AL] - k_sigma * std_max_IR_Tq_store[AL],
                     max_IR_Tq_store[AL] + k_sigma * std_max_IR_Tq_store[AL],
                     color="red", alpha=band_alpha)
        axIR.axhline(max_IR_SS_store[AL], color="red", linestyle="--",
                     label=r"$\max_{\mathbf{b}}\, \mathbb{P}_{\mathcal{F}}^{\text{SS}}(\mathbf{b})$", linewidth=2.5)
        axIR.axhspan(max_IR_SS_store[AL] - k_sigma * std_max_IR_SS_store[AL],
                     max_IR_SS_store[AL] + k_sigma * std_max_IR_SS_store[AL],
                     color="red", alpha=band_alpha)
        
        
        y_glr_max = np.full_like(b, max_IR_Tq_store[AL], dtype=float)
        y_ss_max  = np.full_like(b, max_IR_SS_store[AL], dtype=float)
        
        # axIR.errorbar(
        #     b,
        #     y_glr_max,
        #     yerr=k_sigma * std_max_IR_Tq_store[AL],
        #     color="red",
        #     linestyle="-",
        #     linewidth=2.5,
        #     capsize=4,
        #     errorevery=err_step,
        #     label=r"$\max_{\mathbf{b}}\, \mathbb{P}_{\mathcal{F}}^{\text{GLR}}(\mathbf{b})$"
        # )
        
        # axIR.errorbar(
        #     b,
        #     y_ss_max,
        #     yerr=k_sigma * std_max_IR_SS_store[AL],
        #     color="red",
        #     linestyle="--",
        #     linewidth=2.5,
        #     capsize=4,
        #     errorevery=err_step+1,
        #     label=r"$\max_{\mathbf{b}}\, \mathbb{P}_{\mathcal{F}}^{\text{SS}}(\mathbf{b})$"
        # )
        

        # ---- axes, grid, labels ----
        axIR.grid(True, alpha=0.3)
        axIR.set_xlim(0, np.max(b))
        axIR.set_xlabel(r"$ b_i $ [m]", fontsize=18)
        axIR.set_ylabel(r"$\mathbb{P}_{\mathcal{F}}$ [-]", fontsize=18)
        # figIR.suptitle(
        #     rf"{scenario_type} $\mathbb{{P}}_{{\mathcal{{F}}}}$ as a function of $b_i$, $\ell = {AL}$"
        # )
        axIR.legend(fontsize=18, loc="upper right")
        figIR.tight_layout()
        
        
        dirN = rf"C:\Users\bgvannoort\Documents\Dissertation\Chapter 5 SS vs GLR\Figures\gnss_example\{scenario_type}\setup {setup_nr}"
        # ---- Save data for this AL_Bx ----
        if not consider_all_satellites:
            saveFigData = os.path.join(dirN, f"AL={AL} m")
        else:
            saveFigData = os.path.join(dirN, "all_sats",  f"AL={AL} m")

        os.makedirs(saveFigData, exist_ok=True)
        
        # # Save the figures. 
        figIR.savefig(os.path.join(saveFigData, "PF_vs_b.pdf"))
        figIR.savefig(os.path.join(saveFigData, "PF_vs_b.png"))
        
        
        
        # figIR.savefig(os.path.join(<your dir>, f"PF_vs_b_AL_{AL}.pdf"))
     
