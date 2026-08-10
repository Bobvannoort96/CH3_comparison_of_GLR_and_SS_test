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
    
    scenario_type = 'det_only' # can be det_only or det_and_ident

    setup_nr = 1
    N_sims=10
    
    # scenario_type = 'det_only'
    bool_with_sim_var = True
    
    if bool_with_sim_var:
        topdir = rf"Data_GNSS_example/Snellius/with_sim_var/{scenario_type}/setup_{setup_nr}/N_sims_{N_sims}/"
    else:
        topdir = rf"Data_GNSS_example/Snellius/{scenario_type}/setup_{setup_nr}/N_sims_{N_sims}/"
    
    
    
    store_max_IR_SS = np.zeros((len(AL_Bx_list), k+1)) 
    store_max_IR_Tq = np.zeros((len(AL_Bx_list), k+1) )
    
    store_std_max_IR_SS = np.zeros((len(AL_Bx_list), k+1))
    store_std_max_IR_Tq = np.zeros((len(AL_Bx_list), k+1))
    
    
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
            elif hypt <= m: 
                # Bias is array
                bval1 = np.loadtxt(os.path.join(fullresDir, 'b_values.txt'), delimiter=',')
                Prob_H = PHi_list[hypt-1]
                
                PIR_SS_H = np.loadtxt(os.path.join(fullresDir, 'IR_SS.txt'), delimiter=',')
                PIR_Tq_H = np.loadtxt(os.path.join(fullresDir, 'IR_Tq.txt'), delimiter=',')
                
                std_PIR_SS_H = np.loadtxt(os.path.join(fullresDir, 'std_IR_SS.txt'), delimiter=',')
                std_PIR_Tq_H = np.loadtxt(os.path.join(fullresDir, 'std_IR_Tq.txt'), delimiter=',')
                
                
                idx_0_max_SS = np.argmax(PIR_SS_H)
                idx_0_max_Tq = np.argmax(PIR_Tq_H)
                
                sigma_max_IR_SS = std_PIR_SS_H[idx_0_max_SS]*Prob_H
                sigma_max_IR_Tq = std_PIR_Tq_H[idx_0_max_Tq]*Prob_H
                
                
            else:
                # bias is 2d array / a grid
                bval1 = np.loadtxt(os.path.join(fullresDir, 'b1_grid.txt'), delimiter=',')
                bval2 = np.loadtxt(os.path.join(fullresDir, 'b2_grid.txt'), delimiter=',')
                Prob_H = PHi_list[hypt-1]
                
                PIR_SS_H = np.loadtxt(os.path.join(fullresDir, 'IR_SS.txt'), delimiter=',')
                PIR_Tq_H = np.loadtxt(os.path.join(fullresDir, 'IR_Tq.txt'), delimiter=',')
                
                std_PIR_SS_H = np.loadtxt(os.path.join(fullresDir, 'std_IR_SS.txt'), delimiter=',')
                std_PIR_Tq_H = np.loadtxt(os.path.join(fullresDir, 'std_IR_Tq.txt'), delimiter=',')
                
                
                idx_0_max_SS, idx_1_max_SS = np.unravel_index(np.argmax(PIR_SS_H), PIR_SS_H.shape)
                idx_0_max_Tq, idx_1_max_Tq = np.unravel_index(np.argmax(PIR_Tq_H), PIR_Tq_H.shape)
                
                sigma_max_IR_SS = std_PIR_SS_H[idx_0_max_SS, idx_1_max_SS]*Prob_H
                sigma_max_IR_Tq = std_PIR_Tq_H[idx_0_max_Tq, idx_1_max_Tq]*Prob_H
            
            max_IR_SS = np.max(PIR_SS_H) * Prob_H 
            max_IR_Tq = np.max(PIR_Tq_H) * Prob_H 
            
            
            
            store_max_IR_SS[idx_al, hypt] = max_IR_SS 
            store_max_IR_Tq[idx_al, hypt] = max_IR_Tq 
            
            store_std_max_IR_SS[idx_al, hypt] = sigma_max_IR_SS
            store_std_max_IR_Tq[idx_al, hypt] = sigma_max_IR_Tq
    
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
    fig, ax = plt.subplots(1,2, figsize=(18,10)) 
    max_IR = np.max( (PIR_SS_H, PIR_Tq_H))
    
    pcm_SS= ax[0].pcolormesh(
            bval1,
            bval2,
            PIR_SS_H,
            vmin=0,
            vmax=max_IR,
        )
    ax[0].set_xlabel("$b_1$")
    ax[0].set_ylabel("$b_2$")
    ax[0].set_title("SS test")
    
    pcm_Tq= ax[1].pcolormesh(
            bval1,
            bval2,
            PIR_Tq_H,
            vmin=0,
            vmax=max_IR,
        )
    
    ax[1].set_xlabel("$b_1$")
    ax[1].set_ylabel("$b_2$")
    ax[1].set_title("GLR test")
    
    
