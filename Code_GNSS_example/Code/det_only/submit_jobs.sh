#!/bin/bash

# Define m
m=11

# Compute k = m + (m*(m-1))/2
k_alts=$(( m + (m * (m - 1)) / 2 ))



# Double loop over nr_hypts and b_values
for nr_hypts in $(seq 0 $k_alts); do
    
    sbatch submit_GNSS_ex_per_hypt.bash $nr_hypts 
done


echo "All combinations processed!"
