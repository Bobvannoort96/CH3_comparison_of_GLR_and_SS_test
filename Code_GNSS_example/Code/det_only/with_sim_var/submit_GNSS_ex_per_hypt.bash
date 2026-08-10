#!/bin/bash
#SBATCH --job-name=ARAIM_PARALLEL
#SBATCH --partition=fat_genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48 # Increase this to match your node's capability
#SBATCH --time=00:45:00
#SBATCH --mem-per-cpu=6GB

#SBATCH --output=output/output_%j-%a.out

module load 2023
module load numba/0.58.1-foss-2023a
module load matplotlib/3.7.2-gfbf-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
hypt_to_model=$1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 
export NUMEXPR_NUM_THREADS=1

export NUMBA_NUM_THREADS=1


# One single execution that parallelizes internally
# python ./gnss_simulate_hypothesis.py 1
# python ./gnss_simulate_hypothesis_det_only.py --hypt 15 --N-sims 1 --nr-samples 1000 --q1-b-max 20 \
#	--q2-b1-max 0 --q2-b2-max 1
 
python ./main_code/gnss_simulate_hypothesis_det_only_with_sim_var.py --hypt $hypt_to_model --N-sims 10 --setup-nr 3
