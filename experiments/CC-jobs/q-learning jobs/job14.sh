#!/bin/bash
#SBATCH --account=def-mtaylor3_cpu
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

mkdir -p results/q_learning/adaptive
mkdir -p results/q_learning/non_adaptive

python q_learning.py \
    --environments EASY_SPARSE EASY_MEDIUM EASY_DENSE MODERATE_SPARSE MODERATE_MEDIUM MODERATE_DENSE DIFFICULT_SPARSE DIFFICULT_MEDIUM DIFFICULT_DENSE \
    --adaptive_gamma \
    --gamma_values 0.1 0.25 0.5 0.75 0.9 0.99 \
    --alpha_values 0.1 \
    --initial_values 10.0 \
    --max_steps 500000 \
    --eval_steps 500 \
    --n_seeds 30 \
    --save_dir "results/q_learning/adaptive"

python q_learning.py \
    --environments EASY_SPARSE EASY_MEDIUM EASY_DENSE MODERATE_SPARSE MODERATE_MEDIUM MODERATE_DENSE DIFFICULT_SPARSE DIFFICULT_MEDIUM DIFFICULT_DENSE \
    --gamma_values 0.1 0.25 0.5 0.75 0.9 0.99 \
    --alpha_values 0.1 \
    --initial_values 10.0 \
    --max_steps 500000 \
    --eval_steps 500 \
    --n_seeds 30 \
    --save_dir "results/q_learning/non_adaptive/"
