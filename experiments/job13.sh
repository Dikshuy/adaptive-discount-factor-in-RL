#!/bin/bash

mkdir -p results/q_learning

python q_learning.py \
    --environments EASY_SPARSE EASY_MEDIUM EASY_DENSE MODERATE_SPARSE MODERATE_MEDIUM MODERATE_DENSE DIFFICULT_SPARSE DIFFICULT_MEDIUM DIFFICULT_DENSE \
    --adaptive_gamma \
    --gamma_values 0.1 0.25 0.5 0.75 0.9 0.99 \
    --alpha_values 0.1 \
    --initial_values 0.0 \
    --max_steps 250000 \
    --eval_steps 100 \
    --n_seeds 50 \
    --save_dir "results/q_learning"
