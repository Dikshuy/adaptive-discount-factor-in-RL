#!/bin/bash

mkdir -p results/alpha/q_learning

python q_learning.py \
    --gamma_values 0.99 \
    --alpha_values 0.05 0.1 0.5 \
    --initial_values 0.0 10.0 \
    --max_steps 200000 \
    --eval_steps 100 \
    --n_seeds 50 \
    --save_dir "results/alpha/q_learning" \
