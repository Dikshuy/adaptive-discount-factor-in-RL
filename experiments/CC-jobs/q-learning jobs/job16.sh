#!/bin/bash
#SBATCH --account=def-mtaylor3_cpu
#SBATCH --mem-per-cpu=8G
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

mkdir -p results2/q_learning/adaptive
mkdir -p results2/q_learning/non_adaptive

python q_learning_2.py \
    --environments "Gym-Gridworlds/Straight-20-v0" "Gym-Gridworlds/Empty-2x2-v0" "Gym-Gridworlds/Empty-3x3-v0" "Gym-Gridworlds/Empty-Loop-3x3-v0" "Gym-Gridworlds/Empty-10x10-v0" "Gym-Gridworlds/Empty-Distract-6x6-v0" "Gym-Gridworlds/Penalty-3x3-v0" "Gym-Gridworlds/Quicksand-4x4-v0" "Gym-Gridworlds/Quicksand-Distract-4x4-v0" "Gym-Gridworlds/TwoRoom-Quicksand-3x5-v0" "Gym-Gridworlds/Corridor-3x4-v0" "Gym-Gridworlds/Full-4x5-v0" "Gym-Gridworlds/TwoRoom-Distract-Middle-2x11-v0" "Gym-Gridworlds/Barrier-5x5-v0" "Gym-Gridworlds/RiverSwim-6-v0" "Gym-Gridworlds/CliffWalk-4x12-v0" "Gym-Gridworlds/DangerMaze-6x6-v0" \
    --adaptive_gamma \
    --gamma_values 0.1 0.25 0.5 0.75 0.9 0.99 \
    --alpha_values 0.1 \
    --initial_values 0.0 \
    --max_steps 50000 \
    --eval_steps 100 \
    --n_seeds 50 \
    --save_dir "results2/q_learning/adaptive"

python q_learning_2.py \
    --environments "Gym-Gridworlds/Straight-20-v0" "Gym-Gridworlds/Empty-2x2-v0" "Gym-Gridworlds/Empty-3x3-v0" "Gym-Gridworlds/Empty-Loop-3x3-v0" "Gym-Gridworlds/Empty-10x10-v0" "Gym-Gridworlds/Empty-Distract-6x6-v0" "Gym-Gridworlds/Penalty-3x3-v0" "Gym-Gridworlds/Quicksand-4x4-v0" "Gym-Gridworlds/Quicksand-Distract-4x4-v0" "Gym-Gridworlds/TwoRoom-Quicksand-3x5-v0" "Gym-Gridworlds/Corridor-3x4-v0" "Gym-Gridworlds/Full-4x5-v0" "Gym-Gridworlds/TwoRoom-Distract-Middle-2x11-v0" "Gym-Gridworlds/Barrier-5x5-v0" "Gym-Gridworlds/RiverSwim-6-v0" "Gym-Gridworlds/CliffWalk-4x12-v0" "Gym-Gridworlds/DangerMaze-6x6-v0" \
    --gamma_values 0.1 0.25 0.5 0.75 0.9 0.99 \
    --alpha_values 0.1 \
    --initial_values 0.0 \
    --max_steps 50000 \
    --eval_steps 100 \
    --n_seeds 50 \
    --save_dir "results2/q_learning/non_adaptive/"
