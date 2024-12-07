#!/bin/bash
#SBATCH --account=def-mtaylor3_cpu
#SBATCH --mem-per-cpu=8G
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

mkdir -p results/acrobot

GAMMA_VALUES="0.1 0.25 0.5 0.75 0.9 0.95 0.99"
ALPHA_ACTOR_VALUES="0.001"
ALPHA_CRITIC_VALUES="0.01"
EPISODES_EVAL=10
N_SEEDS=15

echo "Running Acrobot Experiment..."

EVAL_STEPS=500
MAX_STEPS=500000
EXPERIMENT_NAME="acrobot"

python3 actor_critic_acrobot.py \
    --gamma_values $GAMMA_VALUES \
    --alpha_actor_values $ALPHA_ACTOR_VALUES \
    --alpha_critic_values $ALPHA_CRITIC_VALUES \
    --episodes_eval $EPISODES_EVAL \
    --eval_steps $EVAL_STEPS \
    --max_steps $MAX_STEPS \
    --n_seeds $N_SEEDS \
    --save_dir "results/acrobot" \
    --experiment_name $EXPERIMENT_NAME

echo "Acrobot Experiment Completed"

echo "========================================================================================="