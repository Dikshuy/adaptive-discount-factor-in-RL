#!/bin/bash
#SBATCH --account=def-mtaylor3_cpu
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

mkdir -p results/alpha/cartpole

GAMMA_VALUES="0.95"
ALPHA_ACTOR_VALUES="0.001 0.005 0.01"
ALPHA_CRITIC_VALUES="0.001 0.005 0.01"
EPISODES_EVAL=10
N_SEEDS=10

echo "Running CartPole Experiment..."

EVAL_STEPS=500
MAX_STEPS=100000
EXPERIMENT_NAME="cartpole"

python3 actor_critic_cartpole.py \
    --gamma_values $GAMMA_VALUES \
    --alpha_actor_values $ALPHA_ACTOR_VALUES \
    --alpha_critic_values $ALPHA_CRITIC_VALUES \
    --episodes_eval $EPISODES_EVAL \
    --eval_steps $EVAL_STEPS \
    --max_steps $MAX_STEPS \
    --n_seeds $N_SEEDS \
    --save_dir "results/alpha/cartpole" \
    --experiment_name $EXPERIMENT_NAME

echo "CartPole Experiment Completed"

echo "========================================================================================="