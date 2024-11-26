#!/bin/bash
#SBATCH --time=1-02:00:00
#SBATCH --output=results-%j.out
#SBATCH --account=def-mtaylor3
#SBATCH --cpus-per-task=2

GAMMA_VALUES="0.1 0.5 0.8 0.99"
ALPHA_ACTOR_VALUES="0.001"
ALPHA_CRITIC_VALUES="0.01"
EPISODES_EVAL=10
EVAL_STEPS=500
MAX_STEPS=1000000
N_SEEDS=30

python3 actor_critic.py \
    --gamma_values $GAMMA_VALUES \
    --alpha_actor_values $ALPHA_ACTOR_VALUES \
    --alpha_critic_values $ALPHA_CRITIC_VALUES \
    --episodes_eval $EPISODES_EVAL \
    --eval_steps $EVAL_STEPS \
    --max_steps $MAX_STEPS \
    --n_seeds $N_SEEDS
