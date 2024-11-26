#!/bin/bash

mkdir -p results/pendulum
mkdir -p results/cartpole
mkdir -p results/acrobot
mkdir -p results/mountain-car

GAMMA_VALUES="0.1 0.5 0.8 0.99"
ALPHA_ACTOR_VALUES="0.001"
ALPHA_CRITIC_VALUES="0.01"
EPISODES_EVAL=1
N_SEEDS=3

echo "Running Pendulum Experiment..."

EVAL_STEPS=5
MAX_STEPS=100
EXPERIMENT_NAME="pendulum"

python3 actor_critic_pendulum.py \
    --gamma_values $GAMMA_VALUES \
    --alpha_actor_values $ALPHA_ACTOR_VALUES \
    --alpha_critic_values $ALPHA_CRITIC_VALUES \
    --episodes_eval $EPISODES_EVAL \
    --eval_steps $EVAL_STEPS \
    --max_steps $MAX_STEPS \
    --n_seeds $N_SEEDS \
    --save_dir "results/pendulum" \
    --experiment_name $EXPERIMENT_NAME

echo "Pendulum Experiment Completed"

echo "-----------------------------------------------------------------------------------------"

echo "Running CartPole Experiment..."

EVAL_STEPS=5
MAX_STEPS=200
EXPERIMENT_NAME="cartpole"

python actor_critic_cartpole.py \
    --gamma_values $GAMMA_VALUES \
    --alpha_actor_values $ALPHA_ACTOR_VALUES \
    --alpha_critic_values $ALPHA_CRITIC_VALUES \
    --episodes_eval $EPISODES_EVAL \
    --eval_steps $EVAL_STEPS \
    --max_steps $MAX_STEPS \
    --n_seeds $N_SEEDS \
    --save_dir "results/cartpole" \
    --experiment_name $EXPERIMENT_NAME

echo "CartPole Experiment Completed"

echo "-----------------------------------------------------------------------------------------"

echo "Running Acrobot Experiment..."

EVAL_STEPS=5
MAX_STEPS=200
EXPERIMENT_NAME="acrobot"

python actor_critic_acrobot.py \
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

echo "-----------------------------------------------------------------------------------------"

echo "Running MoutainCar Experiment..."

EVAL_STEPS=5
MAX_STEPS=200
EXPERIMENT_NAME="mountain-car"

python actor_critic_mountain_car.py \
    --gamma_values $GAMMA_VALUES \
    --alpha_actor_values $ALPHA_ACTOR_VALUES \
    --alpha_critic_values $ALPHA_CRITIC_VALUES \
    --episodes_eval $EPISODES_EVAL \
    --eval_steps $EVAL_STEPS \
    --max_steps $MAX_STEPS \
    --n_seeds $N_SEEDS \
    --save_dir "results/mountain_car" \
    --experiment_name $EXPERIMENT_NAME

echo "Mountain Car Experiment Completed"

echo "========================================================================================="

echo "All experiments completed successfully!"

