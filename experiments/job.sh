#!/bin/bash
#SBATCH --account=def-mtaylor3_cpu
#SBATCH --mem-per-cpu=4G
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

mkdir -p results/pendulum
mkdir -p results/cartpole
mkdir -p results/mountain_car
mkdir -p results/acrobot

GAMMA_VALUES="0.1 0.5 0.8 0.99"
ALPHA_ACTOR_VALUES="0.001"
ALPHA_CRITIC_VALUES="0.01"
EPISODES_EVAL=10
N_SEEDS=30

echo "Running Pendulum Experiment..."

EVAL_STEPS=500
MAX_STEPS=1000000
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
echo ""

echo "Running CartPole Experiment..."

EVAL_STEPS=100
MAX_STEPS=20000
EXPERIMENT_NAME="cartpole"

python3 actor_critic_cartpole.py \
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
echo ""

echo "Running MoutainCar Experiment..."

EVAL_STEPS=500
MAX_STEPS=1000000
EXPERIMENT_NAME="mountain-car"

python3 actor_critic_mountain_car.py \
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

echo "-----------------------------------------------------------------------------------------"
echo ""

echo "Running Acrobot Experiment..."

EVAL_STEPS=500
MAX_STEPS=1000000
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
echo ""

echo "========================================================================================="

echo "All experiments completed successfully!"

