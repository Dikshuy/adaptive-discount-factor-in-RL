#!/bin/bash
#SBATCH --account=def-mtaylor3_cpu
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


mkdir -p test/pendulum

GAMMA_VALUES="0.95"
ALPHA_ACTOR_VALUES="0.001 0.01"
ALPHA_CRITIC_VALUES="0.001 0.01"
EPISODES_EVAL=2
N_SEEDS=5
EVAL_STEPS=50
MAX_STEPS=1000

echo "Running Pendulum Experiment..."

EXPERIMENT_NAME="pendulum"

python3 actor_critic_pendulum.py \
    --gamma_values $GAMMA_VALUES \
    --alpha_actor_values $ALPHA_ACTOR_VALUES \
    --alpha_critic_values $ALPHA_CRITIC_VALUES \
    --episodes_eval $EPISODES_EVAL \
    --eval_steps $EVAL_STEPS \
    --max_steps $MAX_STEPS \
    --n_seeds $N_SEEDS \
    --save_dir "test/pendulum" \
    --experiment_name $EXPERIMENT_NAME

echo "Pendulum Experiment Completed"
echo ""

echo "========================================================================================="

mkdir -p test/cartpole

echo "Running CartPole Experiment..."

EXPERIMENT_NAME="cartpole"

python3 actor_critic_cartpole.py \
    --gamma_values $GAMMA_VALUES \
    --alpha_actor_values $ALPHA_ACTOR_VALUES \
    --alpha_critic_values $ALPHA_CRITIC_VALUES \
    --episodes_eval $EPISODES_EVAL \
    --eval_steps $EVAL_STEPS \
    --max_steps $MAX_STEPS \
    --n_seeds $N_SEEDS \
    --save_dir "test/cartpole" \
    --experiment_name $EXPERIMENT_NAME

echo "CartPole Experiment Completed"
echo ""

echo "========================================================================================="

mkdir -p test/mountain_car

echo "Running MoutainCar Experiment..."

EXPERIMENT_NAME="mountain-car"

python3 actor_critic_mountain_car.py \
    --gamma_values $GAMMA_VALUES \
    --alpha_actor_values $ALPHA_ACTOR_VALUES \
    --alpha_critic_values $ALPHA_CRITIC_VALUES \
    --episodes_eval $EPISODES_EVAL \
    --eval_steps $EVAL_STEPS \
    --max_steps $MAX_STEPS \
    --n_seeds $N_SEEDS \
    --save_dir "test/mountain_car" \
    --experiment_name $EXPERIMENT_NAME

echo "Mountain Car Experiment Completed"
echo ""

echo "========================================================================================="

mkdir -p test/acrobot

echo "Running Acrobot Experiment..."

EXPERIMENT_NAME="acrobot"

python3 actor_critic_acrobot.py \
    --gamma_values $GAMMA_VALUES \
    --alpha_actor_values $ALPHA_ACTOR_VALUES \
    --alpha_critic_values $ALPHA_CRITIC_VALUES \
    --episodes_eval $EPISODES_EVAL \
    --eval_steps $EVAL_STEPS \
    --max_steps $MAX_STEPS \
    --n_seeds $N_SEEDS \
    --save_dir "test/acrobot" \
    --experiment_name $EXPERIMENT_NAME

echo "Acrobot Experiment Completed"
echo ""

echo "========================================================================================="

echo "All experiments completed successfully!"

