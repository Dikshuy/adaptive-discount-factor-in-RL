#!/bin/bash
#SBATCH --time=1-02:00:00
#SBATCH --output=results-%j.out
#SBATCH --account=def-mtaylor3
#SBATCH --cpus-per-task=2

source rl-project/bin/activate
cd adaptive-discount-factor-in-RL/experiments/
python reinforce_cartpole.py
deactivate