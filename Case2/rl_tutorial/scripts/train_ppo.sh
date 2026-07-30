#!/usr/bin/env bash
set -euo pipefail

source "$HOME/venvs/robot_nav_rl/bin/activate"

python -m robot_navigation.scripts.train_ppo\
    --total-timesteps 1500000\
    --seed 1\
    --logdir runs/train\
    --checkpoint-freq 150000\
    --env-name smallhouse\
    --n-steps 2048\
    --batch-size 64\
    --n-epochs 10\
    --lr 0.0003\
    --gae-lambda 0.95\
    --clip-range 0.2\
    --ent-coef 0.0\
    --ros-args --params-file $(ros2 pkg prefix --share robot_navigation)/params/ros.yaml
