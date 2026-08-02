# Decision-Making for Autonomous Mobile Robot using Reinforcement Learning

This repository contains materials related to a project on decision-making for autonomous mobile robots using Reinforcement Learning.

## Presentation

The project slides are available online and include embedded YouTube videos.

[Open the presentation](https://docs.google.com/presentation/d/1_MxhRH_voSYTWjdKaBzGfKcqtco66x2vCFCuTdZrZPk/present)

> Note: To watch the embedded videos, open the presentation with an internet connection.

## Project Overview

The goal of this project is to explore how Reinforcement Learning can be applied to autonomous mobile robots, enabling them to make decisions based on interaction with the environment.

## Topics Covered

- Autonomous mobile robots
- Reinforcement Learning
- Decision-making systems
- Robot navigation
- Learning through interaction

## Repository Content

- **[Case1](Case1/)** — tabular Q-learning on Gymnasium `Taxi-v3` (pure Python, no ROS).
- **[Case2](Case2/)** — continuous-control indoor navigation with **SAC** and **PPO**
  (stable-baselines3) on ROS 2 Humble + Gazebo Classic.
- Presentation slides (link above).

---

# Case 1 — Tabular Q-learning (Taxi-v3)

```bash
python Case1/taxi_q_learning.py train --episodes 5000 --visualize --viz-every 200
python Case1/taxi_q_learning.py test  --load q_table_taxi.npy --episodes 15 --visualize
```

---

# Case 2 — RL Navigation (ROS 2 Humble + Gazebo Classic)

A differential-drive robot with a 2D lidar learns to reach a goal pose inside an
indoor world while avoiding obstacles. Two algorithms are provided and share the
same environment, reward function and metrics, so they can be compared directly:

| | SAC | PPO |
|---|---|---|
| family | off-policy, replay buffer | on-policy, rollout buffer |
| sample efficiency | high | lower — needs more environment steps |
| key knobs | `buffer_size`, `learning_starts`, `tau`, `ent_coef="auto"` | `n_steps`, `n_epochs`, `clip_range`, `gae_lambda` |
| default budget here | 800k steps | 1.5M steps |
| policy network | `MlpPolicy`, `net_arch=[256, 256]` | `MlpPolicy`, `net_arch=[256, 256]` |
| checkpoints | `checkpoints/sac_*.zip` | `checkpoints/ppo_*.zip` |

Two workspaces are provided:

- **[Case2/rl_tutorial](Case2/rl_tutorial/)** — complete reference implementation (answer key).
- **[Case2/rl_curso](Case2/rl_curso/)** — student version: identical to `rl_tutorial`
  except for two marked blocks in
  [train_ppo.py](Case2/rl_curso/src/robot_navigation/robot_navigation/scripts/train_ppo.py)
  — the `PPO(...)` constructor (Exercise 1) and the `model.learn(...)` call (Exercise 2).
  Everything else — environment, reward, SAC, evaluation, plots — is already implemented.
  See [Case2/rl_curso/EXERCICIOS.md](Case2/rl_curso/EXERCICIOS.md).

All commands below are run from a workspace root (`Case2/rl_tutorial` or `Case2/rl_curso`).

## 1. Setup

```bash
chmod +x scripts/*.sh
./scripts/install_requeriments.sh      # ROS 2 Humble + Gazebo Classic + venv at ~/venvs/robot_nav_rl
```

Useful env vars: `TORCH_CPU_ONLY=1`, `SKIP_UPGRADE=1`, `FORCE_VENV_RECREATE=1`.

## 2. Build

Run `colcon` **outside** the venv (`ament_python` needs `setuptools 58.2.0`):

```bash
colcon build --symlink-install
source install/setup.bash
```

The `params/*.yaml` files are installed through `data_files`, so a YAML edit only
takes effect after a rebuild.

## 3. Run — two terminals

The simulator and the RL process are separate processes.

**Terminal 1 — simulation** (worlds: `small_house`, `bookstore`, `hospital`, `racetrack`):

```bash
ros2 launch robot_gazebo small_house.launch.py            # add gui:=false for headless training
ros2 run robot_gazebo teleop [--gui]                      # optional: drive manually to sanity-check
```

**Terminal 2 — training.** Source ROS first; the scripts activate the venv themselves
(`rclpy` lives in the ROS `PYTHONPATH`, not in the venv):

### Train with PPO

```bash
./scripts/train_ppo.sh
```

Equivalent direct call (edit any hyperparameter here):

```bash
python -m robot_navigation.scripts.train_ppo \
  --total-timesteps 1500000 --seed 1 --logdir runs/train --env-name smallhouse \
  --checkpoint-freq 150000 \
  --n-steps 2048 --batch-size 64 --n-epochs 10 \
  --lr 3e-4 --gae-lambda 0.95 --clip-range 0.2 --ent-coef 0.0 \
  --ros-args --params-file $(ros2 pkg prefix --share robot_navigation)/params/ros.yaml
```

PPO flags:

| flag | default | meaning |
|---|---|---|
| `--n-steps` | 2048 | transitions collected per policy update (rollout length) |
| `--batch-size` | 64 | minibatch size; **must divide `--n-steps`** |
| `--n-epochs` | 10 | optimization passes over each rollout |
| `--lr` | 3e-4 | Adam learning rate |
| `--gae-lambda` | 0.95 | GAE bias/variance trade-off |
| `--clip-range` | 0.2 | PPO policy-ratio clipping |
| `--ent-coef` | 0.0 | entropy bonus; raise (e.g. `0.005`) for more exploration |

`gamma=0.99`, `vf_coef=0.5`, `max_grad_norm=0.5` and `net_arch=[256, 256]` are fixed
inside the script.

### Train with SAC

```bash
./scripts/train_sac.sh
```

```bash
python -m robot_navigation.scripts.train_sac \
  --total-timesteps 800000 --seed 1 --logdir runs/train --env-name smallhouse \
  --checkpoint-freq 100000 --learning-starts 20000 \
  --lr 1e-4 --buffer-size 400000 --batch-size 256 \
  --ros-args --params-file $(ros2 pkg prefix --share robot_navigation)/params/ros.yaml
```

Everything after `--ros-args` is forwarded to `rclpy.init` untouched.
Run from the workspace root — `--logdir runs/train` is a relative path.

## 4. Evaluate and plot

Edit the run timestamp in the wrapper first, or call the module directly:

```bash
./scripts/eval_ppo.sh        # or ./scripts/eval_sac.sh
./scripts/plot_metrics.sh    # works for both algorithms — just point --logdir at the run
```

## 5. Run artifacts

Training writes to `runs/train/<env-name>/<YYYYmmdd-HHMMSS>/`:

| file | content |
|---|---|
| `monitor.csv` | SB3 `Monitor` — reward/length per episode |
| `metrics.csv` | per-episode success, collision, mean/final distance to goal |
| `tb/` | TensorBoard logs (`tensorboard --logdir runs/train`) |
| `checkpoints/` | periodic snapshots (`ppo_*.zip` / `sac_*.zip`) |
| `final_model.zip` | model at the end of training |

Evaluation writes `eval_metrics.csv` + `summary_eval.txt` next to the model it loaded;
`plot_metrics` writes `plots/*.png`.

## How to View the Slides

Click the presentation link above to open the slides in Google Slides.  
The videos are hosted on YouTube and should play directly from the online presentation.

## Author

Project developed by LRM — Laboratório de Robótica Móvel.
