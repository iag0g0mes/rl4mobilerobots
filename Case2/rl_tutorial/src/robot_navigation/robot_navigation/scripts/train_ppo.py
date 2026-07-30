import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import datetime
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from robot_navigation.envs import IndoorEnv, RosInterface
from robot_navigation.envs.utils import print_config_rich

from .callbacks import EpisodeMetricsCallback

@dataclass
class RunningParams:
    total_timesteps:int=200000
    seed:int=0
    logdir:str="~/rl/checkpoints"
    ckp_freq:int=10000
    env_name:str="smallhouse"
    n_steps:int=2048
    batch_size:int=64
    n_epochs:int=10
    lr:float=3e-4
    gae_lambda:float=0.95
    clip_range:float=0.2
    ent_coef:float=0.0


def parse_args() -> RunningParams:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="~/rl/checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=25000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--env-name", type=str, default="smallhouse")
    args, ros_args = parser.parse_known_args()

    params = RunningParams()
    params.total_timesteps = int(args.total_timesteps)
    params.seed = int(args.seed)
    params.logdir = str(args.logdir)
    params.ckp_freq = int(args.checkpoint_freq)
    params.env_name = str(args.env_name)
    params.n_steps = int(args.n_steps)
    params.batch_size = int(args.batch_size)
    params.n_epochs = int(args.n_epochs)
    params.lr = float(args.lr)
    params.gae_lambda = float(args.gae_lambda)
    params.clip_range = float(args.clip_range)
    params.ent_coef = float(args.ent_coef)

    # PPO splits each rollout of n_steps transitions into minibatches of
    # batch_size; a non-divisible split leaves a truncated last minibatch
    if params.n_steps % params.batch_size != 0:
        raise ValueError(
            f"n_steps ({params.n_steps}) must be a multiple of batch_size ({params.batch_size})"
        )

    data_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    params.logdir = os.path.join(params.logdir, params.env_name, data_str)
    os.makedirs(params.logdir, exist_ok=True)

    print_config_rich(params)

    return params, ros_args

def main():

    print("[Reinforcement Learning Navigation][train] running...")
    params, ros_args = parse_args()

    print("[Reinforcement Learning Navigation][train] creating ROS interface...")
    ros = RosInterface(args=ros_args)

    print("[Reinforcement Learning Navigation][train] creating Env interface...")
    env = None
    match params.env_name:
        case "smallhouse":
            env = IndoorEnv(
                    ros    = ros,
                    params = params
                )
        case _:
            raise ValueError(f"Environment type unknown!! {params.env_name}")

    assert env is not None, f"Failed to create the environment: {params.env_name}"

    print("[Reinforcement Learning Navigation][train] creating logger...")
    env = Monitor(env, filename=os.path.join(params.logdir, "monitor.csv"), allow_early_resets=True)
    tb_dir = os.path.join(params.logdir, "tb")
    new_logger = configure(tb_dir, ["stdout", "tensorboard"])


    print("[Reinforcement Learning Navigation][train] creating PPO...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=params.lr,
        n_steps=params.n_steps,
        batch_size=params.batch_size,
        n_epochs=params.n_epochs,
        gamma=0.99,
        gae_lambda=params.gae_lambda,
        clip_range=params.clip_range,
        ent_coef=params.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=dict(net_arch=[256,256]),
        device="auto",
        seed=params.seed,
        verbose=1,
        tensorboard_log=tb_dir,
    )
    model.set_logger(new_logger)


    ckpt_dir = os.path.join(params.logdir, "checkpoints")
    callbacks = [
        CheckpointCallback(save_freq=params.ckp_freq, save_path=ckpt_dir, name_prefix="ppo"),
        EpisodeMetricsCallback(csv_path=os.path.join(params.logdir, "metrics.csv")),
    ]

    try:
        model.learn(total_timesteps=params.total_timesteps, callback=callbacks, progress_bar=True)
        model.save(os.path.join(params.logdir, "final_model.zip"))
        print(f"[Reinforcement Learning Navigation][train] Saved final model to: {os.path.join(params.logdir, 'final_model.zip')}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
