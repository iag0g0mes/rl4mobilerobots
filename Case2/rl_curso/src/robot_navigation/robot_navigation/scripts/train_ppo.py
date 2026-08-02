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

    # ╔══════════════════════════════════════════════════════════════════════════╗
    # ║  EXERCÍCIO 1 — INSTANCIAR O AGENTE PPO                                   ║
    # ║                                                                          ║
    # ║  Troque cada  ...  pelo valor correto.                                   ║
    # ║  Os hiperparâmetros já chegam prontos no objeto `params`                 ║
    # ║  (ver a dataclass RunningParams no topo deste arquivo).                  ║
    # ║  As linhas marcadas [OK] já estão preenchidas          ║
    # ║                                                                          ║
    # ║  Doc: stable-baselines3.readthedocs.io/en/master/modules/ppo.html        ║
    # ╚══════════════════════════════════════════════════════════════════════════╝
    model = PPO(
        policy=...,                # (str)   política MLP para observações vetoriais
        env=...,                   # (Env)   ambiente de treino, já embrulhado no Monitor
        learning_rate=...,         # (float) passo do otimizador Adam
        n_steps=...,               # (int)   transições coletadas por rollout
        batch_size=...,            # (int)   minibatch do SGD (precisa dividir n_steps)
        n_epochs=...,              # (int)   passadas de otimização sobre o mesmo rollout
        gamma=0.99,                # [OK]    fator de desconto
        gae_lambda=...,            # (float) trade-off viés/variância do GAE
        clip_range=...,            # (float) clipping da razão de probabilidades
        ent_coef=...,              # (float) peso do bônus de entropia (exploração)
        vf_coef=0.5,               # [OK]    peso da perda da value function
        max_grad_norm=0.5,         # [OK]    clipping da norma do gradiente
        normalize_advantage=True,  # [OK]    normaliza a vantagem por minibatch
        policy_kwargs=dict(net_arch=[256,256]),  # [OK] duas camadas ocultas de 256
        device="auto",             # [OK]    GPU se houver, senão CPU
        seed=...,                  # (int)   semente para reprodutibilidade
        verbose=1,                 # [OK]    log de treino no stdout
        tensorboard_log=...,       # (str)   diretório dos logs do TensorBoard
    )
    # ────────────────────────── FIM DO EXERCÍCIO 1 ─────────────────────────────

    model.set_logger(new_logger)


    ckpt_dir = os.path.join(params.logdir, "checkpoints")
    callbacks = [
        CheckpointCallback(save_freq=params.ckp_freq, save_path=ckpt_dir, name_prefix="ppo"),
        EpisodeMetricsCallback(csv_path=os.path.join(params.logdir, "metrics.csv")),
    ]

    # ╔══════════════════════════════════════════════════════════════════════════╗
    # ║  EXERCÍCIO 2 — DISPARAR O TREINAMENTO                                    ║
    # ║                                                                          ║
    # ║  Chame model.learn(...) passando:                                        ║
    # ║    • o orçamento total de passos de ambiente (vem de `params`)           ║
    # ║    • a lista `callbacks` criada logo acima                               ║
    # ║    • a barra de progresso ligada                                         ║
    # ║                                                                          ║
    # ║  Doc: stable-baselines3.readthedocs.io/en/master/modules/ppo.html        ║
    # ╚══════════════════════════════════════════════════════════════════════════╝
    try:
        model.learn(...) 
        # ──────────────────────── FIM DO EXERCÍCIO 2 ───────────────────────────

        model.save(os.path.join(params.logdir, "final_model.zip"))
        print(f"[Reinforcement Learning Navigation][train] Saved final model to: {os.path.join(params.logdir, 'final_model.zip')}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
