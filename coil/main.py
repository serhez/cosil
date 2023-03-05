# Based on https://github.com/pranz24/pytorch-soft-actor-critic (MIT Licensed)

import argparse

import gym
import gait_track_envs # pyright: reportUnusedImport=false
import numpy as np
import torch

import wandb
from coilobj import CoIL


def main():
    args = parse_args()

    # Set up wandb
    wandb.init(name=args.algo, config=args)

    # Set up environment
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    coil = CoIL(env, args)
    coil.train()

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="GAIL + SAC + co-adaptation")
    parser.add_argument(
        "--env-name",
        default="GaitTrackHalfCheetahOriginal-v0",
        help="Mujoco Gym environment",
    )
    parser.add_argument("--algo", default="GAIL", help="Algorithm GAIL or SAIL or PWIL")
    parser.add_argument(
        "--expert-demos",
        type=str,
        default="data/expert_demos_sampled_GaitTrackHalfCheetah-v0.pt",
        help="Path to the expert demonstration file",
    )
    parser.add_argument(
        "--policy",
        default="Gaussian",
        help="Policy Type: Gaussian | Deterministic (default: Gaussian)",
    )
    parser.add_argument(
        "--eval",
        type=bool,
        default=True,
        help="Evaluates a policy a policy every 10 episode (default: True)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor for reward (default: 0.99)",
    )
    parser.add_argument(
        "--target_entropy",
        type=str,
        default="auto",
        metavar="G",
        help="Target value for entropy",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        metavar="G",
        help="target smoothing coefficient(τ) (default: 0.005)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        metavar="G",
        help="learning rate (default: 0.0003)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        metavar="G",
        help="Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)",
    )
    parser.add_argument(
        "--automatic_entropy_tuning",
        type=bool,
        default=False,
        metavar="G",
        help="Automaically adjust α (default: False)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        metavar="N",
        help="random seed (default: 123456)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="batch size (default: 256)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000001,
        metavar="N",
        help="maximum number of steps (default: 1000000)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        metavar="N",
        help="hidden size (default: 256)",
    )
    parser.add_argument(
        "--updates_per_step",
        type=int,
        default=1,
        metavar="N",
        help="model updates per simulator step (default: 1)",
    )
    parser.add_argument(
        "--start_steps",
        type=int,
        default=10000,
        metavar="N",
        help="Steps sampling random actions (default: 10000)",
    )
    parser.add_argument(
        "--target_update_interval",
        type=int,
        default=1,
        metavar="N",
        help="Value target update per no. of updates per step (default: 1)",
    )
    parser.add_argument(
        "--replay_size",
        type=int,
        default=2000000,
        metavar="N",
        help="size of replay buffer (default: 20000000)",
    )
    parser.add_argument(
        "--cuda", action="store_true", help="run on CUDA (default: False)"
    )
    parser.add_argument("--run-name", default="gail", help="Run name (logging only)")
    parser.add_argument(
        "--log-scale-rewards",
        action="store_true",
        help="Use sigmoid directly as reward or log of sigmoid",
    )
    parser.add_argument(
        "--reward-style",
        type=str,
        default="GAIL",
        help="AIRL-style or GAIL-style reward",
    )
    parser.add_argument(
        "--train-every", type=int, default=1, help="Train every N timesteps"
    )
    parser.add_argument(
        "--explore-morpho-episodes",
        type=int,
        default=800,
        help="Episodes to run morphology exploration for",
    )
    parser.add_argument(
        "--morpho-warmup",
        type=int,
        default=60000,
        help="Steps before starting to optimize for morphology",
    )
    parser.add_argument(
        "--episodes-per-morpho",
        type=int,
        default=5,
        help="Episodes to run of each morphology",
    )
    parser.add_argument(
        "--disc-warmup",
        type=int,
        default=20000,
        help="Steps before starting to train SAC",
    )
    parser.add_argument(
        "--record-test", action="store_true", help="Record tests (may be slow)"
    )
    parser.add_argument(
        "--load-warmup", action="store_true", help="Load previously saved warmup data"
    )
    parser.add_argument(
        "--q-weight-decay", type=float, default=1e-5, help="Q-function weight decay"
    )
    parser.add_argument(
        "--disc-weight-decay",
        type=float,
        default=1e-5,
        help="Discriminator weight decay",
    )
    parser.add_argument(
        "--vae-scaler",
        type=float,
        default=1.0,
        help="Scaling term for VAE loss in SAIL",
    )
    parser.add_argument(
        "--pos-type",
        type=str,
        default="norm",
        choices=["abs", "rel", "norm", "skip"],
        help="Which position marker coordinate to use (absolute, relative, normalized-relative, or skip to omit it)",
    )
    parser.add_argument(
        "--vel-type",
        type=str,
        default="rel",
        choices=["abs", "rel", "norm", "skip"],
        help="Which velocity marker coordinate to use (absolute, relative, normalized-relative, or skip to omit it)",
    )
    parser.add_argument(
        "--expert-legs",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Which legs to use for marker matching on the demonstrator side",
    )
    parser.add_argument(
        "--policy-legs",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Which legs to use for marker matching on the imitator side",
    )
    parser.add_argument(
        "--expert-markers",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Which markers to use for matching on the demonstrator side",
    )
    parser.add_argument(
        "--policy-markers",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Which markers to use for matching on the imitator side",
    )
    parser.add_argument(
        "--learn-disc-transitions",
        action="store_true",
        help="Learn discriminator using s, s' transitions",
    )
    parser.add_argument(
        "--train-distance-value",
        action="store_true",
        help="Learn a separate distance value which is used to optimize morphology",
    )
    parser.add_argument(
        "--co-adapt", action="store_true", help="Adapt morphology as well as behaviour"
    )
    parser.add_argument(
        "--expert-env-name", type=str, default=None, help="Expert env name"
    )
    parser.add_argument(
        "--subject-id",
        type=int,
        default=8,
        help="Expert subject name when using CMU dataset",
    )
    parser.add_argument(
        "--expert-episode-length",
        type=int,
        default=300,
        help="Episode length for non-mocap expert data",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from given policy"
    )
    parser.add_argument(
        "--torso-type",
        type=str,
        default=None,
        nargs="+",
        help="Use torso velocity, position or skip",
    )
    parser.add_argument(
        "--head-type",
        type=str,
        default=None,
        nargs="+",
        help="Use head velocity, position or skip",
    )
    parser.add_argument(
        "--head-wrt",
        type=str,
        default=None,
        nargs="+",
        help="Use head with respect to body part (torso, butt)",
    )
    parser.add_argument(
        "--absorbing-state",
        action="store_true",
        help="Replace terminal states with special absorbing states",
    )
    parser.add_argument(
        "--omit-done",
        action="store_true",
        help="Simply set done=False always for learning purposes. Alternative to absorbing states.",
    )
    parser.add_argument(
        "--save-morphos",
        action="store_true",
        help="Save morphology parameters and corresponding Wasserstein distances for later",
    )
    parser.add_argument(
        "--dist-optimizer",
        default=None,
        choices=["BO", "CMA", "RS"],
        help="Co-adapt for Wasserstein distance, and optimize using algo.",
    )
    parser.add_argument(
        "--bo-gp-mean", choices=["Zero", "Constant", "Linear"], default="Zero"
    )
    parser.add_argument(
        "--acq-weight",
        type=float,
        default=2.0,
        help="BO LCB acquisition function exploration weight",
    )
    parser.add_argument("--fixed-morpho", nargs="+", default=None, type=float)
    parser.add_argument(
        "--normalize-obs", action="store_true", help="Normalize observations for critic"
    )
    parser.add_argument(
        "--save-checkpoints", action="store_true", help="Save buffer and models"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
