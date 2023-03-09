# Based on https://github.com/pranz24/pytorch-soft-actor-critic (MIT Licensed)

import time

import gait_track_envs  # pyright: reportUnusedImport=false
import gym
import numpy as np
import torch
from coil import CoIL
from config import parse_args

import wandb


def main():
    args = parse_args()

    np.random.seed(args.seed)

    for _ in range(args.num_agents):
        args.run_id = str(int(time.time()))
        args.name = f'{args.run_name}-{args.env_name}-{str(args.seed)}-{args.run_id}'
        args.group = f'{args.run_name}-{args.env_name}'
        args.dir_path = f'{args.run_name}/{args.env_name}/{args.seed}'

        # Set up wandb
        if args.use_wandb:
            wandb.init(project=args.project_name,
                       name=args.name,
                       group=args.group,
                       config=args)

        # Set up environment
        env = gym.make(args.env_name)
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)

        coil = CoIL(env, args)
        coil.train()

        env.close()

        # Reset the seed for the next model
        args.seed = np.random.randint(1)


if __name__ == "__main__":
    main()
