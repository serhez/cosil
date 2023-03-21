import time

import gym
import numpy as np
import torch
import wandb
from gait_track_envs import register_env

from methods import CoAL, CoIL, CoSIL
from config import parse_args


def main():
    args = parse_args()

    for _ in range(args.num_agents):
        np.random.seed(args.seed)

        args.run_id = str(int(time.time()))
        args.name = f"{args.run_name}-{args.env_name}-{str(args.seed)}-{args.run_id}"
        args.group = f"{args.run_name}-{args.env_name}"
        args.dir_path = f"{args.run_name}/{args.env_name}/{args.seed}"

        # Set up wandb
        if args.use_wandb:
            wandb.init(
                project=args.project_name, name=args.name, group=args.group, config=args
            )

        # Set up environment
        register_env(args.env_name)
        env = gym.make(args.env_name)
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)

        # Train a model using the selected training method
        if args.method == "CoIL":
            method = CoIL(env, args)
        elif args.method == "CoSIL":
            method = CoSIL(env, args)
        elif args.method == "CoAL":
            method = CoAL(env, args)
        else:
            raise ValueError(f"Invalid training method: {args.method}")
        method.train()

        env.close()

        # Reset the seed for the next model
        args.seed = np.random.randint(1)


if __name__ == "__main__":
    main()
