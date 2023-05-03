import random
import time

import gym
import numpy as np
import torch
from gait_track_envs import register_env

from config import parse_args
from loggers import ConsoleLogger, FileLogger, MultiLogger, WandbLogger
from methods import CoIL, CoSIL


def main():
    config = parse_args()

    for _ in range(config.num_agents):
        config.experiment_id = str(int(time.time()))
        config.name = f"{config.env_name}-{str(config.seed)}-{config.experiment_id}"
        config.dir_path = (
            f"{config.project_name}/{config.group_name}/{config.env_name}/{config.seed}"
        )

        # Set up environment
        register_env(config.env_name)
        env = gym.make(config.env_name)

        # Seeding
        env.seed(config.seed)
        env.action_space.seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if config.device == "cuda":
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)

        # Set up the logger
        loggers_list = config.loggers.split(",")
        loggers = {}
        for logger in loggers_list:
            if logger == "console":
                loggers["console"] = ConsoleLogger()
            elif logger == "file":
                loggers["file"] = FileLogger(
                    config.project_name,
                    config.group_name,
                    config.experiment_id,
                )
            elif logger == "wandb":
                loggers["wandb"] = WandbLogger(
                    config.project_name,
                    config.group_name,
                    config.experiment_id,
                    config,
                )
            else:
                print(f'[WARNING] Logger "{logger}" is not supported')
        logger = MultiLogger(loggers)

        # Train a model using the selected training method
        print(f"Training using method {config.method}")
        if config.method == "CoIL":
            method = CoIL(config, logger, env)
        elif config.method == "CoSIL":
            method = CoSIL(config, logger, env)
        else:
            raise ValueError(f"Invalid training method: {config.method}")
        method.train()

        env.close()

        # Reset the seed for the next model
        config.seed = np.random.randint(low=1, high=np.iinfo(np.int32).max, size=1)[0]


if __name__ == "__main__":
    main()
