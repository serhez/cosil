import random
import time

import gym
import hydra
import numpy as np
import torch
from gait_track_envs import register_env
from omegaconf import DictConfig

from config import setup_config
from loggers import ConsoleLogger, FileLogger, MultiLogger, WandbLogger
from methods import CoIL, CoSIL


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    for _ in range(config.num_agents):
        config.logger.experiment_id = str(int(time.time()))
        config.models_dir_path = f"{config.logger.project_name}/{config.logger.group_name}/{config.env_name}/{config.seed}"

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
        loggers_list = config.logger.loggers.split(",")
        loggers = {}
        for logger in loggers_list:
            if logger == "console":
                loggers["console"] = ConsoleLogger()
            elif logger == "file":
                loggers["file"] = FileLogger(
                    config.logger.project_name,
                    config.logger.group_name,
                    config.logger.experiment_id,
                )
            elif logger == "wandb":
                loggers["wandb"] = WandbLogger(
                    config.logger.project_name,
                    config.logger.group_name,
                    config.logger.experiment_id,
                    config,
                )
            else:
                print(f'[WARNING] Logger "{logger}" is not supported')
        logger = MultiLogger(loggers)

        # Train a model using the selected training method
        logger(f"Training using method {config.method.name}", "INFO", ["wandb"])
        if config.method.name == "coil":
            method = CoIL(config, logger, env)
        elif config.method.name == "cosil":
            method = CoSIL(config, logger, env)
        else:
            raise ValueError(f"Invalid training method: {config.method.name}")
        method.train()

        env.close()

        # Reset the seed for the next model
        config.seed = np.random.randint(low=1, high=np.iinfo(np.int32).max, size=1)[0]


if __name__ == "__main__":
    setup_config()
    main()