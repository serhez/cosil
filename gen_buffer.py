import os
import random
import time

import gym
import hydra
import numpy as np
import torch
from gait_track_envs import register_env
from omegaconf import DictConfig

from common.observation_buffer import ObservationBuffer
from config import setup_config
from loggers import ConsoleLogger, FileLogger, MultiLogger, WandbLogger
from methods import CoIL


def save(buffer: ObservationBuffer, path: str, id: str):
    """
    Saves the buffer to a file.

    Parameters
    ----------
    buffer -> the observation buffer.
    path -> the path to save the buffer.
    id -> the id of the buffer.
    """

    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception:
        raise ValueError("Invalid path")

    buffer_path = os.path.join(path, f"buffer_{id}.pt")

    torch.save(buffer.to_list(), buffer_path)

    return buffer_path


@hydra.main(version_base=None, config_path="configs", config_name="gen_buffer")
def main(config: DictConfig):
    for _ in range(config.num_agents):
        config.logger.run_id = str(int(time.time()))
        config.models_dir_path = f"{config.env_name}/seed-{config.seed}"
        if config.logger.group_name != "":
            config.models_dir_path = (
                f"{config.logger.group_name}/" + config.models_dir_path
            )
        if config.logger.project_name != "":
            config.models_dir_path = (
                f"{config.logger.project_name}/" + config.models_dir_path
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
        loggers_list = config.logger.loggers.split(",")
        loggers = {}
        for logger in loggers_list:
            if logger == "console":
                loggers["console"] = ConsoleLogger()
            elif logger == "file":
                loggers["file"] = FileLogger(
                    config.logger.project_name,
                    config.logger.group_name,
                    config.logger.experiment_name,
                    config.logger.run_id,
                )
            elif logger == "wandb":
                loggers["wandb"] = WandbLogger(
                    config.logger.project_name,
                    config.logger.group_name,
                    config.logger.experiment_name,
                    config.logger.run_id,
                    config,
                )
            elif logger == "":
                pass
            else:
                print(f'[WARNING] Logger "{logger}" is not supported')
        logger = MultiLogger(loggers, config.logger.default_mask.split(","))

        # Train a model using the selected training method
        method = CoIL(config, logger, env)

        try:
            method.train()
        except Exception as e:
            logger.error({"Exception occurred during training": e})
            raise e

        env.close()

        # Save the buffer
        buffer_path = save(method.replay_buffer, config.save_path, config.logger.run_id)
        logger.info(f"Saved buffer to {buffer_path}")

        # Reset the seed for the next model
        config.seed = np.random.randint(low=1, high=np.iinfo(np.int32).max)


if __name__ == "__main__":
    setup_config()
    main()
