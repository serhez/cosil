import random
import time

import gym
import hydra
import numpy as np
import torch
from gait_track_envs import register_env
from omegaconf import DictConfig

from config import setup_config
from loggers import create_multilogger
from methods import create_method


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    for _ in range(config.num_agents):
        config.logger.run_id = str(int(time.time()))
        config.models_dir_path = f"{config.env_name}/seed-{config.seed}"
        if config.logger.group_name != "":
            config.models_dir_path = (
                f"{config.logger.group_name}/" + config.models_dir_path
            )

        # Set up environment
        register_env(config.env_name)
        env = gym.make(config.env_name)
        env.mode = "rbg_array"

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
        logger = create_multilogger(config)

        # Train a model using the selected training method
        logger.info(f"Training using method {config.method.name}")
        method = create_method(config, logger, env)
        try:
            method.train()
        except Exception as e:
            logger.error({"Exception occurred during training": e})
            raise e

        # Reset everything for the next model
        env.close()
        config.seed = np.random.randint(low=1, high=np.iinfo(np.int32).max)


if __name__ == "__main__":
    setup_config()
    main()
