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

# NOTE: Set config.resume to the filepath of the model to be used to record the video
#       config.method.record_test must equal True
#       make sure to also set record_path and storage_path to something sensible


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    config.logger.run_id = str(int(time.time()))
    config.models_dir_path = f"{config.env_name}/seed-{config.seed}"
    if config.logger.group_name != "":
        config.models_dir_path = f"{config.logger.group_name}/" + config.models_dir_path

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
    logger = create_multilogger(config)

    # Train a model using the selected training method
    logger.info(f"Recording video for env {config.env_name}")
    method = create_method(config, logger, env)
    try:
      if hasattr(method, 'morphos'):
        for i in range(len(method.morphos) - 1):
            method._evaluate(i, method.morphos[i], {})
    except Exception as e:
        logger.error({"Exception occurred during video recording": e})
        raise e

    # Reset everything for the next model
    env.close()


if __name__ == "__main__":
    setup_config()
    main()
