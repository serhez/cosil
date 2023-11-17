import os
import random
import time
from typing import List

import gym
import hydra
import numpy as np
import torch
from gait_track_envs import register_env
from omegaconf import DictConfig

from agents import SAC
from common.schedulers import ConstantScheduler
from config import setup_config
from loggers import ConsoleLogger, FileLogger, Logger, MultiLogger, WandbLogger
from rewarders import EnvReward
from utils.model import load_model
from utils.rl import gen_obs_dict


def gen_model_obs(
    config: DictConfig, env: gym.Env, logger: Logger, logger_mask: List[str] = ["wandb"]
) -> dict:
    """
    Generates observations using the trained model.

    The result is a dictionary containing each dimension of the observations as keys and
    a list of that dimension's values for each observation as values.

    Note that the trajectories are flattened, so that each dict item contains the total number of observations.

    Parameters
    ----------
    config -> the configuration.
    env -> the environment.
    logger -> the logger.
    logger_mask -> the loggers to mask when logging.

    Returns
    -------
    obs_dict -> the dictionary containing the observations.
    """

    assert config.resume is not None, "Must provide model path to generate trajectories"

    obs_size = env.observation_space.shape[0]
    if config.absorbing_state:
        obs_size += 1

    rewarder = EnvReward(config.device)
    agent = SAC(
        config,
        logger,
        env.action_space,
        obs_size + env.morpho_params.shape[0] if config.morpho_in_state else obs_size,
        env.morpho_params.shape[0],
        rewarder,
        None,
        ConstantScheduler(0.0),
    )

    load_model(
        config.resume,
        env,
        agent,
        agent_name=config.saved_agent_name,
        morpho_name=config.saved_morpho_name,
        co_adapt=True,
        evaluate=True,
    )

    return gen_obs_dict(
        config.num_obs,
        env,
        agent,
        config.morpho_in_state,
        config.absorbing_state,
        logger,
        logger_mask,
    )


def save(obs: dict, path: str, env_name: str, id: str, logger: Logger):
    """
    Saves the observations to a file.

    Parameters
    ----------
    obs -> the observations.
    path -> the path to save the observations.
    id -> the id of the observations.
    logger -> the logger.
    """

    assert path is not None, "Must provide path to save observations"
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception:
        raise ValueError("Invalid path")

    if path[-1] != "/":
        path += "/"

    file_name = f"{path}{env_name}_demos_{id}.pt"

    logger.info(f"Saving demonstrations to {file_name}")
    torch.save(obs, file_name)


@hydra.main(version_base=None, config_path="configs", config_name="gen_obs")
def main(config: DictConfig):
    config.logger.run_id = str(int(time.time()))
    config.models_dir_path = f"{config.env_name}/{config.seed}"
    if config.logger.group_name != "":
        config.models_dir_path = f"{config.logger.group_name}/" + config.models_dir_path
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

    obs = gen_model_obs(config, env, logger)
    env.close()

    save(obs, config.save_path, config.env_name, config.logger.run_id, logger)


if __name__ == "__main__":
    setup_config()
    main()
