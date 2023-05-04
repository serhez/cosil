import os
import random
import time

import gym
import hydra
import numpy as np
import torch
from gait_track_envs import register_env
from omegaconf import DictConfig

from agents import SAC
from config import setup_config
from loggers import ConsoleLogger, FileLogger, MultiLogger, WandbLogger
from rewarders import EnvReward
from utils.model import load_model


def add_obs(obs_list, info, done):
    obs_list["dones"] = np.append(
        obs_list["dones"],
        done,
    )

    for key, val in info.items():
        if key not in obs_list:
            obs_list[key] = np.empty((0, *val.shape))

        assert (
            len(obs_list[key]) == len(obs_list["dones"]) - 1
        ), "Observations must yield the same info"

        obs_list[key] = np.append(obs_list[key], np.array([val]), axis=0)


def gen_obs(config, logger, env):
    assert config.resume is not None, "Must provide model path to generate observations"

    obs_list = {
        "dones": np.array([]),
    }

    obs_size = env.observation_space.shape[0]
    if config.absorbing_state:
        obs_size += 1

    rewarder = EnvReward(config)
    agent = SAC(
        config,
        logger,
        obs_size,
        env.action_space,
        env.morpho_params.shape[0],
        len(env.morpho_params),
        rewarder,
    )

    load_model(config.resume, env, agent, co_adapt=False, evaluate=True)

    # Generate observations
    for episode in range(config.method.num_episodes):
        state, _ = env.reset()
        feat = np.concatenate([state, env.morpho_params])
        if config.absorbing_state:
            feat = np.concatenate([feat, np.zeros(1)])

        tot_reward = 0
        num_obs = 0
        done = False
        while not done:
            action = agent.select_action(feat, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            next_feat = np.concatenate([next_state, env.morpho_params])
            if config.absorbing_state:
                next_feat = np.concatenate([next_feat, np.zeros(1)])

            add_obs(obs_list, info, terminated or truncated)
            num_obs += 1

            feat = next_feat

            tot_reward += reward
            done = terminated or truncated

        logger(
            {
                "Episode": episode,
                "Reward": tot_reward,
                "Generated observations": num_obs,
            },
            "INFO",
            ["wandb"],
        )

    return obs_list


def save(obs_list: dict, path: str):
    assert path is not None, "Must provide path to save observations"
    try:
        split_path = os.path.split(path)
        dir_path = split_path[0]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except Exception:
        raise ValueError("Invalid path")

    torch.save(obs_list, path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
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
    loggers_list = config.loggers.split(",")
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

    obs = gen_obs(config, logger, env)
    env.close()

    save(obs, config.obs_save_path)


if __name__ == "__main__":
    setup_config()
    main()
