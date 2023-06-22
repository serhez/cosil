import os
import random
import time

import gym
import hydra
import numpy as np
import torch
from gait_track_envs import register_env
from omegaconf import DictConfig

from agents import DualSAC
from common.observation_buffer import ObservationBuffer
from common.schedulers import ConstantScheduler
from config import GAILConfig, SAILConfig, setup_config
from loggers import ConsoleLogger, FileLogger, Logger, MultiLogger, WandbLogger
from rewarders import GAIL, MBC, SAIL, DualRewarder, EnvReward, Rewarder
from utils import dict_add, dict_div
from utils.rl import get_markers_by_ep


def train(
    config: DictConfig,
    logger: Logger,
    agent: DualSAC,
    replay_buffer: ObservationBuffer,
    rewarders: list[Rewarder],
) -> None:
    """
    Train the agent and the rewarders.

    Parameters
    ----------
    `config` -> the configuration dict.
    `logger` -> the logger to use.
    `agent` -> the agent to train.
    `replay_buffer` -> the pre-filled replay buffer.
    `rewarders` -> the rewarders to train.
    """

    logger.info("Pre-training the agent and the rewarders using a pre-filled buffer")

    start = time.time()
    log_dict, logged = {}, 0

    demos = get_markers_by_ep(replay_buffer.all(), 1000, config.device, n_ep=50)

    for step in range(config.updates):
        # Train the rewarders
        batch = replay_buffer.sample(config.rewarder_batch_size)
        for rewarder in rewarders:
            rewarder.train(batch, demos)

        # Train the agent
        batch = replay_buffer.sample(config.batch_size)
        new_log = agent.update_parameters(batch, step, None)
        dict_add(log_dict, new_log)
        logged += 1

        if step % 1000 == 0:
            logger.info(new_log, ["console"])
            logger.info(
                {
                    "Pre-training step": step,
                    "Policy loss": new_log["loss/policy_mean"],
                    "Critic loss": new_log["loss/critic"],
                },
            )

    dict_div(log_dict, logged)

    took = time.time() - start
    logger.info(
        {
            "Pre-training": None,
            "Num. updates": config.updates,
            "Took": took,
        },
    )

    return log_dict


def save(
    agent: DualSAC, rewarders: list[Rewarder], dir_path: str, models_id: int
) -> str:
    """
    Save the agent and the rewarders models to a file as a dict with the following structure:
    ```
    {
        "agent": agent_model_dict,
        "sail": sail_model_dict,
        "mbc": mbc_model_dict,
        "gail": gail_model_dict,
        "env_reward": env_reward_model_dict,
        "dual": dual_model_dict,
    }
    ```

    Parameters
    ----------
    `agent` -> the agent to save.
    `rewarders` -> the rewarders to save.
    `dir_path` -> the directory path where to save the models.
    `models_id` -> the id of the models.

    Returns
    -------
    The path of the saved file, which will be `dir_path/pretrained_models_{models_id}.pt`.
    """

    model = {}
    model["agent"] = agent.get_model_dict()

    for rewarder in rewarders:
        if isinstance(rewarder, SAIL):
            model["sail"] = rewarder.get_model_dict()
        elif isinstance(rewarder, MBC):
            model["mbc"] = rewarder.get_model_dict()
        elif isinstance(rewarder, GAIL):
            model["gail"] = rewarder.get_model_dict()
        elif isinstance(rewarder, EnvReward):
            model["env_reward"] = rewarder.get_model_dict()
        elif isinstance(rewarder, DualRewarder):
            model["dual"] = rewarder.get_model_dict()
        else:
            raise NotImplementedError

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_name = "pretrained_models_" + models_id + ".pt"
    file_path = os.path.join(dir_path, file_name)
    torch.save(model, file_path)

    return file_path


@hydra.main(version_base=None, config_path="configs", config_name="pretrain")
def main(config: DictConfig) -> None:
    """
    Pre-train the agent and the rewarders using a pre-filled buffer, and save the models to a .pt file.
    """

    config.logger.run_id = str(int(time.time()))

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

    # Set up morphology
    morpho = np.array(env.morpho_params)
    env.set_task(*morpho)

    obs_size = env.observation_space.shape[0]
    num_morpho = morpho.shape[0]
    if config.absorbing_state:
        obs_size += 1
    if config.morpho_in_state:
        obs_size += num_morpho
    demo_dim = 36

    # Rewarders
    env_reward = EnvReward(config.device)
    config_gail = config
    config_gail.method.rewarder = GAILConfig()
    gail = GAIL(demo_dim, config_gail)
    config_sail = config
    config_sail.method.rewarder = SAILConfig()
    sail = SAIL(logger, env, demo_dim, config_sail)
    rewarders = [env_reward, gail, sail]

    # Agent
    omega_scheduler = ConstantScheduler(0.0)
    agent = DualSAC(
        config,
        logger,
        env.action_space,
        obs_size,
        num_morpho,
        env_reward,
        gail,
        omega_scheduler,
    )

    # Replay buffer
    replay_buffer = ObservationBuffer(
        config.method.replay_capacity,
        config.method.replay_dim_ratio,
        config.seed,
    )
    obs_list = torch.load(config.method.replay_buffer_path)["buffer"]
    logger.info(
        {
            "Loading pre-filled replay buffer": None,
            "Path": config.method.replay_buffer_path,
            "Number of observations": len(obs_list),
        }
    )
    replay_buffer.replace(obs_list)

    # Train a model using the selected training method
    train(config, logger, agent, replay_buffer, rewarders)

    # Save the models
    saved_path = save(agent, rewarders, config.models_dir_path, config.logger.run_id)
    logger.info("Saved models to " + saved_path)

    env.close()


if __name__ == "__main__":
    setup_config()
    main()
