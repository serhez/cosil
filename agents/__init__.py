from typing import Tuple

import gym
from omegaconf import DictConfig

from common.schedulers import Scheduler
from loggers import Logger
from rewarders import Rewarder

from .agent import Agent
from .dual_reward_sac import DualRewardSAC
from .dual_sac import DualSAC
from .sac import SAC

__all__ = ["Agent", "SAC", "DualSAC", "DualRewardSAC", "create_dual_agents"]


def create_dual_agents(
    config: DictConfig,
    logger: Logger,
    env: gym.Env,
    rl_rewarder: Rewarder,
    il_rewarder: Rewarder,
    pop_omega_scheduler: Scheduler,
    ind_omega_scheduler: Scheduler,
    demo_dim: int,
    obs_size: int,
    num_morpho: int,
) -> Tuple[Agent, Agent]:
    """
    Creates the dual agents for the given configuration (i.e., the individual and population agents).

    Parameters
    ----------
    `config` -> the configuration.
    `logger` -> the logger.
    `env` -> the environment.
    `rl_rewarder` -> the RL rewarder.
    `il_rewarder` -> the IL rewarder.
    `pop_omega_scheduler` -> the population agent's omega scheduler.
    `ind_omega_scheduler` -> the individual agent's omega scheduler.
    `demo_dim` -> the dimension of the demonstrations.
    `obs_size` -> the size of the observations.
    `num_morpho` -> the dimensions of the morphology parameters.

    Returns
    -------
    The individual and population agents.
    """

    ind_agent, pop_agent = None, None

    if config.method.agent.name == "sac":
        assert not config.method.dual_mode == "loss_term" or (
            not config.method.rewarder.name == "gail"
            and not config.method.rewarder.name == "sail"
        ), "Loss-term dual mode cannot be used with GAIL nor SAIL"

        common_args = [
            config,
            logger,
            env.action_space,
            obs_size + num_morpho if config.morpho_in_state else obs_size,
            num_morpho,
            rl_rewarder,
        ]
        if config.method.dual_mode == "loss_term":
            pop_agent = SAC(
                *common_args,
                None,
                pop_omega_scheduler,
                "pop",
            )
            ind_agent = SAC(*common_args, il_rewarder, ind_omega_scheduler, "ind")
        elif config.method.dual_mode == "reward":
            pop_agent = DualRewardSAC(
                *common_args,
                il_rewarder,
                pop_omega_scheduler,
                "pop",
            )
            ind_agent = DualRewardSAC(
                *common_args,
                il_rewarder,
                ind_omega_scheduler,
                "ind",
            )
    elif config.method.agent.name == "dual_sac":
        common_args = [
            config,
            logger,
            env.action_space,
            obs_size + num_morpho if config.morpho_in_state else obs_size,
            num_morpho,
            rl_rewarder,
        ]
        pop_agent = DualSAC(
            *common_args,
            il_rewarder,
            demo_dim,
            pop_omega_scheduler,
            "pop",
        )
        ind_agent = DualSAC(
            *common_args,
            il_rewarder,
            demo_dim,
            ind_omega_scheduler,
            "ind",
        )
    else:
        raise ValueError("Invalid agent")

    return ind_agent, pop_agent
