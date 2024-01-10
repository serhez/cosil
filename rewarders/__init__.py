from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig

from loggers import Logger
from normalizers import Normalizer

from .airl import AIRL
from .env_reward import EnvReward
from .gail import GAIL
from .mbc import MBC
from .rewarder import Rewarder
from .sail import SAIL


def create_rewarder(
    name: str,
    config: DictConfig,
    logger: Logger,
    obs_size: int,
    num_morpho: int,
    action_space: np.ndarray,
    demo_dim: np.ndarray,
    bounds: torch.Tensor,
    normalizer: Optional[Normalizer] = None,
) -> Optional[Rewarder]:
    """
    Create a rewarder, given its name.
    This function does not support the dual rewarder.

    Parameters
    ----------
    `name` -> the name of the rewarder to create.
    `config` -> the configuration object.
    `logger` -> the logger.
    `obs_size` -> the size of the observations.
    `num_morpho` -> the dimensionality of morphologies.
    `action_space` -> the action space of the environment.
    `demo_dim` -> the dimension of the demonstrator's observations.
    `bounds` -> the bounds of the morphology parameters.
    `normalizer` -> the normalizer to use for the rewarder.
    """

    if name == "env":
        return EnvReward(config.device, normalizer, config.method.sparse_mask)
    elif name == "mbc":
        return MBC(
            config.device, bounds, config.method.optimized_demonstrator, normalizer
        )
    elif name == "gail":
        return GAIL(demo_dim, config, normalizer)
    elif name == "airl":
        return AIRL(demo_dim, config, normalizer)
    elif name == "sail":
        return SAIL(
            logger, obs_size, num_morpho, action_space, demo_dim, config, normalizer
        )
    else:
        raise ValueError("Unknown rewarder: {}".format(name))
