import gym
from omegaconf import DictConfig

from loggers import Logger

from .coil import CoIL
from .cosil import CoSIL
from .cosil3 import CoSIL2

__all__ = ["CoIL", "CoSIL", "CoSIL2"]


def create_method(config: DictConfig, logger: Logger, env: gym.Env):
    if config.method.name == "coil":
        return CoIL(config, logger, env)
    elif config.method.name == "cosil":
        return CoSIL(config, logger, env)
    elif config.method.name == "cosil2":
        return CoSIL2(config, logger, env)
    else:
        raise ValueError(f"Invalid training method: {config.method.name}")
