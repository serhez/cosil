import gym
from omegaconf import DictConfig

from loggers import Logger

from .coil import CoIL
from .cosil import CoSIL
from .rl import RL

__all__ = ["RL", "CoIL", "CoSIL"]


def create_method(config: DictConfig, logger: Logger, env: gym.Env):
    if config.method.name == "rl":
        return RL(config, logger, env)
    elif config.method.name == "coil":
        return CoIL(config, logger, env)
    elif config.method.name == "cosil":
        return CoSIL(config, logger, env)
    else:
        raise ValueError(f"Invalid training method: {config.method.name}")
