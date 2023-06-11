import numpy as np

from .alternating import AlternatingScheduler
from .binary import BinaryScheduler
from .constant import ConstantScheduler
from .cosine_annealing import CosineAnnealingScheduler
from .exponential import ExponentialScheduler
from .scheduler import Scheduler
from .step import StepScheduler


def create_scheduler(
    name: str,
    period: int,
    n_init_episodes: int = 1,
    val_init: float = 1.0,
    val_alt: float = 0.0,
    val_decay: float = 1.0,
    T_mult: float = 1.0,
) -> Scheduler:
    """
    Create a scheduler.
    Parameters
    ----------
    `name` -> the name of the scheduler.
    `period` -> the number of episodes after which the scheduler will be reset.
    `n_init_episodes` -> the number of episodes to maintain the alt value (e.g., in the alternating scheduler).
    `val_init` -> the initial value of the scheduler; it will also be used as the `val_max`.
    `val_alt` -> the alternating value of the scheduler; it will also be used as the `val_min`.
    `val_decay` -> the decay rate of the scheduler; it will also be used as the `val_factor`.
    `T_mult` -> the multiplier of the period of the scheduler.

    Returns
    -------
    The scheduler.

    Raises
    ------
    ValueError -> if the scheduler is not supported.
    """

    if name == "alternating":
        return AlternatingScheduler(
            val_init, val_alt, n_init_episodes, period - n_init_episodes
        )
    elif name == "binary":
        return BinaryScheduler(val_init, val_alt, period)
    elif name == "constant":
        return ConstantScheduler(val_init)
    elif name == "cosine_annealing":
        return CosineAnnealingScheduler(val_alt, val_init, period, T_mult)
    elif name == "exponential":
        # This choice of gamma seems reasonable for val, but other choices are possible
        gamma = 1 - np.sqrt(period - 1) / (period - 1)
        return ExponentialScheduler(val_init, gamma, val_decay, period, T_mult)
    elif name == "step":
        return StepScheduler(val_init, val_decay, period)
    else:
        raise ValueError(f"val scheduler is not supported: {name}")
