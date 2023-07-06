from typing import Optional

import numpy as np

from .scheduler import Scheduler


class CosineAnnealingScheduler(Scheduler):
    """
    A cosine annealing scheduler which updates a parameter value based on the current episode.
    The scheduler can perform warm restarts after a certain number of episodes.
    The warm restart intervals can be increased after each restart.
    If no warm restarts are desired, the scheduler can be understood as a cosine annealing scheduler with a maximum episode.
    Note that this scheduler does not work over an infinite horizon because it is based on a periodic function.

    This is an adaptation of PyTorch's `CosineAnnealingWarmRestarts`:
    https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py
    """

    def __init__(
        self,
        min_val: float,
        max_val: float,
        T_0: int,
        T_mult: float = 1.0,
        last_episode: Optional[int] = None,
    ):
        """
        Initializes the cosine annealing scheduler and performs the first step.

        Parameters
        ----------
        min_val -> the parameter's minimum value.
        max_val -> the parameter's maximum value.
        T_0 -> the initial restart interval (if warm restarts are desired) or the maximum episode.
        T_mult -> a factor which increases the restart interval after each restart (only applies if warm restarts are ever performed).
        last_episode -> the last episode if the scheduler is resumed from a checkpoint.
        """

        if not isinstance(T_0, int) or T_0 <= 0:
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if not isinstance(T_mult, float) or T_mult < 1.0:
            raise ValueError(f"Expected float T_mult >= 1.0, but got {T_mult}")
        if last_episode is not None and (
            not isinstance(last_episode, int) or last_episode <= 0
        ):
            raise ValueError(
                f"Expected positive integer last_episode, but got {last_episode}"
            )

        self._min_val = min_val
        self._max_val = max_val

        # Restart attributes
        self._T_0 = T_0
        self._T_i = T_0
        self._T_mult = T_mult
        self._T_cur = (last_episode if last_episode is not None else 0) % self._T_0
        self._n_restarts = 0

        super().__init__(self._calculate_value())

    def _calculate_value(self) -> float:
        """Returns the current parameter value."""

        return (
            self._min_val
            + (self._max_val - self._min_val)
            * (1 + np.cos(np.pi * self._T_cur / self._T_i))
            / 2
        )

    def step(self) -> float:
        self._T_cur = self._T_cur + 1

        # Warm restart
        if self._T_cur >= self._T_i:
            self._T_cur -= self._T_i
            self._T_i *= self._T_mult
            self._n_restarts += 1

        self._value = self._calculate_value()

        return self._value

    def reset(self, hard: bool = False, **new_hyper_params) -> float:
        # Change the hyper parameters
        if "T_0" in new_hyper_params:
            new_T_0 = new_hyper_params["T_0"]
            if not isinstance(new_T_0, int) or new_T_0 <= 0:
                raise ValueError(f"Expected positive integer T_0, but got {new_T_0}")
            self._T_0 = new_T_0

        # Reset the timestep
        self._T_cur = 0
        if hard:
            self._T_i = self._T_0
            self._n_restarts = 0
        else:
            self._T_i = self._T_0 * self._T_mult**self._n_restarts
            self._n_restarts += 1

        # Calculate the new value
        self._value = self._calculate_value()

        return self._value
