from typing import Optional

from .scheduler import Scheduler


class ExponentialScheduler(Scheduler):
    """
    An exponential scheduler which updates a parameter value based on the current episode.
    Optionally, the scheduler can perform warm restarts after a certain number of episodes.
    The warm restart intervals can be increased after each restart.

    This is an adaptation of PyTorch's `ExponentialLR`:
    https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py
    """

    def __init__(
        self,
        init_val: float,
        gamma: float,
        init_val_decay: float = 1.0,
        T_0: Optional[int] = None,
        T_mult: float = 1.0,
        last_episode: Optional[int] = None,
    ):
        """
        Initializes the cosine annealing scheduler and performs the first step.

        Parameters
        ----------
        init_val -> the parameter's initial value.
        gamma -> the decay rate.
        init_val_decay -> the decay rate for the initial value after each warm restart.
        T_0 -> the initial restart interval (if warm restarts are desired).
        T_mult -> a factor which increases the restart interval after each restart (only applies if warm restarts are ever performed).
        last_episode -> the last episode if the scheduler is resumed from a checkpoint.
        """

        if T_0 is not None and (not isinstance(T_0, int) or T_0 <= 0):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if not isinstance(T_mult, float) or T_mult < 1.0:
            raise ValueError(f"Expected float T_mult >= 1.0, but got {T_mult}")
        if not isinstance(init_val_decay, float) or not 0.0 < init_val_decay <= 1.0:
            raise ValueError(
                f"Expected float 0.0 > init_val_decay >= 1.0, but got {init_val_decay}"
            )
        if last_episode is not None and (
            not isinstance(last_episode, int) or last_episode <= 0
        ):
            raise ValueError(
                f"Expected positive integer last_episode, but got {last_episode}"
            )

        self._init_val = init_val
        self._gamma = gamma
        self._init_val_decay = init_val_decay

        # Restart attributes
        self._T_0 = T_0
        self._T_i = T_0
        self._T_mult = T_mult
        self._T_cur = last_episode if last_episode is not None else 0
        if self._T_0 is not None:
            self._T_cur = self._T_cur % self._T_0
        self._n_restarts = 0

        self._last_val = self._calculate_value()

    def _calculate_value(self) -> float:
        """Returns the current parameter value."""

        return self._init_val * self._gamma**self._T_cur

    @property
    def value(self) -> float:
        return self._last_val

    def step(self) -> float:
        self._T_cur = self._T_cur + 1

        # Warm restart
        if self._T_i is not None and self._T_cur >= self._T_i:
            self._T_cur -= self._T_i
            self._T_i *= self._T_mult
            self._init_val *= self._init_val_decay
            self._n_restarts += 1

        self._last_val = self._calculate_value()

        return self._last_val

    def reset(self, hard: bool = False, **new_hyper_params) -> float:
        # Change the hyper parameters
        if "gamma" in new_hyper_params:
            new_gamma = new_hyper_params["gamma"]
            if not isinstance(new_gamma, float):
                raise ValueError(f"Gamma must be a float, but got {new_gamma}")
            self._gamma = new_gamma

        # Reset the timestep
        self._T_cur = 0
        if self._T_0 is not None and self._T_i is not None:
            if hard:
                self._T_i = self._T_0
                self._n_restarts = 0
            else:
                self._T_i = self._T_0 * self._T_mult**self._n_restarts
                self._n_restarts += 1

        # Calculate the new value
        self._last_val = self._calculate_value()

        return self._last_val
