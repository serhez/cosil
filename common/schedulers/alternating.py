from typing import Optional

from .scheduler import Scheduler


class AlternatingScheduler(Scheduler):
    """
    A scheduler which toggles between an initial and alternative value at every step, or automatically after every period of steps.
    """

    def __init__(
        self,
        init_value: float,
        alt_value: float,
        period: Optional[int] = None,
    ):
        """
        Initializes the alternating scheduler and sets the current value to the minimum.

        Parameters
        ----------
        `init_value` -> the parameter's initial value.
        `alt_value` -> the parameter's alternative value.
        `period` -> the number of steps before the parameter is toggled to the other value.
        - If `None`, the parameter will be toggled at every step.
        """

        self._value = init_value
        self._init_value = init_value
        self._alt_value = alt_value
        self._period = period
        self._step = 0

    @property
    def value(self) -> float:
        return self._value

    def step(self) -> float:
        if self._period is not None and self._step % self._period == 0:
            self._value = (
                self._alt_value if self._value == self._init_value else self._init_value
            )

        self._step += 1

        return self._value

    def reset(self, _: bool = False, **new_hyper_params) -> float:
        if "init_value" in new_hyper_params:
            new_value = new_hyper_params["init_value"]
            if not isinstance(new_value, float):
                raise ValueError(
                    f"The scheduler's init_value must be a float, but got {new_value}"
                )
            self._init_value = new_value
        if "alt_value" in new_hyper_params:
            new_value = new_hyper_params["alt_value"]
            if not isinstance(new_value, float):
                raise ValueError(
                    f"The scheduler's alt_value must be a float, but got {new_value}"
                )
            self._alt_value = new_value
        if "period" in new_hyper_params:
            new_value = new_hyper_params["period"]
            if new_value is not None and not isinstance(new_value, int):
                raise ValueError(
                    f"The scheduler's period must be an int, but got {new_value}"
                )
            self._period = new_value

        self._value = self._init_value
        self._step = 0

        return self._value
