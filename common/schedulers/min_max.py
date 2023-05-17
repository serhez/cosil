from .scheduler import Scheduler


class MinMaxScheduler(Scheduler):
    """
    A scheduler which starts with a minimum value and immediately jumps to a maximum value after a period of steps.
    """

    def __init__(
        self,
        min_value: float,
        max_value: float,
        period: int,
    ):
        """
        Initializes the min-max scheduler and sets the current value to the minimum.

        Parameters
        ----------
        min_value -> the parameter's minimum value.
        max_value -> the parameter's maximum value.
        period -> the number of steps before the parameter adopts the maximum value.
        """

        self._value = min_value
        self._min_value = min_value
        self._max_value = max_value
        self._period = period
        self._step = 0

    @property
    def value(self) -> float:
        return self._value

    def step(self) -> float:
        if self._step >= self._period:
            self._value = self._max_value
        self._step += 1
        return self._value

    def reset(self, _: bool = False, **new_hyper_params) -> float:
        if "min_value" in new_hyper_params:
            new_value = new_hyper_params["min_value"]
            if not isinstance(new_value, float):
                raise ValueError(
                    f"The scheduler's min_value must be a float, but got {new_value}"
                )
            self._min_value = new_value
        if "max_value" in new_hyper_params:
            new_value = new_hyper_params["max_value"]
            if not isinstance(new_value, float):
                raise ValueError(
                    f"The scheduler's max_value must be a float, but got {new_value}"
                )
            self._max_value = new_value
        if "period" in new_hyper_params:
            new_value = new_hyper_params["period"]
            if not isinstance(new_value, int):
                raise ValueError(
                    f"The scheduler's period must be an int, but got {new_value}"
                )
            self._period = new_value

        self._value = self._min_value
        self._step = 0

        return self._value
