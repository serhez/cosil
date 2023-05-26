from typing import List, Union

from .scheduler import Scheduler


class StepScheduler(Scheduler):
    """
    A scheduler which multiplies the parameter value by a factor every period of steps.
    """

    def __init__(
        self,
        init_value: float,
        factor: float,
        period: Union[int, List[int]],
    ):
        """
        Initializes the min-max scheduler and sets the current value to the minimum.

        Parameters
        ----------
        `init_value` -> the parameter's initial value.
        `factor` -> the factor by which the parameter value is decreased every period.
        `period` -> the number of steps before the parameter's value is decreased by the `factor`.
        - If `period` is an int, the scheduler will decrease the parameter's value by the `factor` every period steps.
        - If `period` is a list of ints, each element in the list will be used as a period in the order they appear.
        - If the list is exhausted, the last element will be used for all subsequent periods.
        """

        if not isinstance(init_value, float):
            raise ValueError(
                f"The scheduler's init_value must be a float, but got {init_value}"
            )
        if not isinstance(factor, float):
            raise ValueError(
                f"The scheduler's factor must be a float, but got {factor}"
            )
        if not isinstance(period, int) and not isinstance(period, List[int]):
            raise ValueError(
                f"The scheduler's period must be an int or a list of ints, but got {period}"
            )

        self._value = init_value
        self._init_value = init_value
        self._factor = factor
        self._period = period
        self._step = 0

    @property
    def value(self) -> float:
        return self._value

    def step(self) -> float:
        if isinstance(self._period, int):
            period = self._period
        elif isinstance(self._period, list):
            if len(self._period) == 1:
                period = self._period[0]
            else:
                period = self._period.pop(0)

        else:
            raise ValueError(
                f"The scheduler's period must be an int or a list of ints, but got {self._period}"
            )

        if self._step >= period:
            self._value *= self._factor
            self._step = 0
        else:
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
        if "factor" in new_hyper_params:
            new_value = new_hyper_params["factor"]
            if not isinstance(new_value, float):
                raise ValueError(
                    f"The scheduler's factor must be a float, but got {new_value}"
                )
            self._factor = new_value
        if "period" in new_hyper_params:
            new_value = new_hyper_params["period"]
            if not isinstance(new_value, int) and not isinstance(new_value, List[int]):
                raise ValueError(
                    f"The scheduler's period must be an int or a list of ints, but got {new_value}"
                )
            self._period = new_value

        self._value = self._init_value
        self._step = 0

        return self._value
