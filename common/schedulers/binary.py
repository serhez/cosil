from .scheduler import Scheduler


class BinaryScheduler(Scheduler):
    """
    A scheduler which starts with an initial value and jumps to an alternative value after a period of steps, never to return to the initial value.
    """

    def __init__(
        self,
        init_value: float,
        alt_value: float,
        period: int,
    ):
        """
        Initializes the binary scheduler and sets the current value to the initial value.

        Parameters
        ----------
        `init_value` -> the parameter's initial value.
        `alt_value` -> the parameter's alternative value.
        `period` -> the number of steps before the parameter adopts the alternative value.
        """

        super().__init__(init_value)

        self._init_value = init_value
        self._alt_value = alt_value
        self._period = period
        self._step = 0

    def step(self) -> float:
        if self._step >= self._period:
            self._value = self._alt_value
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
            if not isinstance(new_value, int):
                raise ValueError(
                    f"The scheduler's period must be an int, but got {new_value}"
                )
            self._period = new_value

        self._value = self._init_value
        self._step = 0

        return self._value
