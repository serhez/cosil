from .scheduler import Scheduler


class AlternatingScheduler(Scheduler):
    """
    A scheduler which toggles between an initial and alternative value after every period of steps.
    """

    def __init__(
        self,
        init_value: float,
        alt_value: float,
        init_period: int = 1,
        alt_period: int = 1,
    ):
        """
        Initializes the alternating scheduler and sets the current value to the minimum.

        Parameters
        ----------
        `init_value` -> the parameter's initial value.
        `alt_value` -> the parameter's alternative value.
        `init_period` -> the number of steps before the parameter is toggled to `alt_value`.
        `alt_period` -> the number of steps before the parameter is toggled to `init_value`.
        """

        super().__init__(init_value)

        self._init_value = init_value
        self._alt_value = alt_value
        self._init_period = init_period
        self._alt_period = alt_period
        self._step = 0

    def step(self) -> float:
        self._step += 1

        if self.value == self._init_value and self._step % self._init_period == 0:
            self._value = self._alt_value
        elif self.value == self._alt_value and self._step % self._alt_period == 0:
            self._value = self._init_value
            self._step = 0

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
        if "init_period" in new_hyper_params:
            new_value = new_hyper_params["init_period"]
            if new_value is not None and not isinstance(new_value, int):
                raise ValueError(
                    f"The scheduler's init_period must be an int, but got {new_value}"
                )
            self._init_period = new_value
        if "alt_period" in new_hyper_params:
            new_value = new_hyper_params["alt_period"]
            if new_value is not None and not isinstance(new_value, int):
                raise ValueError(
                    f"The scheduler's alt_period must be an int, but got {new_value}"
                )
            self._alt_period = new_value

        self._value = self._init_value
        self._step = 0

        return self._value
