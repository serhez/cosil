from .scheduler import Scheduler


class ConstantScheduler(Scheduler):
    """
    A constant scheduler which keeps the parameter value constant.
    """

    def __init__(
        self,
        value: float,
    ):
        """
        Initializes the constant scheduler.

        Parameters
        ----------
        value -> the parameter's constant value.
        """

        self._value = value

    @property
    def value(self) -> float:
        return self._value

    def step(self) -> float:
        return self._value

    def reset(self, _: bool = False, **new_hyper_params) -> float:
        if "value" in new_hyper_params:
            new_value = new_hyper_params["value"]
            if not isinstance(new_value, float):
                raise ValueError(
                    f"The scheduler value must be a float, but got {new_value}"
                )
            self._value = new_value

        return self._value
