from abc import ABC, abstractmethod


class Scheduler(ABC):
    """The base class for schedulers which update a parameter value based on the current episode."""

    def __init__(self, value: float):
        """
        Initializes the base instance of the scheduler.

        Parameters
        ----------
        `value` -> the initial value of the parameter.
        """

        self._value = value
        self.__bkp_value = value

    def unsafe_set(self, value: float) -> None:
        """
        Sets the value of the parameter.
        This method is unsafe because it does not respect the logic of the scheduler.
        The user is responsible for calling `unsafe_reset` after calling this method and before calling `step`.

        Parameters
        ----------
        `value` -> the new value of the parameter.
        """

        self.__bkp_value = self._value
        self._value = value

    def unsafe_reset(self) -> None:
        """
        Resets the value of the parameter to the last value set by `unsafe_set`.
        This method is unsafe because it does not respect the logic of the scheduler.
        The user is responsible for calling this method after calling `unsafe_set` and before calling `step`.
        """

        self._value = self.__bkp_value

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a `dict`.

        It contains an entry for every variable in `self.__dict__`.

        Returns
        -------
        The scheduler state.
        """

        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict) -> None:
        """Loads the schedulers state.

        Parameters
        ----------
        `state_dict` -> the scheduler state; should be an object returned from a call to `state_dict`.
        """

        self.__dict__.update(state_dict)

    @property
    def value(self) -> float:
        """The last computed value of the parameter."""
        return self._value

    @abstractmethod
    def step(self) -> float:
        """
        Updates the parameter value based on the current episode.
        The computed value is cached and can be retrieved without overhead at `scheduler.value`.

        Returns
        -------
        The computed value.
        """
        pass

    @abstractmethod
    def reset(self, hard: bool = False, **new_hyper_params) -> float:
        """
        Resets the scheduler manually by performing a warm restart.
        It resets its internal step to 0 and its hyperparameters to their initial values.
        New initial values for the hyperparameters can be passed as keyword arguments.

        Parameters
        ----------
        `hard` -> whether to reset the scheduler to its initial state (True) or to respect the warm restart logic of the scheduler (False).
        `new_hyper_params` -> the new hyper parameters to be used by the scheduler.

        Returns
        -------
        The reseted value of the parameter.
        """
        pass
