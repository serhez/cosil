from abc import ABC, abstractmethod
from typing import Any, Dict, Union


class Logger(ABC):
    @abstractmethod
    def log(
        self,
        message: Union[str, Dict[str, Any]],
        level: str = "INFO",
        mask: list[str] = [],
    ) -> bool:
        """
        Log a message to the logger.

        Parameters
        ----------
        message -> the message to log.
        level -> the level of the message (e.g., INFO, WARNING, ERROR, etc.).
        mask -> a list of logger names to not be used to log this message.

        Returns
        -------
        Whether the message was successfully logged.
        """

        pass

    @abstractmethod
    def close(self) -> bool:
        """
        Close the logger.

        Returns
        -------
        Whether the logger was successfully closed.
        """

        pass
