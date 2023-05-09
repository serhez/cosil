from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Union


class Logger(ABC):
    @abstractmethod
    def log(
        self,
        message: Union[str, Dict[str, Any]],
        level: str = "INFO",
        mask: list[str] = ["wandb"],
    ) -> bool:
        """
        Log a message to the logger.

        Parameters
        ----------
        message -> the message to log.
        - If a string, the message will be logged as-is.
        - If a dictionary, the message will be logged as a JSON string.
            - The dictionary must be JSON serializable.
            - You can provide None dictionary values to mean that the key is a header or title of the message.
        level -> the level of the message (e.g., INFO, WARNING, ERROR, etc.).
        mask -> a list of logger names to not be used to log this message.

        Returns
        -------
        Whether the message was successfully logged.
        """

        pass

    def _call_impl(self, *args, **kwargs):
        return self.log(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    def info(
        self, message: Union[str, Dict[str, Any]], mask: list[str] = ["wandb"]
    ) -> bool:
        """
        Wrapper for calling logger.log with level="INFO".

        Parameters
        ----------
        message -> the message to log.
        - If a string, the message will be logged as-is.
        - If a dictionary, the message will be logged as a JSON string.
            - The dictionary must be JSON serializable.
            - You can provide None dictionary values to mean that the key is a header or title of the message.
        mask -> a list of logger names to not be used to log this message.

        Returns
        -------
        Whether the message was successfully logged.
        """

        return self.log(message, "INFO", mask)

    def warning(
        self, message: Union[str, Dict[str, Any]], mask: list[str] = ["wandb"]
    ) -> bool:
        """
        Wrapper for calling logger.log with level="WARNING".

        Parameters
        ----------
        message -> the message to log.
        - If a string, the message will be logged as-is.
        - If a dictionary, the message will be logged as a JSON string.
            - The dictionary must be JSON serializable.
            - You can provide None dictionary values to mean that the key is a header or title of the message.
        mask -> a list of logger names to not be used to log this message.

        Returns
        -------
        Whether the message was successfully logged.
        """

        return self.log(message, "WARNING", mask)

    def error(
        self, message: Union[str, Dict[str, Any]], mask: list[str] = ["wandb"]
    ) -> bool:
        """
        Wrapper for calling logger.log with level="ERROR".

        Parameters
        ----------
        message -> the message to log.
        - If a string, the message will be logged as-is.
        - If a dictionary, the message will be logged as a JSON string.
            - The dictionary must be JSON serializable.
            - You can provide None dictionary values to mean that the key is a header or title of the message.
        mask -> a list of logger names to not be used to log this message.

        Returns
        -------
        Whether the message was successfully logged.
        """

        return self.log(message, "ERROR", mask)

    @abstractmethod
    def close(self) -> bool:
        """
        Close the logger.

        Returns
        -------
        Whether the logger was successfully closed.
        """

        pass
