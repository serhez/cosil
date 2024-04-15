from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union, Tuple


class Logger(ABC):
    def __init__(self, default_mask):
        """
        Initializes the logger base class.

        Parameters
        ----------
        default_mask -> the default mask to use when logging messages.
        """

        self._default_mask = default_mask

    @abstractmethod
    def _log_impl(
        self,
        message,
        level: str = "INFO",
        mask: Optional[list] = None,
    ) -> bool:
        """The child class internal implementation of the log method; not to be called directly."""

        pass

    def log(
        self,
        message: Union[str, Dict[str, Any]],
        level: str = "INFO",
        mask: Optional[list] = None,
    ) -> bool:
        """
        Log a message.

        Parameters
        ----------
        message -> the message to log.
        - If a string, the message will be logged as-is.
        - If a dictionary, the message will be logged as a JSON string.
            - The dictionary must be JSON serializable.
            - You can provide None dictionary values to mean that the key is a header or title of the message.
        level -> the level of the message (e.g., INFO, WARNING, ERROR, etc.).
        mask -> a list of logger names to not be used to log this message.
        - If None, the default mask will be used.

        Returns
        -------
        Whether the message was successfully logged.
        """

        if mask is None:
            mask = self._default_mask
        return self._log_impl(message, level, mask)

    def _call_impl(self, *args, **kwargs):
        return self.log(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    def info(
        self, message: Union[str, Dict[str, Any]], mask: Optional[list] = None
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
        - If None, the default mask will be used.

        Returns
        -------
        Whether the message was successfully logged.
        """

        return self.log(message, "INFO", mask)

    def warning(
        self, message: Union[str, Dict[str, Any]], mask: Optional[list] = None
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
        - If None, the default mask will be used.

        Returns
        -------
        Whether the message was successfully logged.
        """

        return self.log(message, "WARNING", mask)

    def error(
        self, message: Union[str, Dict[str, Any]], mask: Optional[list] = None
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
        - If None, the default mask will be used.

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
