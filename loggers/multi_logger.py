from typing import Any, Dict, Union

from .logger import Logger


class MultiLogger(Logger):
    """Logs to multiple loggers."""

    def __init__(self, loggers: Dict[str, Logger], default_mask: list[str] = []):
        """
        Initializes a multi-logger.

        Parameters
        ----------
        loggers -> a dictionary with the names of the loggers as keys and the loggers as values.
        default_mask -> the default mask to use when logging.
        """

        super().__init__(default_mask)
        self._loggers = loggers

    def _log_impl(
        self,
        message: Union[str, Dict[str, Any]],
        level: str = "INFO",
        mask: list[str] = [],
    ) -> bool:
        """
        Logs a message to multiple loggers.

        Parameters
        ----------
        message -> the message to log.
        level -> the level of the message (e.g., INFO, WARNING, ERROR, etc.).
        mask -> a list of logger names to not be used to log this message.

        Returns
        -------
        Whether the message was logged successfully by all loggers.
        """

        success = True

        for logger in [
            logger
            for logger_name, logger in self._loggers.items()
            if logger_name not in mask
        ]:
            success = success and logger(message, level, mask)

        return success

    def close(self) -> bool:
        """
        Closes the logger.

        Returns
        -------
        Whether all loggers were successfully closed.
        """

        success = True

        for logger in self._loggers.values():
            success = success and logger.close()

        return success
