from datetime import datetime
from typing import Any, Dict, Union

from .logger import Logger


class ConsoleLogger(Logger):
    """Logs to the console (i.e., standard I/O)."""

    def __init__(self):
        """
        Initializes a console logger.
        """

    def log(
        self, message: Union[str, Dict[str, Any]], level: str = "INFO", _=[]
    ) -> bool:
        """
        Logs a message to the console.

        Parameters
        ----------
        message -> the message to log.
        level -> the level of the message (e.g., INFO, WARNING, ERROR, etc.).
        [UNUSED] mask

        Returns
        -------
        True.
        """

        time = datetime.now().strftime("%H:%M:%S")
        if isinstance(message, str):
            printed_level = level + " " if level != "INFO" else ""
            print(f"{printed_level}[{time}] {message}")
        else:
            first = True
            for key, value in message.items():
                printed_level = level + " " if level != "INFO" else ""

                if not first:
                    time = " " * len(time)

                if isinstance(value, float):
                    print(f"{printed_level}[{time}] {key}: {value:.5f}")
                elif value is None:  # Used for headers, titles, etc.
                    print(f"{printed_level}[{time}] {key}")
                else:
                    print(f"{printed_level}[{time}] {key}: {value}")

                first = False

        return True

    def close(self) -> bool:
        """Closes the logger."""

        return True
