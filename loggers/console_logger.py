import json
from datetime import datetime
from typing import Any, Dict, Union

from termcolor import colored

from .logger import Logger


class ConsoleLogger(Logger):
    """Logs to the console (i.e., standard I/O)."""

    def __init__(self):
        """
        Initializes a console logger.
        """

        super().__init__(None)

    def _log_impl(
        self, message: Union[str, Dict[str, Any]], level: str = "INFO", *_
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

        time = "[" + datetime.now().strftime("%H:%M:%S") + "]"

        if level == "INFO":
            printed_level = ""
        else:
            printed_level = "[" + level + "] "

        if level == "WARNING":
            colored_level = colored(printed_level, "yellow")
        elif level == "ERROR":
            colored_level = colored(printed_level, "red")
        else:
            colored_level = colored(printed_level, "cyan")

        if isinstance(message, str):
            print(f"{colored_level}{time} {message}")

        # The first level of the dictionary is printed as a multiline
        # indented message.
        # The rest of the levels are printed as a single line
        # pretifyed depending on the type of the value.
        else:
            first = True
            for key, value in message.items():
                if not first:
                    time = " " * len(time)
                    colored_level = " " * len(printed_level)

                if isinstance(value, float):
                    print(f"{colored_level}{time} {key}: {value:.5f}")
                elif isinstance(value, Union[dict, list]):
                    value = json.dumps(value, indent=4)
                    print(f"{colored_level}{time} {key}: {value}")
                elif value is None:  # Used for headers, titles, etc.
                    print(f"{colored_level}{time} {key}")
                else:
                    print(f"{colored_level}{time} {key}: {value}")

                first = False

        return True

    def close(self) -> bool:
        """Closes the logger."""

        return True
