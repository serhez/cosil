import json
from datetime import datetime
from typing import Any, Dict

from .logger import Logger


class FileLogger(Logger):
    """Logs to a file."""

    def __init__(self, json_file_path: str):
        """
        Initializes a file logger.

        Parameters
        ----------
        file_path -> the path to the file to log to, it must be a json file.
        """

        self._file_path = json_file_path

    def log(self, message: Dict[str, Any], level: str = "INFO", _ = []) -> bool:
        """
        Logs a message to a file.

        Parameters
        ----------
        message -> the message to log.
        level -> the level of the message (e.g., INFO, WARNING, ERROR, etc.).
        [UNUSED] mask

        Returns
        -------
        Whether the message was logged successfully.
        """

        try:
            with open(self._file_path, "a") as file:
                log = {
                    "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    "level": level,
                    "message": message,
                }
                file.write(json.dumps(log))
        except:
            return False

        return True

    def close(self) -> bool:
        """Closes the logger."""

        return True
