import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Union

from .logger import Logger


class FileLogger(Logger):
    """Logs to a file."""

    def __init__(self, project: str, group: str, name: str, id: str):
        """
        Initializes a file logger.

        Parameters
        ----------
        project -> the name of the project.
        group -> the name of the group.
        name -> the name of the experiment.
        id -> the ID of the run.
        """

        TRIES = 10

        # Define file path and name
        dir_path = f"logs/{project}/{group}/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = dir_path + name + "_" + id + ".json"

        # Use a random uuid if the file name is already taken
        while os.path.exists(file_path) and TRIES > 0:
            file_path = dir_path + str(uuid.uuid4()) + ".json"
            TRIES -= 1
        if TRIES == 0:
            print(
                "[WARNING] Could not create log file: too many tries. Subsequent logs will fail to be saved"
            )

        # Create the file
        with open(file_path, "x"):
            pass
        print(f"Logging to file {file_path}")

        self._file_path = file_path

    def log(
        self, message: Union[str, Dict[str, Any]], level: str = "INFO", _=[]
    ) -> bool:
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
            with open(self._file_path) as file:
                try:
                    logs = json.load(file)
                except json.decoder.JSONDecodeError:
                    logs = []

            log = {
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "level": level,
                "message": message,
            }
            logs.append(log)

            with open(self._file_path, "w") as file:
                file.seek(0)
                json.dump(logs, file, indent=4)

        except Exception as e:
            print(f"[ERROR] Error while logging to a file: {e}")
            return False

        return True

    def close(self) -> bool:
        """Closes the logger."""

        return True
