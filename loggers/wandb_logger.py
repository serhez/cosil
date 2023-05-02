import argparse
from typing import Any, Dict, Union

import wandb

from .logger import Logger


class WandbLogger(Logger):
    """Logs to Weights & Biases."""

    def __init__(
        self, project: str, experiment: str, group: str, config: argparse.Namespace
    ):
        """
        Initializes a Weights & Biases logger.

        Parameters
        ----------
        project -> the name of the project to log to.
        name -> the name of the experiment to log to.
        group -> the name of the group to log to.
        config -> the configuration of the experiment.
        """

        wandb.init(project=project, name=experiment, group=group, config=vars(config))

    def log(self, message: Union[str, Dict[str, Any]], level: str = "INFO", *_) -> bool:
        """
        Logs a message to Weights & Biases.

        Parameters
        ----------
        message -> the message to log.
        level -> the level of the message (e.g., INFO, WARNING, ERROR, etc.); not used for messages of type dictionary.
        [UNUSED] mask

        Returns
        -------
        Whether the message was logged successfully.
        """

        try:
            if isinstance(message, str):
                log = {level: message}
            else:
                log = message
            wandb.log(log)
        except Exception as e:
            print(f"[ERROR] Error while logging to wandb: {e}")
            return False

        return True

    def close(self) -> bool:
        """Closes the logger."""

        wandb.finish()
        return True
