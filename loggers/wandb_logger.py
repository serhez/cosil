from .logger import Logger
from typing import Any, Dict
import wandb


class WandbLogger(Logger):
    """Logs to Weights & Biases."""

    def __init__(
        self, project: str, experiment: str, group: str, config: Dict[str, Any]
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

        wandb.init(project=project, name=experiment, group=group, config=config)

    def log(self, message: Dict[str, Any], *_) -> bool:
        """
        Logs a message to Weights & Biases.

        Parameters
        ----------
        message -> the message to log.
        [UNUSED] mask
        [UNUSED] level

        Returns
        -------
        Whether the message was logged successfully.
        """

        try:
            wandb.log(message)
        except:
            return False

        return True

    def close(self) -> bool:
        """Closes the logger."""

        wandb.finish()
        return True
