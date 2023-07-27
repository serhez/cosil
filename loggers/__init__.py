from omegaconf import DictConfig

from .console_logger import ConsoleLogger
from .file_logger import FileLogger
from .logger import Logger
from .multi_logger import MultiLogger
from .wandb_logger import WandbLogger

__all__ = [
    "ConsoleLogger",
    "FileLogger",
    "Logger",
    "MultiLogger",
    "WandbLogger",
    "create_multilogger",
]


def create_multilogger(config: DictConfig) -> MultiLogger:
    loggers_list = config.logger.loggers.split(",")
    loggers = {}
    for logger in loggers_list:
        if logger == "console":
            loggers["console"] = ConsoleLogger()
        elif logger == "file":
            loggers["file"] = FileLogger(
                config.logger.project_name,
                config.logger.group_name,
                config.logger.experiment_name,
                config.logger.run_id,
            )
        elif logger == "wandb":
            loggers["wandb"] = WandbLogger(
                config.logger.project_name,
                config.logger.group_name,
                config.logger.experiment_name,
                config.logger.run_id,
                config,
            )
        elif logger == "":
            pass
        else:
            print(f'[WARNING] Logger "{logger}" is not supported')

    return MultiLogger(loggers, config.logger.default_mask.split(","))
