import torch

from loggers import ConsoleLogger

logger = ConsoleLogger()
model = torch.load(
    "/scratch/work/hernans2/cosil/final/ant/GaitTrackAnt-v0/seed-355/demonstrator_1711202026.pt"
)
morphos = model["morphos"]
logger.info(morphos)
