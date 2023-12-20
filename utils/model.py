import os

import torch


def save_model(data, file_path, file_name, type, logger=None):
    if type == "final":
        dir_path = "models/final/" + file_path
    elif type == "optimal":
        dir_path = "models/optimal/" + file_path
    elif type == "checkpoint":
        dir_path = "models/checkpoints/" + file_path
    else:
        raise ValueError("Invalid model saving type")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    model_path = os.path.join(dir_path, file_name + ".pt")

    if logger is not None:
        logger.info("Saving model to {}".format(model_path))
    else:
        print("Saving model to {}".format(model_path))

    torch.save(data, model_path)

    return model_path
