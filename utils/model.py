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


def load_model(file_path, env, agent, co_adapt=True, evaluate=False, logger=None):
    if logger is not None:
        logger.info("Loading model from {}".format(file_path))
    else:
        print("Loading model from {}".format(file_path))

    model = torch.load(file_path)

    if co_adapt:
        env.set_task(*model["morpho_dict"])
        env.reset()

    agent.load(model["ind_agent"], evaluate=evaluate)
