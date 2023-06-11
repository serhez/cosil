import os
import time
from typing import List

import numpy as np
import torch

from common.observation_buffer import ObservationBuffer


def save(buffer: ObservationBuffer, morphos: List[np.ndarray], path: str, id: str):
    """
    Saves the buffer to a file.

    Parameters
    ----------
    buffer -> the observation buffer.
    path -> the path to save the buffer.
    id -> the id of the buffer.
    """

    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception:
        raise ValueError("Invalid path")

    buffer_path = os.path.join(path, f"buffer_{id}.pt")

    data = {
        "buffer": buffer.to_list(),
        "morpho": morphos,
    }
    torch.save(data, buffer_path)

    return buffer_path


def main():
    """Merges all individual buffers present in data/replay_buffers/individual/ into a single buffer and saves it to data/replay_buffers/merged/."""

    # Get individual buffers to merge and morphologies
    obs_list = []
    morpho_list = []
    dir_path = "data/replay_buffers/individual/"
    for buffer_path in os.listdir(dir_path):
        if buffer_path.endswith(".pt"):
            data = torch.load(dir_path + buffer_path)
            obs = data["buffer"]
            print(f"Loading buffer {buffer_path} with {len(obs)} observations")
            obs_list.append(obs)
            morpho_list.append(data["morpho"])

    # Merge buffers
    capacity = sum([len(obs) for obs in obs_list])
    print(f"Merging {len(obs_list)} buffers with {capacity} observations in total")
    buffer = ObservationBuffer(capacity=capacity)
    for obs in obs_list:
        buffer.push(obs)

    # Save the merged buffer
    buffer_path = save(
        buffer, morpho_list, "data/replay_buffers/merged/", str(int(time.time()))
    )
    print(f"Saved merged buffer to {buffer_path}")


if __name__ == "__main__":
    main()
