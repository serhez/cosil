import os
import time

import torch

from common.observation_buffer import ObservationBuffer


def save(buffer: ObservationBuffer, path: str, id: str):
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

    torch.save(buffer.to_list(), buffer_path)

    return buffer_path


def main():
    """Merges all individual buffers present in data/replay_buffers/individual/ into a single buffer and saves it to data/replay_buffers/merged/."""

    # Get individual buffers to merge
    obs_list = []
    dir_path = "data/replay_buffers/individual/"
    for buffer_path in os.listdir(dir_path):
        if buffer_path.endswith(".pt"):
            obs = torch.load(dir_path + buffer_path)
            print(f"Loading buffer {buffer_path} with {len(obs)} observations")
            obs_list.append(obs)

    # Merge buffers
    capacity = sum([len(obs) for obs in obs_list])
    print(f"Merging {len(obs_list)} buffers with {capacity} observations in total")
    buffer = ObservationBuffer(capacity=capacity)
    for obs in obs_list:
        buffer.push(obs)

    # Save the merged buffer
    buffer_path = save(buffer, "data/replay_buffers/merged/", str(int(time.time())))
    print(f"Saved merged buffer to {buffer_path}")


if __name__ == "__main__":
    main()
