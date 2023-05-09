# From https://github.com/pranz24/pytorch-soft-actor-critic (MIT Licensed)

import os
import pickle
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch


class ObservationBuffer:
    """
    A simple FIFO observation buffer which can be used as a replay buffer or imitation buffer.
    """

    def __init__(self, capacity: int, diminishing_ratio: float = 1.0):
        """
        Initializes a replay buffer.

        Parameters
        ----------
        `capacity` -> the maximum number of observations to store in the buffer.
        - Must be a non-zero positive integer.
        `diminishing_ratio` -> the ratio by which the sampling weight of past observations diminishes when new observations are added.
        - Must be a float in the range (0.0, 1.0].
        - If 1.0, all observations have equal probability of being sampled.
        - Observations are assigned weights in the order they are pushed into the buffer
        - If a list of observations is pushed into the buffer by the same `push()` operation, all such observations are assigned the same weight, regardless of their position in the list.
        """

        assert (
            capacity > 0
        ), "The capacity of an observation buffer must be a non-zero positive integer."

        assert (
            0.0 < diminishing_ratio <= 1.0
        ), "The diminishing ratio of an observation buffer must be a float between 0.0 and 1.0."

        self._capacity = capacity
        self._buffer = []
        self._position = 0
        self._age = np.array([], dtype=np.int64)
        self._diminishing_ratio = diminishing_ratio

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, key: int) -> Any:
        return self._buffer[key]

    @property
    def capacity(self) -> int:
        """
        The capacity of the buffer.
        """

        return self._capacity

    @property
    def distribution(self) -> np.ndarray:
        """
        The distribution of the observations in the buffer.
        """

        distribution = np.ones(len(self._age), dtype=np.float64)
        distribution *= self._diminishing_ratio**self._age
        total = distribution.sum()
        if total > 0.0:  # should always be true unless the buffer is empty
            distribution /= total

        return distribution

    def get_element_shapes(self) -> Optional[List[torch.Size]]:
        """
        Returns the shape of the elements in the buffer.
        Returns None if any of the elements is not a torch tensor.
        """

        shapes = []
        for element in self._buffer:
            if not torch.is_tensor(element):
                return None
            shapes.append(element.shape)
        return shapes

    def to_list(self) -> List[Any]:
        """
        Returns a copy of the contents of the buffer as a list.
        Repeated calls to this method should be avoided, as it creates a copy of the buffer each time; instead, the copy should be stored and reused.
        """

        return self._buffer.copy()

    def clear(self) -> None:
        """
        Clears the replay buffer.
        """

        self._buffer = []
        self._position = 0
        self._age = np.array([], dtype=np.int64)

    def set(self, observations: List[Any]) -> None:
        """
        Sets the contents of the buffer to the given observations.
        If the length of the observations list is greater than the buffer's capacity, the oldest observations are discarded.
        As a side effect, the buffer's position is set to 0 if the length of the observations equals the capacity of the buffer.
        """

        self._buffer = observations[: self._capacity]
        buffer_len = len(self._buffer)
        self._position = 0 if buffer_len == self._capacity else buffer_len
        self._age = np.zeros(buffer_len, dtype=np.int64)

    def _push_impl(self, observation: Any) -> None:
        """
        Pushes an observation into the buffer, replacing the oldest observation if the buffer is full.

        Parameters
        ----------
        observation -> the observation to push into the buffer.
        """

        if len(self._buffer) < self._capacity:
            self._buffer.append(None)
        self._buffer[self._position] = observation
        self._position = (self._position + 1) % self._capacity

    def push(self, observations: Union[Any, List[Any]]) -> None:
        """
        Pushes an observation or a list of observations into the buffer, replacing the oldest observations if the buffer is full.
        It also reduces the weights of previous observations by the `diminishing_ratio`.
        Given a list of observations, it assigns the same weight to all new observations, regardless of their position in the list.
        If the number of observations to push into the buffer is greater than the buffer's capacity, an AssertionError is raised.
        If an empty list of observations is pushed into the buffer, nothing happens.

        Parameters
        ----------
        observations -> the observation/s to push into the buffer.

        Raises
        ----------
        AssertionError -> if the number of observations to push into the buffer is greater than the buffer's capacity.
        """

        if isinstance(observations, list):
            if len(observations) == 0:
                return
            assert (
                len(observations) <= self._capacity
            ), "The number of observations to push into the buffer must be less than or equal to the buffer's capacity."

        # Update the age of old observations
        self._age += 1

        # Increase the size of the age array if needed
        n_obs = len(observations) if isinstance(observations, list) else 1
        new_len = min(self._capacity, len(self._age) + n_obs)
        increase = new_len - len(self._age)
        self._age = np.append(self._age, np.empty(increase, dtype=np.int64))

        # Update the age of new observations
        new_obs_idxs = [(self._position + i) % self._capacity for i in range(n_obs)]
        self._age[new_obs_idxs] = 0

        # Push the new observations
        if isinstance(observations, list):
            for observation in observations:
                self._push_impl(observation)
        else:
            self._push_impl(observations)

    def sample(self, n: int = 1) -> Optional[Tuple[Any, ...]]:
        """
        Samples n observations from the buffer.
        The sampling is done without replacement, unless the buffer contains less than n observations.

        Parameters
        ----------
        n -> the number of observations to sample.

        Returns
        ----------
        A tuple of n observations sampled from the buffer.

        Raises
        ----------
        ValueError -> if the buffer is empty.
        """

        if len(self._buffer) == 0:
            raise ValueError(
                "Could not sample from the observation buffer because it is empty"
            )

        replace = len(self._buffer) < n

        batch_list = np.random.choice(
            self._buffer, size=n, replace=replace, p=self.distribution
        )

        return tuple(map(np.stack, zip(*batch_list)))

    # TODO: Change this for a `get_buffer` method that returns the buffer, and then use that in the `save` method
    def save(self, env_name, suffix="", save_path=None):
        """
        Saves the buffer to a file.

        Parameters
        ----------
        env_name -> the name of the environment.
        suffix -> the suffix to append to the file name.
        save_path -> the path to save the buffer to. If None, saves to "checkpoints/sac_buffer_{env_name}_{suffix}".
        """

        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)

        with open(save_path, "wb") as f:
            pickle.dump(self._buffer, f)

    # TODO: This does not load the capacity nor the age
    def load(self, save_path):
        """
        Loads the buffer from a file.

        Parameters
        ----------
        save_path -> the path to load the buffer from.
        """

        with open(save_path, "rb") as f:
            self._buffer = pickle.load(f)
            self._position = len(self._buffer) % self._capacity
