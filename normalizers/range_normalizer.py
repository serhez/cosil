from typing import Any, Dict

import numpy as np
import torch

from .normalizer import Normalizer


class RangeNormalizer(Normalizer):
    """
    A normalizer which performs range normalization of a tensor.
    The normalization is done by subtracting the minimum or mean value from the tensor and dividing over the range, i.e.: ((value - min_or_mean) / (max - min)) * gamma + beta.
    The mode of normalization can be set using the `mode` parameter, which determines if the minimum or the mean value will be subtracted.
    Subtracting the minimum value will yield a range of values of [beta, gamma + beta]; subtracting the mean value will yield a range of [-gamma + beta, gamma + beta].
    """

    def __init__(self, mode: str = "min", gamma: float = 1.0, beta: float = 0.0):
        """
        Parameters
        ----------
        mode -> the mode to use for normalization, with possible values:
        - "min" -> subtract the minimum value.
        - "mean" -> subtract the mean value.
        gamma -> the gamma scaling parameter, which is multiplied by the normalized values.
        beta -> the beta scaling parameter, which is added to the normalized values.
        """
        super().__init__(gamma, beta)

        # Running statistical measures
        self._max = -np.inf
        self._min = np.inf
        self._sum = 0.0
        self._count = 0

        if mode not in ["min", "mean"]:
            raise ValueError(f"Invalid mode: {mode}")
        self._mode = mode

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        # Update the statistical measures
        self._max = max(self._max, tensor.max().item())
        self._min = min(self._min, tensor.min().item())
        self._sum += tensor.sum().item()
        self._count += tensor.numel()

        if self._mode == "min":
            return (tensor - self._min) / (self._max - self._min)
        elif self._mode == "mean":
            return (tensor - self._sum / self._count) / (self._max - self._min)
        else:
            raise ValueError(f"Invalid normalizer mode: {self._mode}")

    def get_model_dict(self) -> Dict[str, Any]:
        """
        Get the normalizer's parameters.

        Parameters
        ----------
        None.

        Returns
        -------
        A dictionary of the normalizer's parameters, containing the following keys:
        - mode -> the mode to use for normalization, with possible values:
          - "min" -> subtract the minimum value.
          - "mean" -> subtract the mean value.
        - gamma -> the gamma scaling parameter, which is multiplied by the normalized values.
        - beta -> the beta scaling parameter, which is added to the normalized values.
        - max -> the maximum value.
        - min -> the minimum value.
        - sum -> the sum of all values.
        - count -> the number of values.
        """
        model_dict = super().get_model_dict()

        model_dict.update(
            {
                "mode": self._mode,
                "max": self._max,
                "min": self._min,
                "sum": self._sum,
                "count": self._count,
            }
        )

        return model_dict

    def load(self, model: Dict[str, Any]):
        """
        Load the normalizer's parameters from a model.

        Parameters
        ----------
        model -> the model dictionary, containing the following keys:
        - mode -> the mode to use for normalization, with possible values:
          - "min" -> subtract the minimum value.
          - "mean" -> subtract the mean value.
        - gamma -> the gamma scaling parameter, which is multiplied by the normalized values.
        - beta -> the beta scaling parameter, which is added to the normalized values.
        - max -> the maximum value.
        - min -> the minimum value.
        - sum -> the sum of all values.
        - count -> the number of values.

        Returns
        -------
        None.

        Raises
        ------
        ValueError -> if the model is invalid.
        """
        super().load(model)

        try:
            self._mode = model["mode"]
            self._max = model["max"]
            self._min = model["min"]
            self._sum = model["sum"]
            self._count = model["count"]
        except KeyError as e:
            raise ValueError(f"Invalid model: {model}") from e
