from typing import Any, Dict, Optional

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

    def __init__(
        self,
        mode: str = "min",
        gamma: float = 1.0,
        beta: float = 0.0,
        low_clip: Optional[float] = None,
        high_clip: Optional[float] = None,
        fixed_min: Optional[float] = None,
        fixed_mean: Optional[float] = None,
        fixed_max: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        `mode` -> the mode to use for normalization, with possible values:
        - "min" -> subtract the minimum value.
        - "mean" -> subtract the mean value.
        `gamma` -> the gamma scaling parameter, which is multiplied by the normalized values.
        `beta` -> the beta scaling parameter, which is added to the normalized values.
        `low_clip` -> the lower bound for clipping; not applied if set to `None`.
        `high_clip` -> the higher bound for clipping; not applied if set to `None`.
        `fixed_min` -> a fixed minimum value to use for normalization; if set to `None`, the minimum value will be estimated given the values provided to `normalize()`.
        `fixed_mean` -> a fixed mean value to use for normalization; if set to `None`, the mean value will be estimated given the values provided to `normalize()`.
        `fixed_max` -> a fixed maximum value to use for normalization; if set to `None`, the standard deviation will be estimated given the values provided to `normalize()`.
        """
        super().__init__(gamma, beta, low_clip, high_clip)

        # The mode to use for normalization
        if mode not in ["min", "mean"]:
            raise ValueError(f"Invalid mode: {mode}")
        self._mode = mode

        # Running statistical measures
        self._max = -np.inf
        self._min = np.inf
        self._sum = 0.0
        self._count = 0

    def update_stats(self, tensor: torch.Tensor) -> None:
        self._max = max(self._max, tensor.max().item())
        self._min = min(self._min, tensor.min().item())
        self._sum += tensor.sum().item()
        self._count += tensor.numel()

    def _normalize_impl(self, tensor: torch.Tensor) -> torch.Tensor:
        min = self._min if self._fixed_min is None else self._fixed_min
        mean = self._sum / self._count if self._fixed_mean is None else self._fixed_mean
        max = self._max if self._fixed_max is None else self._fixed_max

        if self._mode == "min":
            return (tensor - min) / (max - min)
        elif self._mode == "mean":
            return (tensor - mean) / (max - min)
        else:
            raise ValueError(f"Invalid normalizer mode: {self._mode}")

    def get_model_dict(self) -> Dict[str, Any]:
        """
        Get the normalizer's parameters.

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
        - low_clip -> the lower bound for clipping.
        - high_clip -> the higher bound for clipping.
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
        - low_clip -> the lower bound for clipping.
        - high_clip -> the higher bound for clipping.

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
