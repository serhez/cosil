from typing import Any, Dict, Optional

import numpy as np
import torch

from .normalizer import Normalizer


class ZScoreNormalizer(Normalizer):
    """
    A normalizer which performs z-score normalization of a tensor (a.k.a. standardization).
    The normalization is done by subtracting the minimum or mean value from the tensor and dividing over the standard deviation, i.e.: ((value - mean_or_min) / std) * gamma + beta.
    The mode of normalization can be set using the `mode` parameter, which determines if the minimum or the mean value will be subtracted.
    The normalization will yield values with a standard deviation of 1 (affected also by gamma). If the mode is set to "mean", the values will also have a mean of 0 (affected also by beta).
    Clipping can also be activated using either or both of the `low_clip` and `high_clip` parameters, which will clip the normalized values to the range [low_clip, high_clip].
    """

    EPS = 1e-8
    """A small value to avoid division by zero."""

    def __init__(
        self,
        mode: str = "mean",
        gamma: float = 1.0,
        beta: float = 0.0,
        low_clip: Optional[float] = None,
        high_clip: Optional[float] = None,
        fixed_min: Optional[float] = None,
        fixed_mean: Optional[float] = None,
        fixed_std: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        `mode` -> the mode to use for normalization, with possible values:
        - "mean" -> subtract the mean value.
        - "min" -> subtract the minimum value.
        `gamma` -> the gamma scaling parameter, which is multiplied by the normalized values.
        `beta` -> the beta scaling parameter, which is added to the normalized values.
        `low_clip` -> the lower bound for clipping; not applied if set to `None`.
        `high_clip` -> the higher bound for clipping; not applied if set to `None`.
        `fixed_min` -> a fixed minimum value to use for normalization; if set to `None`, the minimum value will be estimated given the values provided to `normalize()`.
        `fixed_mean` -> a fixed mean value to use for normalization; if set to `None`, the mean value will be estimated given the values provided to `normalize()`.
        `fixed_std` -> a fixed standard deviation value to use for normalization; if set to `None`, the standard deviation will be estimated given the values provided to `normalize()`.
        """
        super().__init__(gamma, beta, low_clip, high_clip)

        # The mode to use for normalization
        if mode not in ["min", "mean"]:
            raise ValueError(f"Invalid mode: {mode}")
        self._mode = mode

        # Fixed values for normalization
        self._fixed_min = fixed_min
        self._fixed_mean = fixed_mean
        self._fixed_std = fixed_std

        # Running statistical measures recorded to compute the mean and standard deviation
        self._min = np.inf
        self._sum = 0.0
        self._count = 0
        self._sqrd_sum = 0.0

        # Statistical measures recorded for reporting
        self._mean = 0.0
        self._std = 0.0

    def update_stats(self, tensor: torch.Tensor) -> None:
        self._min = min(self._min, tensor.min().item())
        self._sum += tensor.sum().item()
        self._count += tensor.numel()
        self._sqrd_sum += (tensor**2).sum().item()

        self._mean = self._sum / self._count
        self._std = np.sqrt((self._sqrd_sum / self._count) - (self._mean**2))

    def _normalize_impl(self, tensor: torch.Tensor) -> torch.Tensor:
        min = self._fixed_min if self._fixed_min is not None else self._min
        mean = self._fixed_mean if self._fixed_mean is not None else self._mean
        std = self._fixed_std if self._fixed_std is not None else self._std

        if self._mode == "min":
            sub_tensor = tensor - min
        elif self._mode == "mean":
            sub_tensor = tensor - mean
        else:
            raise ValueError(f"Invalid normalization mode: {self._mode}")

        return sub_tensor / (std + self.EPS)

    def _get_model_dict_impl(self) -> Dict[str, Any]:
        return {
            "mode": self._mode,
            "sum": self._sum,
            "count": self._count,
            "min": self._min,
            "squared_sum": self._sqrd_sum,
            "fixed_min": self._fixed_min,
            "fixed_mean": self._fixed_mean,
            "fixed_max": self._fixed_max,
            # For reporting only
            "mean": self._mean,
            "std": self._std,
        }

    def _load_impl(self, model: Dict[str, Any]):
        try:
            self._mode = model["mode"]
            self._sum = model["sum"]
            self._count = model["count"]
            self._min = model["min"]
            self._sqrd_sum = model["squared_sum"]
            self._fixed_min = model["fixed_min"]
            self._fixed_mean = model["fixed_mean"]
            self._fixed_max = model["fixed_max"]
            # For reporting only
            self._mean = model["mean"]
            self._std = model["std"]
        except KeyError:
            raise ValueError("Invalid model for ZScoreNormalizer")
