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
    ):
        """
        Parameters
        ----------
        mode -> the mode to use for normalization, with possible values:
        - "mean" -> subtract the mean value.
        - "min" -> subtract the minimum value.
        gamma -> the gamma scaling parameter, which is multiplied by the normalized values.
        beta -> the beta scaling parameter, which is added to the normalized values.
        low_clip -> the lower bound for clipping; not applied if set to None.
        high_clip -> the higher bound for clipping; not applied if set to None.
        """
        super().__init__(gamma, beta)

        # Running statistical measures recorded to compute the mean and standard deviation
        self._min = np.inf
        self._sum = 0.0
        self._count = 0
        self._sqrd_sum = 0.0

        # Statistical measures recorded for reporting
        self._mean = 0.0
        self._std = 0.0

        self._low_clip = low_clip
        self._high_clip = high_clip

        if mode not in ["min", "mean"]:
            raise ValueError(f"Invalid mode: {mode}")
        self._mode = mode

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        # Update the statistical measures
        self._min = min(self._min, tensor.min().item())
        self._sum += tensor.sum().item()
        self._count += tensor.numel()
        self._sqrd_sum += (tensor**2).sum().item()

        self._mean = self._sum / self._count
        self._std = np.sqrt((self._sqrd_sum / self._count) - (self._mean**2))

        if self._mode == "min":
            sub_tensor = tensor - self._min
        elif self._mode == "mean":
            sub_tensor = tensor - self._mean
        else:
            raise ValueError(f"Invalid normalization mode: {self._mode}")

        normalized_tensor = (
            sub_tensor / (self._std + self.EPS) * self._gamma + self._beta
        )

        if self._low_clip is not None or self._high_clip is not None:
            normalized_tensor = torch.clamp(
                normalized_tensor, min=self._low_clip, max=self._high_clip
            )

        return normalized_tensor

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
        - mode -> the mode of normalization.
        - sum -> the running sum of all values.
        - count -> the number of values.
        - min -> the minimum value.
        - squared_sum -> the running sum of the squared values.
        - low_clip -> the lower bound for clipping.
        - high_clip -> the higher bound for clipping.
        - mean -> the running mean value (only reported, not needed to restore the state of the normalizer).
        - std -> the running standard deviation (only reported, not needed to restore the state of the normalizer).
        """
        model_dict = super().get_model_dict()

        model_dict.update(
            {
                "mode": self._mode,
                "sum": self._sum,
                "count": self._count,
                "min": self._min,
                "squared_sum": self._sqrd_sum,
                "low_clip": self._low_clip,
                "high_clip": self._high_clip,
                # For reporting only
                "mean": self._mean,
                "std": self._std,
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
        - mode -> the mode of normalization.
        - sum -> the sum of all values.
        - count -> the number of values.
        - min -> the minimum value.
        - squared_sum -> the sum of the squared values.
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
            self._sum = model["sum"]
            self._count = model["count"]
            self._min = model["min"]
            self._sqrd_sum = model["squared_sum"]
            self._low_clip = model["low_clip"]
            self._high_clip = model["high_clip"]
        except KeyError as e:
            raise ValueError(f"Invalid model: {model}") from e
