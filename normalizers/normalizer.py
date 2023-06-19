from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import torch


class Normalizer(ABC):
    """
    This class describes an interface for normalizer objects.
    """

    def __init__(
        self,
        gamma: float,
        beta: float,
        low_clip: Optional[float] = None,
        high_clip: Optional[float] = None,
    ):
        """
        Initialize the normalizer.

        Parameters
        ----------
        gamma -> the gamma scaling parameter, which is multiplied by the normalized values.
        beta -> the beta scaling parameter, which is added to the normalized values.
        low_clip -> the lower bound for clipping; not applied if set to None.
        high_clip -> the higher bound for clipping; not applied if set to None.
        """

        self._gamma = gamma
        self._beta = beta
        self._low_clip = low_clip
        self._high_clip = high_clip

    @abstractmethod
    def _normalize_impl(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        The internal child class implementation of the normalization method.

        Parameters
        ----------
        tensor -> the tensor to normalize.

        Returns
        -------
        The normalized tensor.
        """
        raise NotImplementedError

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor.

        Parameters
        ----------
        tensor -> the tensor to normalize.

        Returns
        -------
        The normalized tensor.
        """

        # Obtain the normalized tensor given by the child class implementation
        normalized_tensor = self._normalize_impl(tensor)

        # Apply the gamma and beta parameters
        normalized_tensor *= self._gamma
        normalized_tensor += self._beta

        # Clip the normalized tensor if necessary
        if self._low_clip is not None or self._high_clip is not None:
            normalized_tensor = torch.clamp(
                normalized_tensor, min=self._low_clip, max=self._high_clip
            )

        return normalized_tensor

    def _call_impl(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    def get_model_dict(self) -> Dict[str, Any]:
        """
        Get the normalizer's parameters.

        Parameters
        ----------
        None.

        Returns
        -------
        A dictionary of the normalizer's parameters.
        """

        model_dict = {
            "gamma": self._gamma,
            "beta": self._beta,
            "low_clip": self._low_clip,
            "high_clip": self._high_clip,
        }

        return model_dict

    def load(self, model: Dict[str, Any]):
        """
        Load the normalizer's parameters from a model.

        Parameters
        ----------
        model -> a dictionary of the normalizer's parameters.

        Returns
        -------
        None.
        """

        try:
            self._gamma = model["gamma"]
            self._beta = model["beta"]
            self._low_clip = model["low_clip"]
            self._high_clip = model["high_clip"]
        except KeyError as e:
            raise ValueError(f"Invalid model: {model}") from e
