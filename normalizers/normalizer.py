from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import torch


class Normalizer(ABC):
    """
    This class describes an interface for normalizer objects.
    """

    def __init__(self, gamma: float, beta: float):
        """
        Initialize the normalizer.

        Parameters
        ----------
        gamma -> the gamma scaling parameter, which is multiplied by the normalized values.
        beta -> the beta scaling parameter, which is added to the normalized values.

        Returns
        -------
        None.
        """

        self._gamma = gamma
        self._beta = beta

    @abstractmethod
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
        pass

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
        except KeyError as e:
            raise ValueError(f"Invalid model: {model}") from e
