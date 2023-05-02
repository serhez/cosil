from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import torch


class Normalizer(ABC):
    """
    This class describes an interface for normalizer objects.
    """

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

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
