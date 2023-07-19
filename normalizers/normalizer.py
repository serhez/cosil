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
    def update_stats(self, tensor: torch.Tensor) -> None:
        """
        Update the normalizer's statistics without normalizing the tensor.

        Parameters
        ----------
        tensor -> the tensor to update the statistics with.

        Returns
        -------
        None.
        """

        raise NotImplementedError

    @abstractmethod
    def _normalize_impl(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        The internal child class implementation of the normalization method.
        It should not update the internal running statistics, as this is done by the
        parent class.

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
        Normalizes a tensor and updates the internal running statistics.

        Parameters
        ----------
        tensor -> the tensor to normalize.

        Returns
        -------
        The normalized tensor.
        """

        # First update the running statistics
        self.update_stats(tensor)

        # Obtain the normalized tensor given by the child class implementation
        norm_t = self._normalize_impl(tensor)

        # Apply the gamma and beta parameters
        norm_scaled_t = norm_t * self._gamma
        norm_scaled_t += self._beta

        # Clip the normalized tensor if necessary
        if self._low_clip is not None or self._high_clip is not None:
            norm_scaled_t = torch.clamp(
                norm_scaled_t, min=self._low_clip, max=self._high_clip
            )

        return norm_scaled_t

    def _call_impl(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    @abstractmethod
    def _get_model_dict_impl(self) -> Dict[str, Any]:
        """
        The internal child-class-specfic implementation of `get_model_dict`.
        Do not call this method directly.

        Parameters
        ----------
        None.

        Returns
        -------
        A dictionary of the normalizer's parameters.
        """

        raise NotImplementedError

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
        model_dict.update(self._get_model_dict_impl())

        return model_dict

    @abstractmethod
    def _load_impl(self, model: Dict[str, Any]):
        """
        The internal child-class-specfic implementation of `load`.
        Do not call this method directly.
        When implementing this method, make sure not to overwrite the parameters `gamma`,
        `beta`, `low_clip`, and `high_clip`.

        Parameters
        ----------
        model -> a dictionary of the normalizer's parameters.

        Returns
        -------
        None.

        Raises
        ------
        ValueError -> if the model is invalid.
        """

        raise NotImplementedError

    def load(self, model: Dict[str, Any]):
        """
        Load the normalizer's parameters from a model.

        Parameters
        ----------
        model -> a dictionary of the normalizer's parameters.

        Returns
        -------
        None.

        Raises
        ------
        ValueError -> if the model is invalid.
        """

        try:
            self._gamma = model["gamma"]
            self._beta = model["beta"]
            self._low_clip = model["low_clip"]
            self._high_clip = model["high_clip"]
        except KeyError:
            raise ValueError("Invalid normalizer model")

        self._load_impl(model)
