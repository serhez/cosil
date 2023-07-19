from typing import Any, Dict, Optional

import torch

from .normalizer import Normalizer


class ScaleShiftNormalizer(Normalizer):
    """
    A normalizer which only scales and shifts the values of a tensor, without any further normalization.
    The scaling is parameterized by `gamma` and the shifting by `beta`.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        beta: float = 0.0,
        low_clip: Optional[float] = None,
        high_clip: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        gamma -> the gamma scaling parameter, which is multiplied by the normalized values.
        beta -> the beta shifting parameter, which is added to the normalized values.
        low_clip -> the lower bound for clipping; not applied if set to None.
        high_clip -> the higher bound for clipping; not applied if set to None.
        """
        super().__init__(gamma, beta, low_clip, high_clip)

    def update_stats(self, _: torch.Tensor) -> None:
        # There are no statistics to update.
        return

    def _normalize_impl(self, tensor: torch.Tensor) -> torch.Tensor:
        # An identity function; the scaling and shifting is performed by the parent `Normalizer` class.
        return tensor

    def _get_model_dict_impl(self) -> Dict[str, Any]:
        return {}

    def _load_impl(self, _: Dict[str, Any]):
        return
