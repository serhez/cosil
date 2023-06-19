from typing import Optional

from .normalizer import Normalizer
from .range_normalizer import RangeNormalizer
from .scale_shift_normalizer import ScaleShiftNormalizer
from .z_score_normalizer import ZScoreNormalizer


def create_normalizer(
    name: str,
    mode: str,
    gamma: float,
    beta: float,
    low_clip: Optional[float],
    high_clip: Optional[float],
) -> Optional[Normalizer]:
    if name == "range":
        return RangeNormalizer(mode, gamma, beta)
    elif name == "z_score":
        return ZScoreNormalizer(mode, gamma, beta, low_clip, high_clip)
    elif name == "scale_shift":
        return ScaleShiftNormalizer(gamma, beta, low_clip, high_clip)
    elif name == "none":
        return None
    else:
        raise ValueError("Unknown normalizer: {}".format(name))
