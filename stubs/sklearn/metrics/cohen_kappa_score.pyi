from typing import Optional, Union

from numpy import ndarray
from pandas import DataFrame

def cohen_kappa_score(
    y1: Union[ndarray, DataFrame],
    y2: Union[ndarray, DataFrame],
    labels: Optional[Union[ndarray, DataFrame]] = ...,
    weights: Optional[str] = ...,
    sample_weight: Optional[ndarray] = ...,
) -> float: ...
