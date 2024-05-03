from typing import Any, List, Optional, Union

import numpy as np
from pandas import DataFrame

def cohen_kappa_score(
    y1: List[int],
    y2: List[int],
    labels: Optional[Union[np.ndarray[float, Any], DataFrame]] = ...,
    weights: Optional[str] = ...,
    sample_weight: Optional[np.ndarray[float, Any]] = ...,
) -> float: ...
def accuracy_score(
    y_true: List[int],
    y_pred: List[int],
    normalize: bool = ...,
    sample_weight: Optional[List[float]] = ...,
) -> float: ...
def f1_score(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[int]] = ...,
    pos_label: int = ...,
    average: Optional[str] = ...,
    sample_weight: Optional[List[float]] = ...,
    zero_division: str = ...,
) -> float: ...
def precision_score(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[int]] = ...,
    pos_label: int = ...,
    average: Optional[str] = ...,
    sample_weight: Optional[List[float]] = ...,
    zero_division: str = ...,
) -> float: ...
def recall_score(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[int]] = ...,
    pos_label: int = ...,
    average: Optional[str] = ...,
    sample_weight: Optional[List[float]] = ...,
    zero_division: str = ...,
) -> float: ...
