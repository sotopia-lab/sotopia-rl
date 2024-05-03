# Contents of stubs/scipy/stats.pyi

from typing import List, Optional, Tuple, Union

def spearmanr(
    a: Union[List[float], List[int]],
    b: Optional[Union[List[float], List[int]]] = None,
    axis: int = 0,
    nan_policy: str = "propagate",
) -> Tuple[float, float]: ...
