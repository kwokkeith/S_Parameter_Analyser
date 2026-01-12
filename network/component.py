from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Component:
    name: str
    type: str
    S_parameter: dict[
        np.int64, np.ndarray[np.complex128]
    ]  # 2 x 2 S-parameter matrix at diff freq
