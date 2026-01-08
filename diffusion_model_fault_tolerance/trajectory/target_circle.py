from __future__ import annotations
from typing import Tuple
import numpy as np

def generate_circle(t:int = 50, loops: int = 3, radius: float = 0.85, z: float = 0.7, rounding: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 円軌道の(X, Y, Z)を返す．
    n = t * loops
    i = np.arange(n, dtype= np.float64)
    
    x = radius * np.cos(i / (t - 1) * 2.0 * np.pi)
    y = radius * np.sin(i / (t - 1) * 2.0 * np.pi)
    z_arr = np.full(n, z, dtype=np.float64)
    
    if rounding is not None:
        x = np.round(x, rounding)
        y = np.round(y, rounding)

    return x, y, z_arr
