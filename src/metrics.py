# metrics.py
import numpy as np


def compute_returns(price: np.ndarray) -> np.ndarray:
    """Simple arithmetic returns r_t = P_t / P_{t-1} - 1."""
    price = np.asarray(price, dtype=float)
    if price.size < 2:
        return np.array([], dtype=float)
    return price[1:] / price[:-1] - 1.0


def realised_volatility(returns: np.ndarray) -> float:
    """Realised volatility over the whole horizon (sqrt of sum of squared returns)."""
    returns = np.asarray(returns, dtype=float)
    return float(np.sqrt(np.sum(returns ** 2)))


