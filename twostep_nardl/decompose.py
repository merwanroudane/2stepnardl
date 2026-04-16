"""
decompose.py — Partial sum decomposition for NARDL
===================================================
Translates _2snardl_psum() from Mata.
"""
import numpy as np
import pandas as pd


def partial_sums(x, threshold: float = 0.0):
    """
    Decompose a time series into positive and negative partial sums.

    Parameters
    ----------
    x : array-like, shape (T,)
        Input time series (levels).
    threshold : float, optional
        Dead-band threshold (default 0). Changes smaller than |threshold|
        are classified as zero.

    Returns
    -------
    x_pos : np.ndarray, shape (T,)
        Positive partial sum: cumsum(max(Δx - threshold, 0)).
    x_neg : np.ndarray, shape (T,)
        Negative partial sum: cumsum(min(Δx - threshold, 0)).

    Examples
    --------
    >>> import numpy as np
    >>> from twostep_nardl import partial_sums
    >>> x = np.array([1.0, 1.5, 1.3, 1.8, 1.6])
    >>> xp, xn = partial_sums(x)
    """
    x = np.asarray(x, dtype=float)
    T = len(x)

    dx = np.zeros(T)
    dx[1:] = np.diff(x)

    xpos = np.zeros(T)
    xneg = np.zeros(T)

    xpos[0] = max(dx[0] - threshold, 0.0)
    xneg[0] = min(dx[0] - threshold, 0.0)

    for i in range(1, T):
        xpos[i] = xpos[i - 1] + max(dx[i] - threshold, 0.0)
        xneg[i] = xneg[i - 1] + min(dx[i] - threshold, 0.0)

    return xpos, xneg
