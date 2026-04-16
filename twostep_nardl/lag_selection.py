"""
lag_selection.py — AIC/BIC grid search for NARDL lag order
============================================================
Translates the nested lag-selection loops from twostep_nardl.ado.
"""
from __future__ import annotations
import numpy as np
import itertools
from typing import Optional


def _compute_ic(y: np.ndarray, X: np.ndarray, ic: str = "bic") -> float:
    """OLS regression and return AIC or BIC."""
    T = len(y)
    XtX = X.T @ X
    try:
        b = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        return np.inf
    resid = y - X @ b
    rss = resid @ resid
    sigma2 = rss / T
    if sigma2 <= 0:
        return np.inf
    ll = -0.5 * T * (np.log(2 * np.pi) + np.log(sigma2) + 1)
    k = X.shape[1]
    if ic == "aic":
        return -2 * ll + 2 * k
    else:  # bic
        return -2 * ll + k * np.log(T)


def select_lags(
    endog: np.ndarray,            # shape (T,) — first differences of depvar
    ect: np.ndarray,              # shape (T,) — error correction term (lagged)
    depvar_levels: np.ndarray,    # shape (T,) — depvar levels (for lagged diffs)
    pos_vars: list[np.ndarray],   # list of (T,) arrays — x_pos for each asym var
    neg_vars: list[np.ndarray],   # list of (T,) arrays — x_neg for each asym var
    lin_vars: list[np.ndarray],   # list of (T,) arrays — linear vars
    exog: list[np.ndarray],       # list of (T,) arrays — exogenous
    trendvar: Optional[np.ndarray],
    lag_fixed: list[Optional[int]],  # fixed lag for each var (None = auto-select)
    max_lags: list[int],             # max lags for each var
    include_const: bool = True,
    ic: str = "bic",
    min_lag_dep: int = 1,
) -> list[int]:
    """
    Grid search over lag combinations to minimise AIC or BIC.

    Parameters
    ----------
    endog : first differences of dependent variable (D.y)
    ect   : lagged error correction term (L.ect)
    depvar_levels : levels of depvar for constructing lagged diffs
    pos_vars, neg_vars : positive/negative partial sums for each asym variable
    lin_vars : linear (non-decomposed) variables
    exog : exogenous regressors
    trendvar : optional trend variable
    lag_fixed : list (len = 1 + n_asym + n_lin) — None means optimise
    max_lags : list (len = 1 + n_asym + n_lin) — max search depth
    include_const : bool
    ic : 'aic' or 'bic'
    min_lag_dep : minimum lag for depvar (default 1)

    Returns
    -------
    list of int — optimal lags [p, q1, q2, ..., qk, r1, ..., rm]
    """
    T = len(endog)
    n_asym = len(pos_vars)
    n_lin = len(lin_vars)
    n_vars = 1 + n_asym + n_lin

    # Build search grids
    search_ranges = []
    # Depvar lag p (must be >= 1 for ECM)
    p_fixed = lag_fixed[0]
    if p_fixed is not None:
        search_ranges.append([p_fixed])
    else:
        search_ranges.append(list(range(min_lag_dep, max_lags[0] + 1)))

    # Asymmetric variable lags q_i (can be 0)
    for i in range(n_asym):
        q_fixed = lag_fixed[1 + i]
        if q_fixed is not None:
            search_ranges.append([q_fixed])
        else:
            search_ranges.append(list(range(0, max_lags[1 + i] + 1)))

    # Linear variable lags r_i
    for i in range(n_lin):
        r_fixed = lag_fixed[1 + n_asym + i]
        if r_fixed is not None:
            search_ranges.append([r_fixed])
        else:
            search_ranges.append(list(range(0, max_lags[1 + n_asym + i] + 1)))

    best_ic = np.inf
    best_lags = [r[0] for r in search_ranges]

    # Extra regressors that don't change across the grid
    extra_cols = []
    if exog:
        for ev in exog:
            extra_cols.append(ev.reshape(-1, 1))
    if trendvar is not None:
        extra_cols.append(trendvar.reshape(-1, 1))
    if include_const:
        extra_cols.append(np.ones((T, 1)))

    for combo in itertools.product(*search_ranges):
        p = combo[0]
        qs = list(combo[1:1 + n_asym])
        rs = list(combo[1 + n_asym:])

        # Build regressors
        cols = [ect.reshape(-1, 1)]  # L.ect

        # Lagged Dy
        for j in range(1, p):
            dlag = np.zeros(T)
            dlag[j:] = endog[:T - j]
            cols.append(dlag.reshape(-1, 1))

        # Dx_pos and Dx_neg lags
        for i in range(n_asym):
            q_use = max(qs[i], 1)
            dx_pos = np.zeros(T)
            dx_pos[1:] = np.diff(pos_vars[i])
            dx_neg = np.zeros(T)
            dx_neg[1:] = np.diff(neg_vars[i])
            for j in range(q_use):
                if j == 0:
                    cols.append(dx_pos.reshape(-1, 1))
                    cols.append(dx_neg.reshape(-1, 1))
                else:
                    lagged_p = np.zeros(T)
                    lagged_p[j:] = dx_pos[:T - j]
                    lagged_n = np.zeros(T)
                    lagged_n[j:] = dx_neg[:T - j]
                    cols.append(lagged_p.reshape(-1, 1))
                    cols.append(lagged_n.reshape(-1, 1))

        # Dz lags
        for i in range(n_lin):
            r_use = max(rs[i], 1)
            dz = np.zeros(T)
            dz[1:] = np.diff(lin_vars[i])
            for j in range(r_use):
                if j == 0:
                    cols.append(dz.reshape(-1, 1))
                else:
                    lagged_z = np.zeros(T)
                    lagged_z[j:] = dz[:T - j]
                    cols.append(lagged_z.reshape(-1, 1))

        cols.extend(extra_cols)
        X = np.concatenate(cols, axis=1)

        # Drop rows with NaN (lags)
        max_lag_used = max(p, max(qs) if qs else 0, max(rs) if rs else 0)
        start = max_lag_used
        Xs = X[start:]
        ys = endog[start:]
        if Xs.shape[0] < Xs.shape[1] + 5:
            continue

        ic_val = _compute_ic(ys, Xs, ic=ic)
        if ic_val < best_ic:
            best_ic = ic_val
            best_lags = list(combo)

    return best_lags
