"""
critical_values.py — PSS (2001) and KS (2020) bounds-test critical values
==========================================================================

Embeds the full Pesaran-Shin-Smith (2001) asymptotic tables (F and t stats)
for Cases 1-5 and k=0..10 variables, as extracted from ardlbounds.ado.

Also provides the Kripfganz-Schneider (2020) response-surface formula
using the coefficient data shipped in the ardl Stata package.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PSS (2001) Asymptotic Critical Values
# Extracted from _ardl_getpsstable in ardlbounds.ado
#
# Rows : k = 0, 1, ..., 10  (number of decomposed x variables)
# Cols : I(0) 10%, I(1) 10%, I(0) 5%, I(1) 5%, I(0) 2.5%, I(1) 2.5%,
#         I(0) 1%,  I(1) 1%
# ---------------------------------------------------------------------------

_PSS_F = {
    1: np.array([
        [3.00, 3.00, 4.20, 4.20, 5.47, 5.47, 7.17, 7.17],
        [2.44, 3.28, 3.15, 4.11, 3.88, 4.92, 4.81, 6.02],
        [2.17, 3.19, 2.72, 3.83, 3.22, 4.50, 3.88, 5.30],
        [2.01, 3.10, 2.45, 3.63, 2.87, 4.16, 3.42, 4.84],
        [1.90, 3.01, 2.26, 3.48, 2.62, 3.90, 3.07, 4.44],
        [1.81, 2.93, 2.14, 3.34, 2.44, 3.71, 2.82, 4.21],
        [1.75, 2.87, 2.04, 3.24, 2.32, 3.59, 2.66, 4.05],
        [1.70, 2.83, 1.97, 3.18, 2.22, 3.49, 2.54, 3.91],
        [1.66, 2.79, 1.91, 3.11, 2.15, 3.40, 2.45, 3.79],
        [1.63, 2.75, 1.86, 3.05, 2.08, 3.33, 2.34, 3.68],
        [1.60, 2.72, 1.82, 2.99, 2.02, 3.27, 2.26, 3.60],
    ]),
    2: np.array([
        [3.80, 3.80, 4.60, 4.60, 5.39, 5.39, 6.44, 6.44],
        [3.02, 3.51, 3.62, 4.16, 4.18, 4.79, 4.94, 5.58],
        [2.63, 3.35, 3.10, 3.87, 3.55, 4.38, 4.13, 5.00],
        [2.37, 3.20, 2.79, 3.67, 3.15, 4.08, 3.65, 4.66],
        [2.20, 3.09, 2.56, 3.49, 2.88, 3.87, 3.29, 4.37],
        [2.08, 3.00, 2.39, 3.38, 2.70, 3.73, 3.06, 4.15],
        [1.99, 2.94, 2.27, 3.28, 2.55, 3.61, 2.88, 3.99],
        [1.92, 2.89, 2.17, 3.21, 2.43, 3.51, 2.73, 3.90],
        [1.85, 2.85, 2.11, 3.15, 2.33, 3.42, 2.62, 3.77],
        [1.80, 2.80, 2.04, 3.08, 2.24, 3.35, 2.50, 3.68],
        [1.76, 2.77, 1.98, 3.04, 2.18, 3.28, 2.41, 3.61],
    ]),
    3: np.array([
        [6.58, 6.58,  8.21,  8.21,  9.80,  9.80, 11.79, 11.79],
        [4.04, 4.78,  4.94,  5.73,  5.77,  6.68,  6.84,  7.84],
        [3.17, 4.14,  3.79,  4.85,  4.41,  5.52,  5.15,  6.36],
        [2.72, 3.77,  3.23,  4.35,  3.69,  4.89,  4.29,  5.61],
        [2.45, 3.52,  2.86,  4.01,  3.25,  4.49,  3.74,  5.06],
        [2.26, 3.35,  2.62,  3.79,  2.96,  4.18,  3.41,  4.68],
        [2.12, 3.23,  2.45,  3.61,  2.75,  3.99,  3.15,  4.43],
        [2.03, 3.13,  2.32,  3.50,  2.60,  3.84,  2.96,  4.26],
        [1.95, 3.06,  2.22,  3.39,  2.48,  3.70,  2.79,  4.10],
        [1.88, 2.99,  2.14,  3.30,  2.37,  3.60,  2.65,  3.97],
        [1.83, 2.94,  2.06,  3.24,  2.28,  3.50,  2.54,  3.86],
    ]),
    4: np.array([
        [ 5.37,  5.37,  6.29,  6.29,  7.14,  7.14,  8.26,  8.26],
        [ 4.05,  4.49,  4.68,  5.15,  5.30,  5.83,  6.10,  6.73],
        [ 3.38,  4.02,  3.88,  4.61,  4.37,  5.16,  4.99,  5.85],
        [ 2.97,  3.74,  3.38,  4.23,  3.80,  4.68,  4.30,  5.23],
        [ 2.68,  3.53,  3.05,  3.97,  3.40,  4.36,  3.81,  4.92],
        [ 2.49,  3.38,  2.81,  3.76,  3.11,  4.13,  3.50,  4.63],
        [ 2.33,  3.25,  2.63,  3.62,  2.90,  3.94,  3.27,  4.39],
        [ 2.22,  3.17,  2.50,  3.50,  2.76,  3.81,  3.07,  4.23],
        [ 2.13,  3.09,  2.38,  3.41,  2.62,  3.70,  2.93,  4.06],
        [ 2.05,  3.02,  2.30,  3.33,  2.52,  3.60,  2.79,  3.93],
        [ 1.98,  2.97,  2.21,  3.25,  2.42,  3.52,  2.68,  3.84],
    ]),
    5: np.array([
        [ 9.81,  9.81, 11.64, 11.64, 13.36, 13.36, 15.73, 15.73],
        [ 5.59,  6.26,  6.56,  7.30,  7.46,  8.27,  8.74,  9.63],
        [ 4.19,  5.06,  4.87,  5.85,  5.49,  6.59,  6.34,  7.52],
        [ 3.47,  4.45,  4.01,  5.07,  4.52,  5.62,  5.17,  6.36],
        [ 3.03,  4.06,  3.47,  4.57,  3.89,  5.07,  4.40,  5.72],
        [ 2.75,  3.79,  3.12,  4.25,  3.47,  4.67,  3.93,  5.23],
        [ 2.53,  3.59,  2.87,  4.00,  3.19,  4.38,  3.60,  4.90],
        [ 2.38,  3.45,  2.69,  3.83,  2.98,  4.16,  3.34,  4.63],
        [ 2.26,  3.34,  2.55,  3.68,  2.82,  4.02,  3.15,  4.43],
        [ 2.16,  3.24,  2.43,  3.56,  2.67,  3.87,  2.97,  4.24],
        [ 2.07,  3.16,  2.33,  3.46,  2.56,  3.76,  2.84,  4.10],
    ]),
}

_PSS_t = {
    # Cases 2,4 identical to 3,5 for t-stat (unaffected by restricted deterministics)
    1: np.array([
        [-1.62, -1.62, -1.95, -1.95, -2.24, -2.24, -2.58, -2.58],
        [-1.62, -2.28, -1.95, -2.60, -2.24, -2.90, -2.58, -3.22],
        [-1.62, -2.68, -1.95, -3.02, -2.24, -3.31, -2.58, -3.66],
        [-1.62, -3.00, -1.95, -3.33, -2.24, -3.64, -2.58, -3.97],
        [-1.62, -3.26, -1.95, -3.60, -2.24, -3.89, -2.58, -4.23],
        [-1.62, -3.49, -1.95, -3.83, -2.24, -4.12, -2.58, -4.44],
        [-1.62, -3.70, -1.95, -4.04, -2.24, -4.34, -2.58, -4.67],
        [-1.62, -3.90, -1.95, -4.23, -2.24, -4.54, -2.58, -4.88],
        [-1.62, -4.09, -1.95, -4.43, -2.24, -4.72, -2.58, -5.07],
        [-1.62, -4.26, -1.95, -4.61, -2.24, -4.89, -2.58, -5.25],
        [-1.62, -4.42, -1.95, -4.76, -2.24, -5.06, -2.58, -5.44],
    ]),
    3: np.array([
        [-2.57, -2.57, -2.86, -2.86, -3.13, -3.13, -3.43, -3.43],
        [-2.57, -2.91, -2.86, -3.22, -3.13, -3.50, -3.43, -3.82],
        [-2.57, -3.21, -2.86, -3.53, -3.13, -3.80, -3.43, -4.10],
        [-2.57, -3.46, -2.86, -3.78, -3.13, -4.05, -3.43, -4.37],
        [-2.57, -3.66, -2.86, -3.99, -3.13, -4.26, -3.43, -4.60],
        [-2.57, -3.86, -2.86, -4.19, -3.13, -4.46, -3.43, -4.79],
        [-2.57, -4.04, -2.86, -4.38, -3.13, -4.66, -3.43, -4.99],
        [-2.57, -4.23, -2.86, -4.57, -3.13, -4.85, -3.43, -5.19],
        [-2.57, -4.40, -2.86, -4.72, -3.13, -5.02, -3.43, -5.37],
        [-2.57, -4.56, -2.86, -4.88, -3.13, -5.18, -3.42, -5.54],
        [-2.57, -4.69, -2.86, -5.03, -3.13, -5.34, -3.43, -5.68],
    ]),
    5: np.array([
        [-3.13, -3.13, -3.41, -3.41, -3.65, -3.66, -3.96, -3.97],
        [-3.13, -3.40, -3.41, -3.69, -3.65, -3.96, -3.96, -4.26],
        [-3.13, -3.63, -3.41, -3.95, -3.65, -4.20, -3.96, -4.53],
        [-3.13, -3.84, -3.41, -4.16, -3.65, -4.42, -3.96, -4.73],
        [-3.13, -4.04, -3.41, -4.36, -3.65, -4.62, -3.96, -4.96],
        [-3.13, -4.21, -3.41, -4.52, -3.65, -4.79, -3.96, -5.13],
        [-3.13, -4.37, -3.41, -4.69, -3.65, -4.96, -3.96, -5.31],
        [-3.13, -4.53, -3.41, -4.85, -3.65, -5.14, -3.96, -5.49],
        [-3.13, -4.68, -3.41, -5.01, -3.65, -5.30, -3.96, -5.65],
        [-3.13, -4.82, -3.41, -5.15, -3.65, -5.44, -3.96, -5.79],
        [-3.13, -4.96, -3.41, -5.29, -3.65, -5.59, -3.96, -5.94],
    ]),
}
# Cases 2 -> same as 3 for t; Case 4 -> same as 5 for t
_PSS_t[2] = _PSS_t[3]
_PSS_t[4] = _PSS_t[5]

_PSS_COL_LABELS = [
    "I(0) 10%", "I(1) 10%",
    "I(0)  5%", "I(1)  5%",
    "I(0) 2.5%","I(1) 2.5%",
    "I(0)  1%", "I(1)  1%",
]
_K_LABELS = [f"k={i}" for i in range(11)]


def pss_cv_table(case: int = 3, stat: str = "F") -> pd.DataFrame:
    """
    Return the full PSS (2001) asymptotic critical value table as a DataFrame.

    Parameters
    ----------
    case : int in {1,2,3,4,5}
    stat : 'F' or 't'

    Returns
    -------
    pd.DataFrame shaped (11, 8) — rows k=0..10, cols I(0)/I(1) at 10/5/2.5/1%.
    """
    stat = stat.upper()
    if stat == "T":
        stat = "t"
    if stat not in ("F", "t"):
        raise ValueError("stat must be 'F' or 't'")
    if case not in (1, 2, 3, 4, 5):
        raise ValueError("case must be in {1,2,3,4,5}")

    src = _PSS_F if stat == "F" else _PSS_t
    arr = src[case]
    return pd.DataFrame(arr, index=_K_LABELS, columns=_PSS_COL_LABELS)


def bounds_cv_table(
    k: int,
    case: int = 3,
    stat: str = "F",
    sig_levels: list[float] | None = None,
) -> pd.DataFrame:
    """
    Return critical values for a given k and case.

    Parameters
    ----------
    k : int — number of long-run x-variables (0..10)
    case : int in {1,2,3,4,5}
    stat : 'F' or 't'
    sig_levels : list of significance levels (default [0.10, 0.05, 0.01])

    Returns
    -------
    pd.DataFrame shaped (1 + n_sig, 2) with I(0) and I(1) columns.
    """
    if sig_levels is None:
        sig_levels = [0.10, 0.05, 0.01]

    stat = stat.upper() if stat.upper() == "F" else "t"
    full = pss_cv_table(case=case, stat=stat)

    # map sig level to column pairs
    level_map = {0.10: 0, 0.05: 2, 0.025: 4, 0.01: 6}

    rows = {}
    for sl in sig_levels:
        col_idx = level_map.get(sl)
        if col_idx is None:
            raise ValueError(f"sig_level {sl} not available; choose from {list(level_map)}")
        row = full.iloc[k, col_idx:col_idx + 2].values
        rows[f"{int(sl*100)}%"] = {"I(0)": row[0], "I(1)": row[1]}

    return pd.DataFrame(rows).T


def _format_cv_table(k: int, case: int = 3) -> str:
    """
    Render a beautiful text table for the bounds test at all standard levels.
    Used by postestimation.bounds_test().
    """
    lines = []
    lines.append(" PSS (2001) Asymptotic Critical Values")
    lines.append(f" Case {case}  |  k = {k}")
    lines.append("-" * 50)
    lines.append(f"  {'Level':>6}  {'F: I(0)':>8}  {'F: I(1)':>8}  {'t: I(0)':>8}  {'t: I(1)':>8}")
    lines.append("-" * 50)

    F_full = pss_cv_table(case=case, stat="F")
    # t-stat: use appropriate case mapping
    t_case = case if case in _PSS_t else (3 if case == 2 else 5)
    T_full = pss_cv_table(case=t_case, stat="t")

    level_map = {0.10: (0, "10%"), 0.05: (2, " 5%"), 0.025: (4, "2.5%"), 0.01: (6, " 1%")}
    for sl, (col_idx, label) in level_map.items():
        F_i0 = F_full.iloc[k, col_idx]
        F_i1 = F_full.iloc[k, col_idx + 1]
        t_i0 = T_full.iloc[k, col_idx]
        t_i1 = T_full.iloc[k, col_idx + 1]
        lines.append(f"  {label:>6}  {F_i0:>8.3f}  {F_i1:>8.3f}  {t_i0:>8.3f}  {t_i1:>8.3f}")
    lines.append("-" * 50)
    return "\n".join(lines)


def get_decision(F_stat: float, t_stat: float, k: int,
                 case: int = 3, sig_level: float = 0.05) -> str:
    """
    Returns 'reject', 'no rejection', or 'inconclusive' for the bounds test.
    """
    level_map = {0.10: 0, 0.05: 2, 0.025: 4, 0.01: 6}
    col = level_map.get(sig_level, 2)

    F_full = pss_cv_table(case=case, stat="F")
    t_case = case if case in _PSS_t else (3 if case == 2 else 5)
    T_full = pss_cv_table(case=t_case, stat="t")

    F_ci0 = F_full.iloc[k, col]
    F_ci1 = F_full.iloc[k, col + 1]
    t_ci0 = T_full.iloc[k, col]
    t_ci1 = T_full.iloc[k, col + 1]

    # Reject: both exceed I(1) bound
    if F_stat > F_ci1 and t_stat < t_ci1:
        return "reject"
    # No rejection: either below I(0) bound
    if F_stat < F_ci0 or t_stat > t_ci0:
        return "no rejection"
    return "inconclusive"
