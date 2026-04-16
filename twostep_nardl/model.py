"""
model.py — TwoStepNARDL main estimation class
=============================================
Implements the two-step (FM-OLS + OLS) estimator of
Cho, Greenwood-Nimmo and Shin (2019) and the one-step OLS
approach of Shin, Yu and Greenwood-Nimmo (2014).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .decompose import partial_sums
from .estimators import fmols, fmtols, FMOLSResult, FMTOLSResult
from .lag_selection import select_lags


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class NARDLResults:
    """
    Container for all twostep_nardl estimation results.
    Mirrors e() scalars/matrices from Stata.
    """
    # ---- inputs ----
    depvar: str
    xvars: list[str]
    asymvars: list[str]
    linvars: list[str]
    method: str              # 'twostep' | 'onestep'
    step1: str               # 'fmols' | 'ols' | 'fmtols' | 'tols'
    case: int
    lags: list[int]          # [p, q1, ..., qk, r1, ..., rm]
    threshold: list[float]

    # ---- long-run ----
    b_lr: np.ndarray         # shape (2*k,)
    V_lr: np.ndarray         # shape (2*k, 2*k)
    b_lin: np.ndarray        # shape (n_lin,)
    V_lin: np.ndarray        # shape (n_lin, n_lin)
    lr_names: list[str]      # ['x1_pos', 'x1_neg', ...]
    lin_names: list[str]

    # ---- short-run ECM ----
    b_sr: np.ndarray         # shape (n_sr,)
    V_sr: np.ndarray         # shape (n_sr, n_sr)
    V_sr_hc: np.ndarray      # HC-robust VCE for SR
    sr_names: list[str]

    # ---- combined ----
    b_combined: np.ndarray
    V_combined: np.ndarray
    combined_names: list[str]
    eq_labels: list[str]     # 'ADJ', 'LR', 'SR' for each coef

    # ---- fit statistics ----
    nobs: int
    df_m: int
    df_r: int
    r2: float
    r2_a: float
    rmse: float
    ll: float
    rss: float
    F_stat: float

    # ---- ECM ----
    rho: float               # speed of adjustment
    t_bdm: float             # BDM t-statistic
    F_pss: float             # PSS F-statistic
    tau2: float              # long-run variance (step 1)

    # ---- Wald tests ----
    W_lr: float
    p_lr: float
    W_sr: float
    p_sr: float
    W_impact: float
    p_impact: float

    # ---- lag info ----
    do_lagselect: bool
    ic_type: str
    ic_opt: float
    numcombs: int
    p_lag: int
    q_lag: int
    k: int
    k_lin: int

    # ---- stored series ----
    ect_series: pd.Series        # the ECT (full index)
    pos_series: dict             # {varname: pd.Series of x_pos}
    neg_series: dict             # {varname: pd.Series of x_neg}
    index: pd.Index
    endog_name: str
    data_aligned: pd.DataFrame   # aligned dataset used in estimation

    # ---- internal: regression inputs for diagnostics ----
    _sr_endog: np.ndarray = field(repr=False)   # D.y
    _sr_X: np.ndarray     = field(repr=False)   # SR design matrix
    _sr_touse: np.ndarray = field(repr=False)   # bool mask

    # ----------------------------------------------------------------
    def summary(self) -> str:
        """Print a publication-style summary table."""
        return _format_summary(self)

    def __str__(self) -> str:
        return self.summary()

    # convenience
    @property
    def coef(self) -> pd.Series:
        return pd.Series(self.b_combined, index=self.combined_names)

    @property
    def stderr(self) -> pd.Series:
        return pd.Series(np.sqrt(np.diag(self.V_combined)), index=self.combined_names)

    @property
    def tvalues(self) -> pd.Series:
        return self.coef / self.stderr

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series(
            2 * stats.t.sf(np.abs(self.tvalues.values), df=self.df_r),
            index=self.combined_names,
        )


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def _stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _format_summary(res: NARDLResults) -> str:
    lines = []
    w = 78
    lines.append("=" * w)
    lag_str = ",".join(str(l) for l in res.lags)
    title = f"  Nonlinear ARDL({lag_str}) Estimation"
    lines.append(f"{title:<50}{'Obs':>10} = {res.nobs:>8,}")
    lines.append("=" * w)
    method_lbl = f"One-step OLS" if res.method == "onestep" else f"Long-run: {res.step1.upper()}"
    lines.append(f"  Method  : {method_lbl:<38}{'R-squared':>10} = {res.r2:>8.4f}")
    lines.append(f"{'':37}{'Adj R-sq':>13} = {res.r2_a:>8.4f}")
    lines.append(f"{'':37}{'F-stat':>13} = {res.F_stat:>8.2f}")
    lines.append(f"{'':37}{'RMSE':>13} = {res.rmse:>8.4f}")
    lines.append("=" * w)
    lines.append("")

    # Coefficient table
    header = f"  {'Variable':<24} {'Coef':>10} {'Std.Err':>10} {'t-stat':>9} {'P>|t|':>8}  {'[95% CI]'}"
    lines.append(header)
    lines.append("-" * w)

    se = np.sqrt(np.diag(res.V_combined))
    t_vals = res.b_combined / se
    p_vals = 2 * stats.t.sf(np.abs(t_vals), df=res.df_r)
    ci_lo = res.b_combined - 1.96 * se
    ci_hi = res.b_combined + 1.96 * se

    prev_eq = ""
    for i, (nm, eq) in enumerate(zip(res.combined_names, res.eq_labels)):
        if eq != prev_eq:
            lines.append(f"  [{eq}]")
            prev_eq = eq
        b, s, t, p = res.b_combined[i], se[i], t_vals[i], p_vals[i]
        st = _stars(p)
        ci = f"[{ci_lo[i]:>8.4f}, {ci_hi[i]:>8.4f}]"
        lines.append(f"  {nm:<24} {b:>10.4f} {s:>10.4f} {t:>9.3f} {p:>8.4f}  {ci} {st}")

    lines.append("=" * w)

    # Wald tests
    lines.append(f"  {'Asymmetry Tests':<30}{'Wald':>10}  {'p-value':>10}  {'Decision'}")
    lines.append("-" * w)
    for label, W, p in [
        ("Long-run", res.W_lr, res.p_lr),
        ("Short-run (additive)", res.W_sr, res.p_sr),
        ("Short-run (impact)", res.W_impact, res.p_impact),
    ]:
        dec = "Reject" if p < 0.10 else "  --"
        st = _stars(p)
        lines.append(f"  {label:<30}{W:>10.3f}  {p:>10.4f}  {dec} {st}")
    lines.append("=" * w)

    # ECM
    lines.append(f"  Cointegration   t-stat = {res.t_bdm:>8.3f}    Speed of adj. = {res.rho:>7.4f}")
    lines.append("-" * w)
    lines.append("  *** p<0.01, ** p<0.05, * p<0.10")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TwoStepNARDL:
    """
    Two-step Nonlinear ARDL estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Time-series dataset. Must have a DatetimeIndex or integer index.
    depvar : str
        Name of the dependent variable.
    xvars : list of str
        Names of independent variables (must include decompose variables).
    decompose : list of str
        Subset of xvars to decompose via partial sums.
    lags : list of int or None, optional
        Fixed lag structure. Length must match [depvar] + xvars. Use None
        for auto-selection. E.g. [2, 2, 2] fixes all lags at 2.
        A single int applies to all. Default None (auto-select all).
    maxlags : int or list of int, optional
        Maximum lag for grid search. Default 4.
    ic : str, optional
        Information criterion: 'bic' (default) or 'aic'.
    method : str, optional
        'twostep' (default, FM-OLS step 1) or 'onestep' (single OLS).
    step1 : str, optional
        First-step estimator for two-step: 'fmols' (default for k=1),
        'ols', 'tols', 'fmtols' (default for k>1).
    threshold : float or list of float, optional
        Decomposition threshold (default 0).
    bwidth : int, optional
        HAC bandwidth (default 0 = floor(T^0.25)).
    case : int, optional
        PSS deterministic case 1-5 (default 3 = unrestricted intercept).
    trendvar : str, optional
        Name of trend variable (required for cases 4+5).
    restricted : bool, optional
        Restrict deterministics to LR (cases 2 and 4). Default False.
    exog : list of str, optional
        Additional exogenous variables.
    level : float, optional
        Confidence level (default 0.95).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from twostep_nardl import TwoStepNARDL
    >>> # Assume df has columns 'y' and 'x' with a time index
    >>> model = TwoStepNARDL(df, 'y', ['x'], decompose=['x'], lags=[1, 1])
    >>> results = model.fit()
    >>> print(results)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        depvar: str,
        xvars: list[str],
        decompose: list[str],
        lags: Optional[Union[int, list[Optional[int]]]] = None,
        maxlags: Union[int, list[int]] = 4,
        ic: str = "bic",
        method: str = "twostep",
        step1: str = "fmols",
        threshold: Union[float, list[float]] = 0.0,
        bwidth: int = 0,
        case: int = 3,
        trendvar: Optional[str] = None,
        restricted: bool = False,
        exog: Optional[list[str]] = None,
        level: float = 0.95,
    ):
        self.data = data.copy()
        self.depvar = depvar
        self.xvars = list(xvars)
        self.decompose = list(decompose)
        self.linvars = [v for v in self.xvars if v not in self.decompose]

        # Validate decompose
        bad = set(self.decompose) - set(self.xvars)
        if bad:
            raise ValueError(f"decompose variables {bad} are not in xvars")

        # Lags
        n_all = 1 + len(self.xvars)   # depvar + all x vars
        if lags is None:
            self._lags = [None] * n_all
            self._do_lagselect = True
        elif isinstance(lags, int):
            self._lags = [lags] * n_all
            self._do_lagselect = False
        else:
            if len(lags) == 1:
                self._lags = list(lags) * n_all
            elif len(lags) != n_all:
                raise ValueError(f"lags must have length 1 or {n_all}")
            else:
                self._lags = list(lags)
            self._do_lagselect = any(l is None for l in self._lags)

        # Maxlags
        if isinstance(maxlags, int):
            self._maxlags = [maxlags] * n_all
        else:
            self._maxlags = list(maxlags) if len(maxlags) == n_all else list(maxlags) * n_all

        # Dep var must have at least 1 lag
        if self._lags[0] is not None and self._lags[0] == 0:
            raise ValueError("Dependent variable must have at least 1 lag")
        if self._maxlags[0] == 0:
            self._maxlags[0] = 1

        self.ic = ic.lower()
        self.method = method.lower()
        self.step1 = step1.lower()
        self.bwidth = bwidth
        self.case = case
        self.trendvar = trendvar
        self.restricted = restricted
        self.exog = exog or []
        self.level = level

        # Threshold per decomposed variable
        n_asym = len(self.decompose)
        if isinstance(threshold, (int, float)):
            self.threshold = [float(threshold)] * n_asym
        else:
            self.threshold = [float(t) for t in threshold]
        if len(self.threshold) == 1 and n_asym > 1:
            self.threshold = self.threshold * n_asym

        # Adjust step1 for k>1
        k_asym = len(self.decompose)
        if k_asym > 1:
            if self.step1 == "fmols":
                self.step1 = "fmtols"
            elif self.step1 == "ols":
                self.step1 = "tols"

    # ----------------------------------------------------------------
    def fit(self) -> NARDLResults:
        """
        Estimate the model and return a NARDLResults object.
        """
        df = self.data.dropna(subset=[self.depvar] + self.xvars + self.exog).copy()
        T_full = len(df)
        y_full = df[self.depvar].values.astype(float)
        idx = df.index
        k_asym = len(self.decompose)
        k_lin = len(self.linvars)

        # ---- Partial sum decomposition ----
        pos_series: dict[str, pd.Series] = {}
        neg_series: dict[str, pd.Series] = {}
        xpos_list, xneg_list, xorig_list = [], [], []

        asym_idx = 0
        for xv in self.decompose:
            thr = self.threshold[asym_idx]
            xp, xn = partial_sums(df[xv].values, threshold=thr)
            nm_p = f"_{xv}_pos_nardl"
            nm_n = f"_{xv}_neg_nardl"
            df[nm_p] = xp
            df[nm_n] = xn
            pos_series[xv] = pd.Series(xp, index=idx, name=nm_p)
            neg_series[xv] = pd.Series(xn, index=idx, name=nm_n)
            xpos_list.append(xp)
            xneg_list.append(xn)
            xorig_list.append(df[xv].values.astype(float))
            asym_idx += 1

        lin_arrays = [df[v].values.astype(float) for v in self.linvars]
        exog_arrays = [df[v].values.astype(float) for v in self.exog]
        trend_array = df[self.trendvar].values.astype(float) if self.trendvar else None

        # ---- Build estimation arrays ----
        xpos_arr = np.column_stack(xpos_list) if k_asym > 0 else np.empty((T_full, 0))
        xneg_arr = np.column_stack(xneg_list) if k_asym > 0 else np.empty((T_full, 0))
        x_arr = np.column_stack(xorig_list) if k_asym > 0 else np.empty((T_full, 0))
        zlin_arr = np.column_stack(lin_arrays) if k_lin > 0 else np.empty((T_full, 0))

        # ---- Step 1: Long-run estimation (two-step) ----
        tau2 = 0.0
        if self.method == "twostep":
            if k_asym == 1:
                use_fmols = self.step1 in ("fmols",)
                step1_res = fmols(
                    y_full, xpos_arr[:, 0], x_arr[:, 0], xneg_arr[:, 0],
                    zlin=zlin_arr if k_lin > 0 else None,
                    use_fmols=use_fmols, bw=self.bwidth,
                )
                b_lr = step1_res.beta_lr       # shape (2,)
                V_lr = step1_res.V_lr
                b_lin = step1_res.beta_lin
                V_lin = step1_res.V_lin
                ect_full = step1_res.ect
                tau2 = step1_res.tau2
            else:
                use_fm = self.step1 in ("fmtols",)
                step1_res = fmtols(
                    y_full, xpos_arr, x_arr, xneg_arr,
                    zlin=zlin_arr if k_lin > 0 else None,
                    use_fm=use_fm, bw=self.bwidth,
                )
                b_lr = np.concatenate([step1_res.beta_pos, step1_res.beta_neg])  # (2k,)
                V_lr = step1_res.V_lr
                b_lin = step1_res.beta_lin
                V_lin = step1_res.V_lin
                ect_full = step1_res.ect
                tau2 = step1_res.tau2

            # Store ECT
            df["_ect_nardl"] = ect_full

        # ---- Determine lags ----
        # Determine maximum lag used for sample alignment
        lags = self._lags.copy()
        maxlags = self._maxlags.copy()
        if lags[0] is None and not self._do_lagselect:
            lags[0] = 1

        # Use max possible lags for sample trimming
        max_lag = max(
            (l if l is not None else m) for l, m in zip(lags, maxlags)
        )
        # Trim sample for lags
        start_idx = max_lag
        T = T_full - start_idx

        # Sliced arrays
        y = y_full[start_idx:]
        Dy = np.diff(y_full)[start_idx - 1:] if start_idx > 0 else np.diff(y_full)

        if self.method == "twostep":
            ect = ect_full[start_idx - 1:T_full - 1]  # L.ect
        # else: will be built after one-step OLS

        # ---- Build SR design matrix ----
        def _build_sr_matrix(
            lags_p: int,
            lags_q: list[int],
            lags_r: list[int],
            is_onestep: bool = False,
        ) -> tuple[np.ndarray, list[str]]:
            """
            Build SR regressor matrix for the ECM.
            Returns (X, column_names).
            """
            cols, names = [], []

            if is_onestep:
                # One-step: L.y, L.x_pos, L.x_neg, [L.z], lags of diffs
                L_y = y_full[start_idx - 1:T_full - 1]
                cols.append(L_y.reshape(-1, 1)); names.append(f"L.{self.depvar}")
                for i, xv in enumerate(self.decompose):
                    pv = xpos_list[i][start_idx - 1:T_full - 1]
                    nv = xneg_list[i][start_idx - 1:T_full - 1]
                    cols.append(pv.reshape(-1, 1)); names.append(f"L.{xv}_pos")
                    cols.append(nv.reshape(-1, 1)); names.append(f"L.{xv}_neg")
                for zv, zar in zip(self.linvars, lin_arrays):
                    L_z = zar[start_idx - 1:T_full - 1]
                    cols.append(L_z.reshape(-1, 1)); names.append(f"L.{zv}")
            else:
                # Two-step: L.ect
                cols.append(ect.reshape(-1, 1)); names.append("L.ect")

            # Lagged Dy
            for j in range(1, lags_p):
                lag_arr = np.zeros(T)
                lag_arr[:] = y_full[start_idx - j:T_full - j] - y_full[start_idx - j - 1:T_full - j - 1]
                cols.append(lag_arr.reshape(-1, 1))
                names.append(f"LD{j}.{self.depvar}")

            # Dx_pos, Dx_neg lags
            for i, xv in enumerate(self.decompose):
                q = max(lags_q[i], 1)
                dx_pos = np.diff(xpos_list[i])  # length T_full-1
                dx_neg = np.diff(xneg_list[i])
                for j in range(q):
                    p_lag = dx_pos[start_idx - 1 - j:T_full - 1 - j]
                    n_lag = dx_neg[start_idx - 1 - j:T_full - 1 - j]
                    nm = f"D.{xv}_pos" if j == 0 else f"LD{j}.{xv}_pos"
                    cols.append(p_lag.reshape(-1, 1)); names.append(nm)
                    nm = f"D.{xv}_neg" if j == 0 else f"LD{j}.{xv}_neg"
                    cols.append(n_lag.reshape(-1, 1)); names.append(nm)

            # Dz lags
            for zv, zar in zip(self.linvars, lin_arrays):
                zi = self.xvars.index(zv)
                r = max(lags_r[zi - len(self.decompose)] if lags_r else 1, 1)
                dz = np.diff(zar)
                for j in range(r):
                    z_lag = dz[start_idx - 1 - j:T_full - 1 - j]
                    nm = f"D.{zv}" if j == 0 else f"LD{j}.{zv}"
                    cols.append(z_lag.reshape(-1, 1)); names.append(nm)

            # Exog
            for ev, ear in zip(self.exog, exog_arrays):
                cols.append(ear[start_idx:].reshape(-1, 1)); names.append(ev)

            # Trend (non-restricted)
            if self.trendvar and not self.restricted:
                cols.append(trend_array[start_idx:].reshape(-1, 1))
                names.append(self.trendvar)

            # Constant
            if self.case != 1:
                cols.append(np.ones((T, 1))); names.append("_cons")

            return np.concatenate(cols, axis=1), names

        # ---- Lag selection ----
        final_lags = lags.copy()
        ic_opt = np.nan
        numcombs = 1

        if self._do_lagselect and self.method == "twostep":
            # Use a simplified grid search directly on SR
            p_range = [lags[0]] if lags[0] is not None else list(range(1, maxlags[0] + 1))
            q_ranges = []
            for i in range(len(self.xvars)):
                li = lags[1 + i]
                q_ranges.append([li] if li is not None else list(range(0, maxlags[1 + i] + 1)))

            numcombs = len(p_range)
            for qr in q_ranges:
                numcombs *= len(qr)

            best_ic_val = np.inf
            best_p = p_range[0]
            best_qs = [qr[0] for qr in q_ranges]

            import itertools
            for combo in itertools.product(p_range, *q_ranges):
                p_try = combo[0]
                qs_try = list(combo[1:])
                qs_asym = qs_try[:k_asym]
                qs_lin = qs_try[k_asym:]

                try:
                    X_try, _ = _build_sr_matrix(p_try, qs_asym, qs_lin)
                    ic_val = _ic_from_ols(Dy, X_try, self.ic)
                except Exception:
                    continue
                if ic_val < best_ic_val:
                    best_ic_val = ic_val
                    best_p = p_try
                    best_qs = qs_try

            final_lags = [best_p] + best_qs
            ic_opt = best_ic_val

        elif self._do_lagselect and self.method == "onestep":
            # Same but for one-step
            p_range = [lags[0]] if lags[0] is not None else list(range(1, maxlags[0] + 1))
            q_ranges = []
            for i in range(len(self.xvars)):
                li = lags[1 + i]
                q_ranges.append([li] if li is not None else list(range(0, maxlags[1 + i] + 1)))

            numcombs = len(p_range)
            for qr in q_ranges:
                numcombs *= len(qr)

            import itertools
            best_ic_val = np.inf
            best_p, best_qs = p_range[0], [qr[0] for qr in q_ranges]

            for combo in itertools.product(p_range, *q_ranges):
                p_try = combo[0]
                qs_try = list(combo[1:])
                qs_asym = qs_try[:k_asym]
                qs_lin = qs_try[k_asym:]
                try:
                    X_try, _ = _build_sr_matrix(p_try, qs_asym, qs_lin, is_onestep=True)
                    ic_val = _ic_from_ols(Dy, X_try, self.ic)
                except Exception:
                    continue
                if ic_val < best_ic_val:
                    best_ic_val = ic_val
                    best_p, best_qs = p_try, qs_try
            final_lags = [best_p] + best_qs
            ic_opt = best_ic_val
        else:
            final_lags = [l if l is not None else 1 for l in lags]

        p_lag = final_lags[0] - 1   # lags of Dy in SR (dep ARDL order - 1)
        q_lag_max = max(final_lags[1:1 + k_asym]) if k_asym > 0 else 0
        q_lag = max(q_lag_max, 1)

        qs_asym = final_lags[1:1 + k_asym]
        qs_lin = final_lags[1 + k_asym:] if k_lin > 0 else []

        # ---- One-step: build and run OLS ----
        if self.method == "onestep":
            X_os, names_os = _build_sr_matrix(final_lags[0], qs_asym, qs_lin, is_onestep=True)
            b_sr_raw, V_sr_raw, V_sr_hc_raw, fit_stats = _ols_fit(Dy, X_os)

            # For one-step, extract rho = b[L.y]
            rho_hat = float(b_sr_raw[0])
            t_bdm = rho_hat / np.sqrt(float(V_sr_raw[0, 0]))

            # PSS F-stat: joint test on level vars
            n_levels = 1 + 2 * k_asym + k_lin
            F_pss = _wald_F(b_sr_raw[:n_levels], V_sr_raw[:n_levels, :n_levels], df_r=fit_stats[2])

            # Derive LR coefficients by delta method: beta = -theta/rho
            b_lr, V_lr, b_lin, V_lin = _onestep_lr(
                b_sr_raw, V_sr_raw, k_asym, k_lin, self.decompose, self.linvars
            )

            # Generate ECT
            alpha_hat = b_sr_raw[0]  # rho in L.y
            # ECT not directly available; approximated from SR
            ect_full = np.full(T_full, np.nan)
            ect = np.full(T, np.nan)

            b_sr = b_sr_raw
            V_sr = V_sr_raw
            V_sr_hc = V_sr_hc_raw

            # Build compatible SR vector (drop level terms)
            n_skip = 1 + 2 * k_asym + k_lin
            b_sr_compat = np.concatenate([[b_sr_raw[0]], b_sr_raw[n_skip:]])
            names_compat = [names_os[0]] + names_os[n_skip:]
            # Build matching VCE for b_sr_compat
            compat_idx = [0] + list(range(n_skip, len(b_sr_raw)))
            V_sr_compat = V_sr_hc_raw[np.ix_(compat_idx, compat_idx)]

        else:  # twostep
            X_sr, names_sr = _build_sr_matrix(final_lags[0], qs_asym, qs_lin)
            b_sr, V_sr, V_sr_hc, fit_stats = _ols_fit(Dy, X_sr)

            rho_hat = float(b_sr[0])
            se_rho = np.sqrt(float(V_sr[0, 0]))
            t_bdm = rho_hat / se_rho
            F_pss = t_bdm ** 2

            b_sr_compat = b_sr
            names_compat = names_sr

        # ---- Wald tests ----
        W_lr, p_lr = _wald_lr(b_lr, V_lr, k_asym)
        _V_sr_for_wald = V_sr_hc if self.method == "twostep" else V_sr_compat
        try:
            W_sr, p_sr = _wald_sr(b_sr_compat, _V_sr_for_wald,
                                   p_lag, q_lag, k_asym, len(qs_lin), "additive")
        except Exception:
            W_sr, p_sr = np.nan, np.nan
        try:
            W_impact, p_impact = _wald_sr(b_sr_compat, _V_sr_for_wald,
                                           p_lag, q_lag, k_asym, len(qs_lin), "impact")
        except Exception:
            W_impact, p_impact = np.nan, np.nan

        # ---- Assemble combined coefficient vector ----
        lr_names = []
        for xv in self.decompose:
            lr_names.extend([f"{xv}_pos", f"{xv}_neg"])
        lin_names = list(self.linvars)

        sr_start = 1 if self.method == "twostep" else 1 + 2 * k_asym + k_lin
        b_sr2 = b_sr_compat[1:]   # drop ECT
        names_sr2 = names_compat[1:]

        if k_lin > 0:
            b_combined = np.concatenate([[rho_hat], b_lr, b_lin, b_sr2])
            eq_labels = (["ADJ"] + ["LR"] * (2 * k_asym + k_lin) + ["SR"] * len(b_sr2))
            combined_names = ["L.ect"] + lr_names + lin_names + names_sr2
        else:
            b_combined = np.concatenate([[rho_hat], b_lr, b_sr2])
            eq_labels = (["ADJ"] + ["LR"] * (2 * k_asym) + ["SR"] * len(b_sr2))
            combined_names = ["L.ect"] + lr_names + names_sr2

        # Build combined VCE (block-diagonal: ADJ/SR share one block, LR separate)
        n_adj = 1
        n_lr = len(b_lr)
        n_lin_lr = k_lin
        n_sr2 = len(b_sr2)
        n_total = n_adj + n_lr + n_lin_lr + n_sr2

        V_combined = np.zeros((n_total, n_total))
        V_combined[0, 0] = float(V_sr[0, 0])
        V_combined[1:1 + n_lr, 1:1 + n_lr] = V_lr
        if n_lin_lr > 0:
            s = 1 + n_lr
            V_combined[s:s + n_lin_lr, s:s + n_lin_lr] = V_lin
        s2 = 1 + n_lr + n_lin_lr
        if n_sr2 > 0:
            sr_offset = 1  # in V_sr
            V_combined[s2:s2 + n_sr2, s2:s2 + n_sr2] = V_sr[sr_offset:sr_offset + n_sr2, sr_offset:sr_offset + n_sr2]
            V_combined[0, s2:s2 + n_sr2] = V_sr[0, sr_offset:sr_offset + n_sr2]
            V_combined[s2:s2 + n_sr2, 0] = V_sr[sr_offset:sr_offset + n_sr2, 0]

        # ---- ECT series ----
        ect_series = pd.Series(ect_full, index=idx, name="_ect_nardl") if self.method == "twostep" else pd.Series(np.nan, index=idx, name="_ect_nardl")

        return NARDLResults(
            depvar=self.depvar,
            xvars=self.xvars,
            asymvars=self.decompose,
            linvars=self.linvars,
            method=self.method,
            step1=self.step1,
            case=self.case,
            lags=final_lags,
            threshold=self.threshold,
            b_lr=b_lr,
            V_lr=V_lr,
            b_lin=b_lin if k_lin > 0 else np.array([]),
            V_lin=V_lin if k_lin > 0 else np.empty((0, 0)),
            lr_names=lr_names,
            lin_names=lin_names,
            b_sr=b_sr,
            V_sr=V_sr,
            V_sr_hc=V_sr_hc,
            sr_names=names_compat,
            b_combined=b_combined,
            V_combined=V_combined,
            combined_names=combined_names,
            eq_labels=eq_labels,
            nobs=int(fit_stats[0]),
            df_m=int(fit_stats[1]),
            df_r=int(fit_stats[2]),
            r2=float(fit_stats[3]),
            r2_a=float(fit_stats[4]),
            rmse=float(fit_stats[5]),
            ll=float(fit_stats[6]),
            rss=float(fit_stats[7]),
            F_stat=float(fit_stats[8]),
            rho=rho_hat,
            t_bdm=float(t_bdm),
            F_pss=float(F_pss),
            tau2=float(tau2),
            W_lr=float(W_lr),
            p_lr=float(p_lr),
            W_sr=float(W_sr),
            p_sr=float(p_sr),
            W_impact=float(W_impact),
            p_impact=float(p_impact),
            do_lagselect=self._do_lagselect,
            ic_type=self.ic,
            ic_opt=float(ic_opt) if not np.isnan(ic_opt) else np.nan,
            numcombs=int(numcombs),
            p_lag=p_lag,
            q_lag=q_lag,
            k=k_asym,
            k_lin=k_lin,
            ect_series=ect_series,
            pos_series=pos_series,
            neg_series=neg_series,
            index=idx,
            endog_name=self.depvar,
            data_aligned=df,
            _sr_endog=Dy,
            _sr_X=X_sr if self.method == "twostep" else X_os,
            _sr_touse=np.ones(T, dtype=bool),
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _ic_from_ols(y: np.ndarray, X: np.ndarray, ic: str) -> float:
    T, k = X.shape
    try:
        b = np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        return np.inf
    resid = y - X @ b
    rss = float(resid @ resid)
    sigma2 = rss / T
    if sigma2 <= 0:
        return np.inf
    ll = -0.5 * T * (np.log(2 * np.pi * sigma2) + 1)
    return -2 * ll + (2 * k if ic == "aic" else k * np.log(T))


def _ols_fit(y: np.ndarray, X: np.ndarray):
    """OLS + HC-robust VCE. Returns (b, V, V_hc, fit_stats)."""
    T, k = X.shape
    try:
        b = np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        raise RuntimeError("OLS failed (singular matrix)")
    resid = y - X @ b
    XtX = X.T @ X
    XtXinv = np.linalg.pinv(XtX)

    df_m = k - 1  # -1 for constant
    df_r = T - k
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    mss = tss - rss
    r2 = 1 - rss / tss if tss > 0 else 0.0
    r2_a = 1 - (rss / df_r) / (tss / (T - 1)) if T > 1 else 0.0
    rmse = np.sqrt(rss / df_r) if df_r > 0 else np.nan
    sigma2 = rss / T
    ll = -0.5 * T * (np.log(2 * np.pi * sigma2) + 1)
    F_stat = (mss / df_m) / (rss / df_r) if df_m > 0 and df_r > 0 else np.nan

    V = (rss / df_r) * XtXinv

    # HC3 sandwich VCE
    h = np.diag(X @ XtXinv @ X.T)
    e_hc = resid / (1 - h + 1e-12)
    meat = (X * e_hc.reshape(-1, 1)).T @ (X * e_hc.reshape(-1, 1))
    V_hc = XtXinv @ meat @ XtXinv

    fit_stats = (T, df_m, df_r, r2, r2_a, rmse, ll, rss, F_stat)
    return b, V, V_hc, fit_stats


def _wald_lr(b_lr: np.ndarray, V_lr: np.ndarray, k: int):
    """Wald test H0: beta_pos == beta_neg for all k variables."""
    if k == 1:
        # H0: b_lr[0] - b_lr[1] = 0
        diff = b_lr[0] - b_lr[1]
        R = np.array([[1.0, -1.0]])
        var = float(np.squeeze(R @ V_lr @ R.T))
        W = diff ** 2 / var if var > 0 else 0.0
        p = 1 - stats.chi2.cdf(W, df=1)
        return float(W), float(p)
    else:
        # H0: beta_pos_i = beta_neg_i for i=1..k
        R = np.hstack([np.eye(k), -np.eye(k)])
        diff = R @ b_lr
        meat = R @ V_lr @ R.T
        try:
            W = float(diff @ np.linalg.solve(meat, diff))
        except Exception:
            W = np.nan
        p = 1 - stats.chi2.cdf(W, df=k) if not np.isnan(W) else np.nan
        return float(W), float(p)


def _wald_sr(b_sr: np.ndarray, V_sr: np.ndarray,
             p_lag: int, q_lag: int, k: int, n_lin_sr: int,
             test_type: str):
    """
    SR Wald test.
    b_sr layout: [ECT, LD1.y,..., D.xpos1,...,LD(q-1).xpos_k,
                   D.xneg1,..., lin_sr..., _cons]
    """
    n_params = len(b_sr)
    idx_pi_pos_start = 1 + p_lag
    idx_pi_neg_start = idx_pi_pos_start + q_lag * k

    if test_type == "additive":
        R = np.zeros((k, n_params))
        for i in range(k):
            for j in range(q_lag):
                pos_col = idx_pi_pos_start + j * k + i
                neg_col = idx_pi_neg_start + j * k + i
                if pos_col < n_params:
                    R[i, pos_col] = 1.0
                if neg_col < n_params:
                    R[i, neg_col] = -1.0
    else:  # impact
        R = np.zeros((k, n_params))
        for i in range(k):
            pos_col = idx_pi_pos_start + i
            neg_col = idx_pi_neg_start + i
            if pos_col < n_params:
                R[i, pos_col] = 1.0
            if neg_col < n_params:
                R[i, neg_col] = -1.0

    diff = R @ b_sr
    meat = R @ V_sr @ R.T
    try:
        W = float(diff @ np.linalg.solve(meat, diff))
    except Exception:
        W = np.nan
    df = k
    p = 1 - stats.chi2.cdf(W, df=df) if not np.isnan(W) else np.nan
    return float(W), float(p)


def _wald_F(b: np.ndarray, V: np.ndarray, df_r: int) -> float:
    """F-statistic for joint significance."""
    k = len(b)
    try:
        W = float(b @ np.linalg.solve(V, b))
    except Exception:
        return np.nan
    return W / k


def _onestep_lr(b_sr_raw, V_sr_raw, k_asym, k_lin, decompose, linvars):
    """Extract LR coefficients from one-step OLS via delta method."""
    rho = float(b_sr_raw[0])
    # Positions in b_sr_raw: [rho, L.xpos1, L.xneg1, ..., L.z1, ...]
    b_lr = np.zeros(2 * k_asym)
    for i in range(k_asym):
        theta_pos = float(b_sr_raw[1 + 2 * i])
        theta_neg = float(b_sr_raw[2 + 2 * i])
        b_lr[i] = -theta_pos / rho if rho != 0 else np.nan
        b_lr[k_asym + i] = -theta_neg / rho if rho != 0 else np.nan

    # Delta method VCE (simplified)
    n = len(b_sr_raw)
    J = np.zeros((2 * k_asym, n))
    for i in range(k_asym):
        theta_pos = float(b_sr_raw[1 + 2 * i])
        theta_neg = float(b_sr_raw[2 + 2 * i])
        J[i, 0] = theta_pos / rho ** 2
        J[i, 1 + 2 * i] = -1.0 / rho
        J[k_asym + i, 0] = theta_neg / rho ** 2
        J[k_asym + i, 2 + 2 * i] = -1.0 / rho
    V_lr = J @ V_sr_raw @ J.T

    if k_lin > 0:
        offset = 1 + 2 * k_asym
        b_lin = -b_sr_raw[offset:offset + k_lin] / rho
        J_lin = np.zeros((k_lin, n))
        for i in range(k_lin):
            th = float(b_sr_raw[offset + i])
            J_lin[i, 0] = th / rho ** 2
            J_lin[i, offset + i] = -1.0 / rho
        V_lin = J_lin @ V_sr_raw @ J_lin.T
    else:
        b_lin = np.array([])
        V_lin = np.empty((0, 0))

    return b_lr, V_lr, b_lin, V_lin
