"""
postestimation.py — Post-estimation analysis for TwoStepNARDL results
=====================================================================
Implements all estat sub-commands from twostep_nardl_estat.ado:
  bounds_test, wald_test, diagnostics, multipliers, half_life,
  asymadj, irf, ecm_table, predict
"""
from __future__ import annotations

import textwrap
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .model import NARDLResults, _wald_sr
from .critical_values import _format_cv_table, get_decision, pss_cv_table


# ---------------------------------------------------------------------------
# bounds_test  (estat ectest)
# ---------------------------------------------------------------------------

def bounds_test(
    res: NARDLResults,
    sig_levels: list[float] = (0.10, 0.05, 0.01),
    show_table: bool = True,
) -> dict:
    """
    PSS (2001) bounds test for cointegration.

    Parameters
    ----------
    res : NARDLResults
    sig_levels : sequence of float
    show_table : bool — print the output table

    Returns
    -------
    dict with keys: F_pss, t_bdm, rho, case, k, decisions
    """
    F = float(res.F_pss)
    t = float(res.t_bdm)
    k = res.k
    case = res.case
    N = res.nobs

    lines = []
    lines.append("\n" + "=" * 78)
    lines.append("  Pesaran, Shin and Smith (2001) Bounds Test for Cointegration")
    lines.append("=" * 78)
    lines.append(f"  H0: no level relationship")
    lines.append(f"\n  F-statistic (PSS)   = {F:>12.4f}")
    lines.append(f"  t-statistic (BDM)   = {t:>12.4f}")
    lines.append(f"  Speed of adj. (rho) = {res.rho:>12.4f}")
    lines.append(f"  Case                = {case:>12}")
    lines.append(f"  k (asymm. vars)     = {k:>12}")
    lines.append(f"  Obs                 = {N:>12,}")
    lines.append("")

    # Print CV table
    cv_text = _format_cv_table(k=k, case=case)
    lines.append(cv_text)
    lines.append("")

    # Decision matrix
    decisions = {}
    lines.append(f"  {'Level':>6}  {'F decision':^15}  {'t decision':^15}  {'Overall':^15}")
    lines.append("  " + "-" * 57)
    for sl in sig_levels:
        f_dec = get_decision(F, t, k, case, sl)
        decisions[sl] = f_dec
        pct = f"{int(sl*100)}%"
        lines.append(f"  {pct:>6}  {f_dec:^15}  {f_dec:^15}  {f_dec:^15}")
    lines.append("=" * 78)
    lines.append("  Reject H0 if both F and t exceed I(1) bounds.")
    lines.append("  Cannot reject if either statistic is below I(0) bounds.")

    if show_table:
        print("\n".join(lines))

    return {
        "F_pss": F, "t_bdm": t, "rho": res.rho,
        "case": case, "k": k,
        "decisions": decisions,
    }


# ---------------------------------------------------------------------------
# wald_test  (estat waldtest)
# ---------------------------------------------------------------------------

def wald_test(res: NARDLResults, show_table: bool = True) -> dict:
    """Wald tests for LR and SR asymmetry."""
    lines = []
    lines.append("\n" + "=" * 78)
    lines.append("  Asymmetry Tests")
    lines.append("=" * 78)
    lines.append(f"  {'Hypothesis':<35}{'Wald':>10}  {'p-value':>10}  {'':>8}")
    lines.append("-" * 78)

    results = {}
    for label, W, p in [
        ("Long-run symmetry", res.W_lr, res.p_lr),
        ("Short-run symmetry (additive)", res.W_sr, res.p_sr),
        ("Short-run symmetry (impact)", res.W_impact, res.p_impact),
    ]:
        st = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        lines.append(f"  {label:<35}{W:>10.4f}  {p:>10.4f}  {st:>8}")
        results[label] = {"W": W, "p": p}

    lines.append("=" * 78)
    lines.append("  *** p<0.01, ** p<0.05, * p<0.10")

    if show_table:
        print("\n".join(lines))
    return results


# ---------------------------------------------------------------------------
# diagnostics  (estat diagnostics)
# ---------------------------------------------------------------------------

def diagnostics(res: NARDLResults, show_table: bool = True) -> dict:
    """
    Residual diagnostics: Breusch-Godfrey, White, normality, RESET.
    Uses statsmodels for the tests.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import (
            acorr_breusch_godfrey, het_white, linear_reset
        )
        from statsmodels.stats.stattools import jarque_bera
    except ImportError:
        raise RuntimeError("statsmodels is required for diagnostics()")

    y = res._sr_endog
    X = res._sr_X
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ b
    T, k = X.shape

    out = {}
    lines = []
    lines.append("\n" + "=" * 78)
    lines.append("  Residual Diagnostics")
    lines.append("=" * 78)
    lines.append(f"  {'Test':<35}{'Statistic':>12}  {'p-value':>10}  {'':>6}")
    lines.append("-" * 78)

    # Fit OLS model via statsmodels for structured diagnostics
    sm_model = sm.OLS(y, X).fit()

    # 1. Breusch-Godfrey serial correlation
    try:
        bg_lags = max(1, res.p_lag + 1)
        bg_result = acorr_breusch_godfrey(sm_model, nlags=bg_lags)
        bg_chi2, bg_p = bg_result[0], bg_result[1]
        st = "***" if bg_p < 0.01 else "**" if bg_p < 0.05 else "*" if bg_p < 0.10 else ""
        lines.append(f"  {'Serial correlation (BG)':<35}{bg_chi2:>12.4f}  {bg_p:>10.4f}  {st:>6}")
        out["bg_chi2"], out["bg_p"] = float(bg_chi2), float(bg_p)
    except Exception as e:
        lines.append(f"  {'Serial correlation (BG)':<35}{'(unavailable)':>24}")

    # 2. White heteroskedasticity
    try:
        wh_result = het_white(sm_model.resid, sm_model.model.exog)
        wh_chi2, wh_p = wh_result[0], wh_result[1]
        st = "***" if wh_p < 0.01 else "**" if wh_p < 0.05 else "*" if wh_p < 0.10 else ""
        lines.append(f"  {'Heteroskedasticity (White)':<35}{wh_chi2:>12.4f}  {wh_p:>10.4f}  {st:>6}")
        out["white_chi2"], out["white_p"] = float(wh_chi2), float(wh_p)
    except Exception:
        lines.append(f"  {'Heteroskedasticity (White)':<35}{'(unavailable)':>24}")

    # 3. Normality (Jarque-Bera)
    try:
        jb_stat, jb_p, _, _ = jarque_bera(resid)
        st = "***" if jb_p < 0.01 else "**" if jb_p < 0.05 else "*" if jb_p < 0.10 else ""
        lines.append(f"  {'Normality (Jarque-Bera)':<35}{jb_stat:>12.4f}  {jb_p:>10.4f}  {st:>6}")
        out["jb_stat"], out["jb_p"] = float(jb_stat), float(jb_p)
    except Exception:
        lines.append(f"  {'Normality (Jarque-Bera)':<35}{'(unavailable)':>24}")

    # 4. RESET (Ramsey)
    try:
        reset_result = linear_reset(sm_model, power=2, use_f=True)
        rs_F, rs_p = reset_result.statistic, reset_result.pvalue
        st = "***" if rs_p < 0.01 else "**" if rs_p < 0.05 else "*" if rs_p < 0.10 else ""
        lines.append(f"  {'Functional form (RESET)':<35}{rs_F:>12.4f}  {rs_p:>10.4f}  {st:>6}")
        out["reset_F"], out["reset_p"] = float(rs_F), float(rs_p)
    except Exception:
        lines.append(f"  {'Functional form (RESET)':<35}{'(unavailable)':>24}")

    lines.append("=" * 78)
    lines.append("  *** p<0.01, ** p<0.05, * p<0.10")
    lines.append("  Rejection indicates violation of the regression assumption.")

    if show_table:
        print("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# multipliers  (estat multiplier)
# ---------------------------------------------------------------------------

def multipliers(
    res: NARDLResults,
    horizon: int = 40,
    show_table: bool = True,
) -> pd.DataFrame:
    """
    Compute cumulative dynamic multipliers.

    Returns a pd.DataFrame with columns ['pos', 'neg', 'diff'] and
    horizon rows (1..H).
    """
    rho = res.rho
    p_lag = res.p_lag
    q_lag = res.q_lag
    k = res.k
    b_sr = res.b_sr

    # Extract phi (lagged Dy coefficients) from b_sr
    phi = np.zeros(p_lag)
    for j in range(p_lag):
        if 1 + j < len(b_sr):
            phi[j] = float(b_sr[1 + j])

    # Extract pi_pos, pi_neg: shape (q_lag, k)
    pi_pos = np.zeros((q_lag, k))
    pi_neg = np.zeros((q_lag, k))
    idx_start = 1 + p_lag  # after rho + phi
    for j in range(q_lag):
        for i in range(k):
            col = idx_start + j * k + i
            if col < len(b_sr):
                pi_pos[j, i] = float(b_sr[col])
    idx_start2 = idx_start + q_lag * k
    for j in range(q_lag):
        for i in range(k):
            col = idx_start2 + j * k + i
            if col < len(b_sr):
                pi_neg[j, i] = float(b_sr[col])

    beta_pos = res.b_lr[:k]
    beta_neg = res.b_lr[k:]

    dy_pos = np.zeros((horizon, k))
    dy_neg = np.zeros((horizon, k))

    for h in range(horizon):
        if h < q_lag:
            dy_pos[h, :] += pi_pos[h, :]
            dy_neg[h, :] += pi_neg[h, :]

        for j in range(min(p_lag, h)):
            dy_pos[h, :] += phi[j] * dy_pos[h - j - 1, :]
            dy_neg[h, :] += phi[j] * dy_neg[h - j - 1, :]

        if h >= 1:
            cum_pos_prev = dy_pos[:h, :].sum(axis=0)
            cum_neg_prev = dy_neg[:h, :].sum(axis=0)
            dy_pos[h, :] += rho * (cum_pos_prev - beta_pos)
            dy_neg[h, :] += rho * (cum_neg_prev - beta_neg)

    mult_pos = np.cumsum(dy_pos, axis=0)
    mult_neg = np.cumsum(dy_neg, axis=0)

    # For k=1 return simple DataFrame
    if k == 1:
        df = pd.DataFrame({
            "pos": mult_pos[:, 0],
            "neg": mult_neg[:, 0],
            "diff": mult_pos[:, 0] - mult_neg[:, 0],
        }, index=range(1, horizon + 1))
    else:
        cols = {}
        for i in range(k):
            xv = res.asymvars[i]
            cols[f"{xv}_pos"] = mult_pos[:, i]
            cols[f"{xv}_neg"] = mult_neg[:, i]
            cols[f"{xv}_diff"] = mult_pos[:, i] - mult_neg[:, i]
        df = pd.DataFrame(cols, index=range(1, horizon + 1))

    if show_table:
        lines = []
        lines.append("\n" + "=" * 78)
        lines.append("  Cumulative Dynamic Multipliers")
        lines.append("=" * 78)
        lines.append(f"  {'Horizon':>7}  {'Positive':>12}  {'Negative':>12}  {'Difference':>12}  {'Asymmetry'}")
        lines.append("-" * 78)
        show_h = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40]
        for h in show_h:
            if h > horizon:
                continue
            if k == 1:
                mp = df.loc[h, "pos"]
                mn = df.loc[h, "neg"]
                md = df.loc[h, "diff"]
                asym = "***" if abs(md) > 0.1 * max(abs(mp), abs(mn)) else "**" if abs(md) > 0.05 * max(abs(mp), abs(mn)) else ""
                lines.append(f"  {h:>7}  {mp:>12.4f}  {mn:>12.4f}  {md:>12.4f}  {asym}")
        if k == 1:
            lr_p = float(res.b_lr[0])
            lr_n = float(res.b_lr[1])
            lines.append("-" * 78)
            lines.append(f"  {'LR eq.':>7}  {lr_p:>12.4f}  {lr_n:>12.4f}  {lr_p - lr_n:>12.4f}")
        lines.append("=" * 78)
        print("\n".join(lines))

    return df


# ---------------------------------------------------------------------------
# half_life  (estat halflife)
# ---------------------------------------------------------------------------

def half_life(
    res: NARDLResults,
    horizon: int = 40,
    show_table: bool = True,
) -> dict:
    """Half-life and persistence profile of the ECM."""
    rho = float(res.rho)
    p_lag = res.p_lag

    # SE of rho from combined VCE
    rho_se = float(np.sqrt(res.V_combined[0, 0]))
    rho_t = rho / rho_se if rho_se > 0 else np.nan
    rho_p = 2 * stats.t.sf(abs(rho_t), df=res.df_r)

    lines = []
    lines.append("\n" + "=" * 78)
    lines.append("  Half-Life & Persistence Analysis")
    lines.append("=" * 78)
    st = "***" if rho_p < 0.01 else "**" if rho_p < 0.05 else "*" if rho_p < 0.10 else ""
    lines.append(f"  ECM coefficient (rho)    = {rho:>12.6f}  ({st})")
    lines.append(f"  Std. Error               = {rho_se:>12.6f}")
    lines.append(f"  t-statistic              = {rho_t:>12.4f}")
    lines.append("")

    result = {"rho": rho, "rho_se": rho_se, "rho_t": rho_t, "rho_p": rho_p}

    if rho >= 0:
        lines.append("  WARNING: rho >= 0. No error correction (non-convergent ECM).")
        lines.append("-" * 78)
    else:
        hl = -np.log(2) / np.log(1 + rho)
        mal = -(1 + rho) / rho
        full_adj = np.log(0.01) / np.log(1 + rho)
        lines.append(f"  Half-Life (50%)          = {hl:>12.2f} periods")
        lines.append(f"  Mean Adjustment Lag      = {mal:>12.2f} periods")
        lines.append(f"  99% Adjustment Time      = {full_adj:>12.2f} periods")
        lines.append("")
        result.update({"half_life": hl, "mal": mal, "full_adj": full_adj})

    # Persistence profile
    phi = np.zeros(p_lag)
    for j in range(p_lag):
        if 1 + j < len(res.b_sr):
            phi[j] = float(res.b_sr[1 + j])

    nlevels = p_lag + 1
    a = np.zeros(nlevels)
    if p_lag == 0:
        a[0] = 1 + rho
    else:
        a[0] = 1 + rho + (phi[0] if p_lag > 0 else 0)
        for j in range(1, p_lag):
            a[j] = phi[j] - phi[j - 1]
        a[p_lag] = -phi[p_lag - 1]

    pp = np.zeros(horizon + 1)
    pp[0] = 1.0
    for h in range(1, horizon + 1):
        for j in range(min(h, nlevels)):
            pp[h] += a[j] * pp[h - j - 1]

    pp_hl = horizon
    for h in range(1, horizon + 1):
        if abs(pp[h]) < 0.5:
            pp_hl = h
            break

    lines.append("  Persistence Profile (Pesaran & Shin, 1996)")
    lines.append("  " + "-" * 55)
    lines.append(f"  {'Horizon':>6}  {'PP(h)':>12}  {'% Remaining':>12}")
    lines.append("  " + "-" * 55)
    for h in range(0, horizon + 1):
        if h <= 10 or h % 5 == 0 or h == horizon or h == pp_hl:
            marker = "  <-- Half-life" if h == pp_hl else ""
            lines.append(f"  {h:>6}  {pp[h]:>12.6f}  {pp[h]*100:>11.2f}%{marker}")
    lines.append("  " + "-" * 55)
    lines.append(f"  PP Half-Life = {pp_hl} periods")
    lines.append("=" * 78)

    result["pp_halflife"] = pp_hl
    result["pp_series"] = pd.Series(pp, index=range(horizon + 1), name="PP(h)")

    if show_table:
        print("\n".join(lines))
    return result


# ---------------------------------------------------------------------------
# asymadj  (estat asymadj)
# ---------------------------------------------------------------------------

def asymadj(
    res: NARDLResults,
    horizon: int = 40,
    show_table: bool = True,
) -> dict:
    """Asymmetric adjustment speed analysis."""
    rho = float(res.rho)
    b_lr = res.b_lr
    k = res.k

    lines = []
    lines.append("\n" + "=" * 78)
    lines.append("  Asymmetric Adjustment Speed")
    lines.append("=" * 78)

    out = {}
    for i in range(k):
        xv = res.asymvars[i]
        bp = float(b_lr[i])
        bn = float(b_lr[k + i])

        hl_pos = -np.log(2) / np.log(1 + rho) if rho < 0 else np.inf
        hl_neg = hl_pos  # same ECM coefficient; asymmetry is in pi

        lines.append(f"\n  Variable: {xv}")
        lines.append(f"    LR beta+ = {bp:>10.4f}   LR beta- = {bn:>10.4f}")
        lines.append(f"    Half-life = {hl_pos:>10.2f} periods (shared ECM)")
        direction = "increases" if abs(bp) > abs(bn) else "decreases"
        lines.append(f"    Long-run impact larger for: {direction}")
        out[xv] = {"beta_pos": bp, "beta_neg": bn, "half_life": hl_pos}

    lines.append("\n" + "=" * 78)
    if show_table:
        print("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# irf  (estat irf)
# ---------------------------------------------------------------------------

def irf(
    res: NARDLResults,
    horizon: int = 20,
    show_table: bool = True,
) -> pd.DataFrame:
    """Impulse response functions."""
    mult = multipliers(res, horizon=horizon, show_table=False)

    lines = []
    lines.append("\n" + "=" * 78)
    lines.append("  Impulse Response Functions")
    lines.append("=" * 78)
    lines.append(f"  {'Horizon':>7}  {'Pos (period)':>14}  {'Neg (period)':>14}  {'Pos (cum)':>12}  {'Neg (cum)':>12}")
    lines.append("-" * 78)

    if res.k == 1:
        pos_cum = mult["pos"].values
        neg_cum = mult["neg"].values
        pos_period = np.diff(np.concatenate([[0], pos_cum]))
        neg_period = np.diff(np.concatenate([[0], neg_cum]))
        for h in range(1, horizon + 1):
            lines.append(f"  {h:>7}  {pos_period[h-1]:>14.4f}  {neg_period[h-1]:>14.4f}  {pos_cum[h-1]:>12.4f}  {neg_cum[h-1]:>12.4f}")

    lines.append("=" * 78)
    if show_table:
        print("\n".join(lines))

    if res.k == 1:
        return pd.DataFrame({
            "pos_period": pos_period, "neg_period": neg_period,
            "pos_cum": pos_cum, "neg_cum": neg_cum,
        }, index=range(1, horizon + 1))
    return mult


# ---------------------------------------------------------------------------
# ecm_table  (estat ecmtable)  — publication-quality table matching CGS2019
# ---------------------------------------------------------------------------

def ecm_table(res: NARDLResults) -> None:
    """
    Display a publication-quality ECM parameter table in the style of
    Table 9 in Cho, Greenwood-Nimmo & Shin (2019).
    """
    from scipy import stats as st

    b_comb = res.b_combined
    se_comb = np.sqrt(np.diag(res.V_combined))
    t_comb = b_comb / se_comb
    p_comb = 2 * st.t.sf(np.abs(t_comb), df=res.df_r)

    def _row(label, b, se, t, p, indent=4):
        star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        return f"  {'':<{indent}}{label:<28} {b:>10.4f}  {se:>10.4f}  {t:>9.3f}  {p:>8.4f} {star}"

    lines = []
    lines.append("\n" + "=" * 78)
    lines.append("  Two-Step NARDL: ECM Parameter Estimates (CGS 2019, Table 9 Style)")
    lines.append("=" * 78)

    # Panel A: Long-run
    lines.append("\n  Panel A: Long-Run Estimates")
    lines.append("-" * 78)
    lines.append(f"  {'Variable':<32} {'Coef':>10}  {'Std.Err':>10}  {'t-stat':>9}  {'P>|t|':>8}")
    lines.append("-" * 78)

    i = 0
    for eq, nm in zip(res.eq_labels, res.combined_names):
        if eq in ("LR",):
            lines.append(_row(nm, b_comb[i], se_comb[i], t_comb[i], p_comb[i]))
        i += 1

    lines.append("")
    lines.append(f"  Panel B: Short-Run Dynamic Estimates")
    lines.append("-" * 78)
    lines.append(f"  {'Variable':<32} {'Coef':>10}  {'Std.Err':>10}  {'t-stat':>9}  {'P>|t|':>8}")
    lines.append("-" * 78)

    i = 0
    for eq, nm in zip(res.eq_labels, res.combined_names):
        if eq in ("ADJ", "SR"):
            lines.append(_row(nm, b_comb[i], se_comb[i], t_comb[i], p_comb[i]))
        i += 1

    lines.append("-" * 78)
    lines.append(f"  Adj. R-sq = {res.r2_a:.4f}   RMSE = {res.rmse:.4f}   "
                 f"F = {res.F_stat:.2f}   N = {res.nobs:,}")
    lines.append("=" * 78)
    lines.append("  *** p<0.01, ** p<0.05, * p<0.10")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def predict(res: NARDLResults, kind: str = "xb") -> pd.Series:
    """
    Generate predictions after estimation.

    Parameters
    ----------
    res : NARDLResults
    kind : 'xb' (fitted values), 'residuals', or 'ecterm'

    Returns
    -------
    pd.Series aligned to res.index
    """
    if kind == "ecterm":
        return res.ect_series.rename("ect")

    X = res._sr_X
    y = res._sr_endog
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ b

    T_total = len(res.index)
    T_est = len(yhat)
    start = T_total - T_est

    result = pd.Series(np.nan, index=res.index, name=kind)
    result.iloc[start:] = yhat if kind == "xb" else (y - yhat)
    return result
