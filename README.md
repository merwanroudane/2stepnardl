# twostep-nardl

> **Two-step Nonlinear ARDL (NARDL) estimation in Python — PyPI package**

[![PyPI](https://img.shields.io/pypi/v/twostep-nardl)](https://pypi.org/project/twostep-nardl/)
[![Python](https://img.shields.io/pypi/pyversions/twostep-nardl)](https://pypi.org/project/twostep-nardl/)
[![License](https://img.shields.io/pypi/l/twostep-nardl)](LICENSE)

---

## Description

`twostep-nardl` is a production-ready Python library implementing the
**two-step Nonlinear ARDL** estimator of Cho, Greenwood-Nimmo and Shin (2019)
and the one-step OLS approach of Shin, Yu and Greenwood-Nimmo (2014).

It is a faithful Python port of the Stata package `twostep_nardl` v3.0.0
by Merwan Roudane, including all numerical routines (FM-OLS, FM-TOLS,
Newey-West HAC), post-estimation commands and PSS (2001) critical values.

### Features

- ✅ **Two-step estimation**: FM-OLS / FM-TOLS long-run + OLS ECM short-run
- ✅ **One-step estimation**: single OLS ARDL-ECM (SYG 2014)
- ✅ **Lag selection**: AIC/BIC grid search with partial or full automation
- ✅ **Partial sum decomposition** with threshold support
- ✅ **PSS (2001) bounds test** — complete critical value tables (Cases 1–5, k=0..10)
- ✅ **Wald tests** for LR, SR additive and SR impact asymmetry
- ✅ **Cumulative dynamic multipliers** with LR convergence verification
- ✅ **Persistence profile / half-life** analysis (Pesaran & Shin 1996)
- ✅ **Impulse response functions**
- ✅ **Residual diagnostics**: BG serial correlation, White het., JB normality, RESET
- ✅ **Publication-quality charts** via matplotlib
- ✅ **Pandas integration** — works directly on DataFrames

---

## Installation

```bash
pip install twostep-nardl
```

Or install from source:

```bash
git clone https://github.com/merwan-roudane/twostep-nardl.git
cd twostep-nardl
pip install -e ".[dev]"
```

---

## Quick Start

```python
import pandas as pd
import numpy as np
from twostep_nardl import TwoStepNARDL
from twostep_nardl.postestimation import (
    bounds_test, wald_test, multipliers, half_life
)
from twostep_nardl.plotting import plot_multipliers, plot_halflife

# Load your time-series data
df = pd.read_csv("your_data.csv", index_col=0)

# Estimate NARDL(1,1) with FM-OLS long-run step
model = TwoStepNARDL(
    data      = df,
    depvar    = "y",
    xvars     = ["x"],
    decompose = ["x"],       # decompose 'x' into positive & negative partial sums
    lags      = [1, 1],      # [p, q] — or None for automatic selection
    method    = "twostep",   # 'twostep' (FM-OLS) or 'onestep' (OLS)
    step1     = "fmols",     # 'fmols', 'ols'
    case      = 3,           # PSS deterministic case (intercept unrestricted)
)

results = model.fit()
print(results)               # publication-quality table

# Bounds test (PSS 2001)
bounds_test(results)

# Wald tests for asymmetry
wald_test(results)

# Dynamic multipliers (H = 40 periods)
mult = multipliers(results, horizon=40)
plot_multipliers(results, horizon=40)

# Half-life & persistence profile
half_life(results, horizon=40)
plot_halflife(results, horizon=40)
```

---

## API Reference

### `TwoStepNARDL`

```python
TwoStepNARDL(
    data,           # pd.DataFrame
    depvar,         # str — dependent variable name
    xvars,          # list[str] — all regressors
    decompose,      # list[str] — variables to decompose (must be in xvars)
    lags=None,      # int | list[int|None] | None — fixed lags (None = auto)
    maxlags=4,      # int | list[int] — max lag for selection
    ic='bic',       # 'bic' | 'aic'
    method='twostep',  # 'twostep' | 'onestep'
    step1='fmols',  # 'fmols' | 'ols' | 'fmtols' | 'tols'
    threshold=0.0,  # float | list[float] — dead-band threshold per variable
    bwidth=0,       # int — HAC bandwidth (0 = floor(T^0.25))
    case=3,         # int 1-5 — PSS deterministic case
    trendvar=None,  # str | None — trend variable name
    restricted=False,  # bool — restrict deterministics to LR (cases 2, 4)
    exog=None,      # list[str] | None — additional exogenous variables
    level=0.95,     # float — confidence level
)
```

### `NARDLResults` attributes

| Attribute | Description |
|---|---|
| `b_lr` | Long-run coefficients `[beta_pos, beta_neg]` for each decomposed variable |
| `V_lr` | Long-run VCE matrix |
| `b_sr` | Short-run ECM coefficients |
| `rho` | Speed of adjustment coefficient |
| `t_bdm` | BDM t-statistic for cointegration |
| `F_pss` | PSS F-statistic for cointegration |
| `r2`, `r2_a` | R² and adjusted R² (SR equation) |
| `W_lr`, `p_lr` | Wald test for long-run asymmetry |
| `W_sr`, `p_sr` | Wald test for short-run additive asymmetry |
| `W_impact`, `p_impact` | Wald test for short-run impact asymmetry |
| `ect_series` | Error correction term (pd.Series) |

### Post-estimation functions

```python
from twostep_nardl.postestimation import (
    bounds_test, wald_test, diagnostics, multipliers,
    half_life, asymadj, irf, ecm_table, predict
)
```

| Function | Description |
|---|---|
| `bounds_test(res)` | PSS (2001) bounds test with CV table |
| `wald_test(res)` | Long-run and short-run asymmetry tests |
| `diagnostics(res)` | BG serial correlation, White, JB, RESET |
| `multipliers(res, horizon)` | Cumulative dynamic multipliers |
| `half_life(res, horizon)` | Persistence profile and half-life |
| `asymadj(res)` | Asymmetric adjustment speed |
| `irf(res, horizon)` | Period-by-period impulse responses |
| `ecm_table(res)` | Publication-quality ECM parameter table |
| `predict(res, kind)` | Fitted values / residuals / ECT |

### Plotting

```python
from twostep_nardl.plotting import (
    plot_multipliers, plot_halflife, plot_irf, plot_cusum
)
```

### Critical Value Tables

```python
from twostep_nardl import pss_cv_table, bounds_cv_table

# Full PSS table for Case 3, F-statistic
df = pss_cv_table(case=3, stat="F")
print(df)

# Specific CVs for k=1 at 10%, 5%, 1%
cv = bounds_cv_table(k=1, case=3, stat="F", sig_levels=[0.10, 0.05, 0.01])
print(cv)
```

---

## PSS (2001) Critical Value Table (Case III, F-test)

| k  | I(0) 10% | I(1) 10% | I(0) 5% | I(1) 5% | I(0) 2.5% | I(1) 2.5% | I(0) 1% | I(1) 1% |
|----|----------|----------|---------|---------|-----------|-----------|---------|---------|
| 0  | 6.58     | 6.58     | 8.21    | 8.21    | 9.80      | 9.80      | 11.79   | 11.79   |
| 1  | 4.04     | 4.78     | 4.94    | 5.73    | 5.77      | 6.68      | 6.84    | 7.84    |
| 2  | 3.17     | 4.14     | 3.79    | 4.85    | 4.41      | 5.52      | 5.15    | 6.36    |
| 3  | 2.72     | 3.77     | 3.23    | 4.35    | 3.69      | 4.89      | 4.29    | 5.61    |
| 4  | 2.45     | 3.52     | 2.86    | 4.01    | 3.25      | 4.49      | 3.74    | 5.06    |
| 5  | 2.26     | 3.35     | 2.62    | 3.79    | 2.96      | 4.18      | 3.41    | 4.68    |
| …  | …        | …        | …       | …       | …         | …         | …       | …       |

Full tables for all 5 cases and k=0..10 are embedded in `critical_values.py`.

---

## References

- Cho, J.S., Greenwood-Nimmo, M. and Shin, Y. (2019). *Two-step estimation of
  the nonlinear autoregressive distributed lag model.*
- Shin, Y., Yu, B. and Greenwood-Nimmo, M. (2014). *Modelling asymmetric
  cointegration and dynamic multipliers in a nonlinear ARDL framework.*
  In: Festschrift in Honor of Peter Schmidt. Springer.
- Pesaran, M.H., Shin, Y. and Smith, R.J. (2001). *Bounds testing approaches
  to the analysis of level relationships.* Journal of Applied Econometrics.
- Kripfganz, S. and Schneider, D.C. (2020). *Response surface regressions for
  critical value bounds.* Oxford Bulletin of Economics and Statistics.

---

## License

MIT License. See [LICENSE](LICENSE).

## Author

Merwan Roudane — <merwanroudane920@gmail.com>
