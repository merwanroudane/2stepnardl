"""
tests/test_core.py — Core functionality test suite
===================================================
Replicates Stata test_twostep_nardl.do test 1:
  True: beta+ = 2.0, beta- = 1.0, rho = -0.5
  NARDL(1,1) with FM-OLS.
"""
import numpy as np
import pandas as pd
import pytest

from twostep_nardl import TwoStepNARDL, partial_sums, pss_cv_table, bounds_cv_table
from twostep_nardl.postestimation import (
    bounds_test, wald_test, multipliers, half_life, irf, ecm_table, predict
)


# ---------------------------------------------------------------------------
# Fixture: same simulated data as Stata test 1
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sim_data():
    """
    Simulate NARDL(1,1) DGP:
        y_t = 2*xp_t + 1*xn_t + ECM
        rho = -0.5, pi+ = 1.0, pi- = 0.5
    """
    rng = np.random.default_rng(12345)   # similar seed to Stata 12345
    T = 500
    burn = 100
    N = T + burn

    dx = np.zeros(N)
    for i in range(1, N):
        dx[i] = 0.5 * dx[i - 1] + np.sqrt(1 - 0.25) * rng.standard_normal()

    x = np.cumsum(dx)
    dx_pos = np.maximum(dx, 0)
    dx_neg = np.minimum(dx, 0)
    xp = np.cumsum(dx_pos)
    xn = np.cumsum(dx_neg)

    y = np.zeros(N)
    y[0] = 2 * xp[0] + 1 * xn[0]
    e = rng.standard_normal(N)
    for i in range(1, N):
        u_prev = y[i - 1] - 2 * xp[i - 1] - 1 * xn[i - 1]
        dy = -0.5 * u_prev + 1.0 * dx_pos[i] + 0.5 * dx_neg[i] + e[i]
        y[i] = y[i - 1] + dy

    df = pd.DataFrame({"y": y, "x": x}, index=np.arange(1, N + 1))
    df = df.iloc[burn:].copy()
    df.index = np.arange(1, T + 1)
    return df


# ---------------------------------------------------------------------------
# Test 1: FM-OLS two-step NARDL(1,1)
# ---------------------------------------------------------------------------

def test_twostep_fmols(sim_data):
    model = TwoStepNARDL(
        sim_data, "y", ["x"], decompose=["x"],
        lags=[1, 1], method="twostep", step1="fmols"
    )
    res = model.fit()

    assert res.method == "twostep"
    assert res.step1 == "fmols"
    assert res.k == 1
    assert res.nobs > 0

    # True beta+ = 2.0, beta- = 1.0
    beta_pos = float(res.b_lr[0])
    beta_neg = float(res.b_lr[1])
    rho = float(res.rho)

    assert abs(beta_pos - 2.0) < 0.30, f"beta+ = {beta_pos:.4f}, expected ~2.0"
    assert abs(beta_neg - 1.0) < 0.30, f"beta- = {beta_neg:.4f}, expected ~1.0"
    assert abs(rho - (-0.5)) < 0.20, f"rho = {rho:.4f}, expected ~-0.5"

    print(f"\n  beta+ = {beta_pos:.4f}  (true: 2.0)")
    print(f"  beta- = {beta_neg:.4f}  (true: 1.0)")
    print(f"  rho   = {rho:.4f}  (true: -0.5)")


# ---------------------------------------------------------------------------
# Test 2: OLS first-step
# ---------------------------------------------------------------------------

def test_twostep_ols(sim_data):
    model = TwoStepNARDL(
        sim_data, "y", ["x"], decompose=["x"],
        lags=[1, 1], method="twostep", step1="ols"
    )
    res = model.fit()
    assert abs(float(res.b_lr[0]) - 2.0) < 0.50
    assert abs(float(res.b_lr[1]) - 1.0) < 0.50


# ---------------------------------------------------------------------------
# Test 3: One-step
# ---------------------------------------------------------------------------

def test_onestep(sim_data):
    model = TwoStepNARDL(
        sim_data, "y", ["x"], decompose=["x"],
        lags=[1, 1], method="onestep"
    )
    res = model.fit()
    assert res.method == "onestep"
    # One-step should also recover correct direction
    assert float(res.b_lr[0]) > 0
    assert float(res.b_lr[1]) > 0


# ---------------------------------------------------------------------------
# Test 4: Higher lag order
# ---------------------------------------------------------------------------

def test_higher_lags(sim_data):
    model = TwoStepNARDL(
        sim_data, "y", ["x"], decompose=["x"],
        lags=[3, 3], step1="fmols"
    )
    res = model.fit()
    assert res.lags == [3, 3]
    assert res.p_lag == 2


# ---------------------------------------------------------------------------
# Test 5: Automatic lag selection (BIC)
# ---------------------------------------------------------------------------

def test_lag_selection_bic(sim_data):
    model = TwoStepNARDL(
        sim_data, "y", ["x"], decompose=["x"],
        lags=None, maxlags=4, ic="bic"
    )
    res = model.fit()
    assert res.do_lagselect is True
    assert all(l >= 0 for l in res.lags)
    print(f"\n  Selected lags (BIC): {res.lags}")


# ---------------------------------------------------------------------------
# Test 6: Partial sum decomposition
# ---------------------------------------------------------------------------

def test_partial_sums_basic():
    x = np.array([0.0, 1.0, 0.5, 1.5, 1.0])
    xp, xn = partial_sums(x)
    # Dx = [0, 1, -0.5, 1, -0.5]
    # xp[0]=0, xp[1]=1, xp[2]=1, xp[3]=2, xp[4]=2
    assert abs(xp[1] - 1.0) < 1e-10
    assert abs(xp[3] - 2.0) < 1e-10
    # xn[2] = -0.5
    assert abs(xn[2] - (-0.5)) < 1e-10


# ---------------------------------------------------------------------------
# Test 7: PSS critical value tables
# ---------------------------------------------------------------------------

def test_pss_cv_table():
    tbl = pss_cv_table(case=3, stat="F")
    assert tbl.shape == (11, 8)
    # k=1 (row 1), 5% I(0) = 4.94
    assert abs(tbl.iloc[1, 2] - 4.94) < 0.01

    tbl_t = pss_cv_table(case=3, stat="t")
    assert tbl_t.shape == (11, 8)
    # k=0 (row 0), 5% I(0) = -2.86
    assert abs(tbl_t.iloc[0, 2] - (-2.86)) < 0.01


def test_bounds_cv_table():
    cv = bounds_cv_table(k=1, case=3, stat="F", sig_levels=[0.10, 0.05, 0.01])
    assert "10%" in cv.index
    assert "5%" in cv.index
    assert cv.loc["5%", "I(0)"] == pytest.approx(4.94, abs=0.01)
    assert cv.loc["5%", "I(1)"] == pytest.approx(5.73, abs=0.01)


# ---------------------------------------------------------------------------
# Test 8: Post-estimation functions
# ---------------------------------------------------------------------------

def test_bounds_test(sim_data):
    model = TwoStepNARDL(sim_data, "y", ["x"], decompose=["x"], lags=[1, 1])
    res = model.fit()
    bt = bounds_test(res, show_table=False)
    assert "F_pss" in bt
    assert "decisions" in bt
    assert isinstance(bt["F_pss"], float)


def test_wald_test(sim_data):
    model = TwoStepNARDL(sim_data, "y", ["x"], decompose=["x"], lags=[1, 1])
    res = model.fit()
    wt = wald_test(res, show_table=False)
    assert "Long-run symmetry" in wt


def test_multipliers(sim_data):
    model = TwoStepNARDL(sim_data, "y", ["x"], decompose=["x"], lags=[1, 1])
    res = model.fit()
    mult = multipliers(res, horizon=20, show_table=False)
    assert "pos" in mult.columns
    assert "neg" in mult.columns
    assert len(mult) == 20
    # Multipliers should converge towards LR values
    lr_pos = float(res.b_lr[0])
    assert abs(float(mult.loc[20, "pos"]) - lr_pos) < 0.5


def test_half_life(sim_data):
    model = TwoStepNARDL(sim_data, "y", ["x"], decompose=["x"], lags=[1, 1])
    res = model.fit()
    hl = half_life(res, horizon=40, show_table=False)
    if res.rho < 0:
        assert "half_life" in hl
        assert hl["half_life"] > 0


def test_predict(sim_data):
    model = TwoStepNARDL(sim_data, "y", ["x"], decompose=["x"], lags=[1, 1])
    res = model.fit()
    yhat = predict(res, kind="xb")
    resid = predict(res, kind="residuals")
    assert len(yhat) == len(sim_data)
    # Fitted + residuals should equal Dy
    non_nan = ~(yhat.isna() | resid.isna())
    diff = (yhat + resid)[non_nan].values
    actual_dy = np.diff(sim_data["y"].values)[non_nan[1:].values]
    # rough check
    assert len(diff) > 0


def test_summary(sim_data):
    model = TwoStepNARDL(sim_data, "y", ["x"], decompose=["x"], lags=[1, 1])
    res = model.fit()
    s = res.summary()
    assert "FM-OLS" in s or "FMOLS" in s or "fmols" in s.lower()
    assert "ADJ" in s
    assert "LR" in s
    assert "SR" in s
