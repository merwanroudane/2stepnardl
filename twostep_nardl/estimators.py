"""
estimators.py — FM-OLS and FM-TOLS long-run estimators
=======================================================
Translates _2snardl_neweywest(), _2snardl_fmols(), _2snardl_fmtols()
from _2snardl_mata.mata.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Newey-West HAC estimator
# ---------------------------------------------------------------------------

def newey_west_hac(G: np.ndarray, bw: int = 0):
    """
    Newey-West HAC estimator.

    Parameters
    ----------
    G : np.ndarray, shape (T, m)
        Residual/innovation matrix (demeaned internally).
    bw : int
        Bandwidth. If <= 0, uses floor(T^0.25).

    Returns
    -------
    Sigma : np.ndarray, shape (m, m)  — long-run covariance
    Pi    : np.ndarray, shape (m, m)  — one-sided long-run covariance
    """
    T, m = G.shape
    ell = int(np.floor(T ** 0.25)) if bw <= 0 else int(bw)

    G = G - G.mean(axis=0)

    Gamma0 = G.T @ G / T
    Sigma = Gamma0.copy()
    Pi = Gamma0.copy()

    for kk in range(1, ell + 1):
        Gamma_k = G[:T - kk, :].T @ G[kk:, :] / T
        omega_k = 1.0 - kk / (1.0 + ell)
        Sigma += omega_k * (Gamma_k + Gamma_k.T)
        Pi += Gamma_k.T

    return Sigma, Pi


# ---------------------------------------------------------------------------
# Dataclasses for results
# ---------------------------------------------------------------------------

@dataclass
class FMOLSResult:
    """Result from FM-OLS (k=1 asymmetric variable)."""
    beta_lr: np.ndarray    # shape (2,) — [beta_pos, beta_neg]
    V_lr: np.ndarray       # shape (2,2)
    beta_lin: np.ndarray   # shape (n_lin,)
    V_lin: np.ndarray      # shape (n_lin, n_lin)
    V_lr_full: np.ndarray  # shape (2+n_lin, 2+n_lin)
    tau2: float
    alpha: float           # intercept
    ect: np.ndarray        # shape (T,)
    uhat: np.ndarray       # shape (T,)
    method: str            # 'fmols' | 'ols'
    n_lin: int


@dataclass
class FMTOLSResult:
    """Result from FM-TOLS (k>1 asymmetric variables)."""
    beta_pos: np.ndarray   # shape (k,)
    beta_neg: np.ndarray   # shape (k,)
    V_lr: np.ndarray       # shape (2k, 2k)
    beta_lin: np.ndarray   # shape (n_lin,)
    V_lin: np.ndarray      # shape (n_lin, n_lin)
    V_lr_full: np.ndarray  # shape (2k+n_lin, 2k+n_lin)
    tau2: float
    ect: np.ndarray        # shape (T,)
    uhat: np.ndarray       # shape (T,)
    method: str            # 'fmtols' | 'tols'
    n_lin: int


# ---------------------------------------------------------------------------
# FM-OLS  (k = 1 asymmetric variable)
# ---------------------------------------------------------------------------

def fmols(
    y: np.ndarray,
    xpos: np.ndarray,
    x: np.ndarray,
    xneg: np.ndarray,
    zlin: np.ndarray | None = None,
    use_fmols: bool = True,
    bw: int = 0,
) -> FMOLSResult:
    """
    Fully-Modified OLS (or plain OLS) for k=1 decomposed variable.

    Model: y = alpha + lambda*x_pos + eta*x_neg + gamma'*z + u
    Long-run: beta_pos = lambda + eta, beta_neg = eta

    Parameters
    ----------
    y, xpos, x, xneg : np.ndarray, shape (T,)
    zlin : np.ndarray, shape (T, n_lin) or None
    use_fmols : bool — True for FM-OLS, False for plain OLS
    bw : int — HAC bandwidth
    """
    T = len(y)
    n_lin = zlin.shape[1] if (zlin is not None and zlin.ndim == 2) else (
        len(zlin[0]) if (zlin is not None and zlin.ndim > 0 and len(zlin) > 0) else 0
    )
    if zlin is None or (isinstance(zlin, np.ndarray) and zlin.size == 0):
        zlin_arr = np.empty((T, 0))
        n_lin = 0
    else:
        zlin_arr = np.atleast_2d(zlin).T if zlin.ndim == 1 else zlin
        n_lin = zlin_arr.shape[1]

    # Build Q = [1, x_pos, x, z1, ..., zm]
    Q = np.column_stack([np.ones(T), xpos, x] + (
        [zlin_arr] if n_lin > 0 else []))

    QQ = Q.T @ Q
    QQinv = np.linalg.pinv(QQ)
    rho_hat = QQinv @ (Q.T @ y)
    uhat = y - Q @ rho_hat

    # First differences
    dx = np.zeros(T)
    dx[1:] = np.diff(x)

    if n_lin > 0:
        dz = np.zeros((T, n_lin))
        dz[1:, :] = np.diff(zlin_arr, axis=0)
    else:
        dz = np.empty((T, 0))

    if not use_fmols:
        tau2 = float(np.var(uhat, ddof=1))
        V_rho = tau2 * QQinv
        method = "ols"
    else:
        # G = [dx, dz, uhat]
        n_nonstat = 1 + n_lin
        G_cols = [dx.reshape(-1, 1)]
        if n_lin > 0:
            G_cols.append(dz)
        G_cols.append(uhat.reshape(-1, 1))
        G = np.concatenate(G_cols, axis=1)

        Sigma_hat, Pi_hat = newey_west_hac(G, bw)

        Sigma11 = Sigma_hat[:n_nonstat, :n_nonstat]
        sigma12 = Sigma_hat[:n_nonstat, n_nonstat]
        sigma22 = float(Sigma_hat[n_nonstat, n_nonstat])

        Sigma11_inv = np.linalg.pinv(Sigma11)

        # FM-adjusted dependent variable
        G_nonstat = G[:, :n_nonstat]
        y_tilde = y - G_nonstat @ Sigma11_inv @ sigma12

        # Bias correction
        Pi12 = Pi_hat[:n_nonstat, n_nonstat]
        Pi11 = Pi_hat[:n_nonstat, :n_nonstat]
        nu_hat = Pi12 - Pi11 @ Sigma11_inv @ sigma12

        # Selection matrix: [0, 0, I_{n_nonstat}] so S' * nu_hat selects
        # non-stationary part of Q (cols 2..2+n_nonstat)
        S = np.zeros((n_nonstat, Q.shape[1]))
        S[:, 2:2 + n_nonstat] = np.eye(n_nonstat)

        Qy_adj = Q.T @ y_tilde - T * S.T @ nu_hat
        rho_hat = QQinv @ Qy_adj
        uhat = y - Q @ rho_hat

        tau2 = float(sigma22 - sigma12 @ Sigma11_inv @ sigma12)
        V_rho = tau2 * QQinv
        method = "fmols"

    # beta_pos = lambda + eta (indices 1,2 in rho_hat)
    # beta_neg = eta
    beta_lr = np.array([rho_hat[1] + rho_hat[2], rho_hat[2]])

    # Jacobian for delta method
    J = np.zeros((2, len(rho_hat)))
    J[0, 1] = 1.0; J[0, 2] = 1.0   # d(beta_pos)/d(lambda)=1, /d(eta)=1
    J[1, 2] = 1.0                    # d(beta_neg)/d(eta)=1
    V_lr = J @ V_rho @ J.T

    if n_lin > 0:
        beta_lin = rho_hat[3:3 + n_lin]
        V_lin = V_rho[3:3 + n_lin, 3:3 + n_lin]
        J_full = np.vstack([J, np.zeros((n_lin, len(rho_hat)))])
        J_full[2:, 3:3 + n_lin] = np.eye(n_lin)
        V_lr_full = J_full @ V_rho @ J_full.T
    else:
        beta_lin = np.array([])
        V_lin = np.empty((0, 0))
        V_lr_full = V_lr.copy()

    # ECT
    ect = y - rho_hat[0] - beta_lr[0] * xpos - beta_lr[1] * xneg
    if n_lin > 0:
        ect -= zlin_arr @ beta_lin

    return FMOLSResult(
        beta_lr=beta_lr,
        V_lr=V_lr,
        beta_lin=beta_lin,
        V_lin=V_lin,
        V_lr_full=V_lr_full,
        tau2=tau2,
        alpha=float(rho_hat[0]),
        ect=ect,
        uhat=uhat,
        method=method,
        n_lin=n_lin,
    )


# ---------------------------------------------------------------------------
# FM-TOLS  (k > 1 asymmetric variables)
# ---------------------------------------------------------------------------

def fmtols(
    y: np.ndarray,
    xpos: np.ndarray,    # shape (T, k)
    x: np.ndarray,       # shape (T, k)
    xneg: np.ndarray,    # shape (T, k)
    zlin: np.ndarray | None = None,
    use_fm: bool = True,
    bw: int = 0,
) -> FMTOLSResult:
    """
    Fully-Modified TOLS (or plain TOLS) for k>1 decomposed variables.

    Model: y = alpha + trend*t + lambda'*m_hat + eta'*x + gamma'*z + u
    where m_hat = x_pos - mu_hat * t  (detrended positive partial sums)
    beta_pos = lambda + eta,  beta_neg = eta
    """
    T = len(y)
    kk = xpos.shape[1] if xpos.ndim == 2 else 1

    if zlin is None or (isinstance(zlin, np.ndarray) and zlin.size == 0):
        zlin_arr = np.empty((T, 0))
        n_lin = 0
    else:
        zlin_arr = np.atleast_2d(zlin).T if zlin.ndim == 1 else zlin
        n_lin = zlin_arr.shape[1]

    tt = np.arange(1, T + 1, dtype=float)

    # Detrend x_pos
    mhat = np.zeros((T, kk))
    for i in range(kk):
        mu_hat_i = (tt @ xpos[:, i]) / (tt @ tt)
        mhat[:, i] = xpos[:, i] - mu_hat_i * tt

    # First differences
    dmhat = np.zeros((T, kk))
    dmhat[1:, :] = np.diff(mhat, axis=0)

    dx = np.zeros((T, kk))
    dx[1:, :] = np.diff(x, axis=0) if x.ndim == 2 else np.diff(x).reshape(-1, 1)

    if n_lin > 0:
        dz = np.zeros((T, n_lin))
        dz[1:, :] = np.diff(zlin_arr, axis=0)
    else:
        dz = np.empty((T, 0))

    # R = [1, t, m_hat, x, z]
    R_cols = [np.ones((T, 1)), tt.reshape(-1, 1), mhat, x if x.ndim == 2 else x.reshape(-1, 1)]
    if n_lin > 0:
        R_cols.append(zlin_arr)
    R = np.concatenate(R_cols, axis=1)

    RR = R.T @ R
    RRinv = np.linalg.pinv(RR)
    rho_hat = RRinv @ (R.T @ y)
    uhat = y - R @ rho_hat

    # n_nonstat = 2*kk + n_lin
    n_nonstat = 2 * kk + n_lin

    if not use_fm:
        tau2 = float(np.var(uhat, ddof=1))
        V_rho = tau2 * RRinv
        method = "tols"
    else:
        ell_t_cols = [dmhat, dx if dx.ndim == 2 else dx.reshape(-1, 1)]
        if n_lin > 0:
            ell_t_cols.append(dz)
        ell_t = np.concatenate(ell_t_cols, axis=1)

        G = np.concatenate([ell_t, uhat.reshape(-1, 1)], axis=1)
        Sigma_bar, Pi_bar = newey_west_hac(G, bw)

        Sigma11 = Sigma_bar[:n_nonstat, :n_nonstat]
        sigma12 = Sigma_bar[:n_nonstat, n_nonstat]
        sigma22 = float(Sigma_bar[n_nonstat, n_nonstat])

        Sigma11_inv = np.linalg.pinv(Sigma11)

        y_bar = y - ell_t @ Sigma11_inv @ sigma12

        nu_bar = (Pi_bar[:n_nonstat, n_nonstat]
                  - Pi_bar[:n_nonstat, :n_nonstat] @ Sigma11_inv @ sigma12)

        # Sbar maps R to non-stationary cols: skip [1, t] => start at col 2
        Sbar = np.zeros((n_nonstat, R.shape[1]))
        Sbar[:, 2:2 + n_nonstat] = np.eye(n_nonstat)

        Ry_adj = R.T @ y_bar - T * Sbar.T @ nu_bar
        rho_hat = RRinv @ Ry_adj
        uhat = y - R @ rho_hat

        tau2 = float(sigma22 - sigma12 @ Sigma11_inv @ sigma12)
        V_rho = tau2 * RRinv
        method = "fmtols"

    # rho_hat: [alpha, trend, lambda_1..lambda_kk, eta_1..eta_kk, gamma_1..gamma_n_lin]
    lambda_hat = rho_hat[2:2 + kk]
    eta_hat = rho_hat[2 + kk:2 + 2 * kk]
    beta_pos = lambda_hat + eta_hat
    beta_neg = eta_hat

    if n_lin > 0:
        beta_lin = rho_hat[2 + 2 * kk:2 + 2 * kk + n_lin]
        V_lin = V_rho[2 + 2 * kk:2 + 2 * kk + n_lin, 2 + 2 * kk:2 + 2 * kk + n_lin]
    else:
        beta_lin = np.array([])
        V_lin = np.empty((0, 0))

    # Jacobian: [beta_pos, beta_neg] = J * rho_hat
    J = np.zeros((2 * kk, len(rho_hat)))
    J[:kk, 2:2 + kk] = np.eye(kk)      # d(beta_pos)/d(lambda)
    J[:kk, 2 + kk:2 + 2 * kk] = np.eye(kk)  # d(beta_pos)/d(eta)
    J[kk:, 2 + kk:2 + 2 * kk] = np.eye(kk)  # d(beta_neg)/d(eta)
    V_lr = J @ V_rho @ J.T

    if n_lin > 0:
        J_full = np.zeros((2 * kk + n_lin, len(rho_hat)))
        J_full[:2 * kk, :] = J
        J_full[2 * kk:, 2 + 2 * kk:2 + 2 * kk + n_lin] = np.eye(n_lin)
        V_lr_full = J_full @ V_rho @ J_full.T
    else:
        V_lr_full = V_lr.copy()

    # ECT
    x_arr = x if x.ndim == 2 else x.reshape(-1, 1)
    ect = y - rho_hat[0] - xpos @ beta_pos - xneg @ beta_neg
    if n_lin > 0:
        ect -= zlin_arr @ beta_lin

    return FMTOLSResult(
        beta_pos=beta_pos,
        beta_neg=beta_neg,
        V_lr=V_lr,
        beta_lin=beta_lin,
        V_lin=V_lin,
        V_lr_full=V_lr_full,
        tau2=tau2,
        ect=ect,
        uhat=uhat,
        method=method,
        n_lin=n_lin,
    )
