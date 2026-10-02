"""
SABR volatility smile / surface fitting.

References (in the project folder):
  [H]  Hagan, Kumar, Lesniewski, Woodward, "Managing Smile Risk", Wilmott 2002.
  [O]  Obloj, "Fine-Tune Your Smile: Correction to Hagan et al.", 2008.

Model (forward measure):  dF = a F^beta dW1,  da = nu a dW2,  d<W1,W2> = rho dt,  a(0) = alpha.

Two equivalent-Black-vol formulas are provided, both of the form I0(K) * (1 + I1(K) * T):
  * "hagan": eq. (2.17a)-(2.17c) of [H]
  * "obloj": corrected leading term I0 from [O] (Berestycki-Busca-Florent), with Hagan's I1.
    Consistent as beta -> 1 and pushes the negative-density problem at low strikes /
    long maturities further out. This is the default.
The ATM value is identical under both formulas, so alpha can be backed out of the ATM vol with
Hagan's eq. (2.18) either way.

Calibration follows [H] section 3: fix beta, parameterize by the ATM vol instead of alpha
(alpha solved on the fly from the ATM cubic), and fit rho (skew) and nu (smile) by least squares.
Each expiry is fitted independently; SabrSurface interpolates between expiries in total variance
at constant log-moneyness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares

RHO_MAX = 0.999


@dataclass(frozen=True)
class SabrParams:
    alpha: float
    beta: float
    rho: float
    nu: float


# ---------------------------------------------------------------- helpers
def _z_over_x(z: np.ndarray, rho: float) -> np.ndarray:
    """z / X(z) with X(z) = log((sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho)); stable near z = 0."""
    z = np.asarray(z, float)
    out = np.empty_like(z)
    small = np.abs(z) < 1e-7
    out[small] = 1.0 - 0.5 * rho * z[small]
    zs = z[~small]
    x = np.log((np.sqrt(1.0 - 2.0 * rho * zs + zs * zs) + zs - rho) / (1.0 - rho))
    out[~small] = zs / x
    return out


def _phi(y: np.ndarray) -> np.ndarray:
    """(1 - exp(-y)) / y, with phi(0) = 1."""
    y = np.asarray(y, float)
    out = np.ones_like(y)
    nz = np.abs(y) > 1e-12
    out[nz] = -np.expm1(-y[nz]) / y[nz]
    return out


def _i1(F: float, K: np.ndarray, p: SabrParams) -> np.ndarray:
    """First-order time correction (common to [H] and [O], eq. (3) of [O])."""
    q = 1.0 - p.beta
    fk = F * K
    return (q * q / 24.0 * p.alpha ** 2 / fk ** q
            + 0.25 * p.rho * p.beta * p.nu * p.alpha / fk ** (0.5 * q)
            + (2.0 - 3.0 * p.rho ** 2) / 24.0 * p.nu ** 2)


# ---------------------------------------------------------------- vol formulas
def hagan_vol(F: float, K, T: float, p: SabrParams) -> np.ndarray:
    """Hagan et al. (2002) eq. (2.17a)-(2.17c)."""
    K = np.atleast_1d(np.asarray(K, float))
    q = 1.0 - p.beta
    lfk = np.log(F / K)
    fk_q2 = (F * K) ** (0.5 * q)
    denom = fk_q2 * (1.0 + q ** 2 / 24.0 * lfk ** 2 + q ** 4 / 1920.0 * lfk ** 4)
    z = p.nu / p.alpha * fk_q2 * lfk
    return p.alpha / denom * _z_over_x(z, p.rho) * (1.0 + _i1(F, K, p) * T)


def obloj_vol(F: float, K, T: float, p: SabrParams) -> np.ndarray:
    """Obloj (2008) corrected I0 (Table 1, Berestycki et al. column) times Hagan's (1 + I1 T).

    With x = ln(F/K), q = 1 - beta:
        z  = nu (F^q - K^q) / (alpha q)            (-> nu x / alpha as beta -> 1)
        I0 = nu x / X(z)
    rewritten stably as  I0 = alpha / (F^q phi(q x)) * z / X(z),  which also covers nu = 0 and K = F.
    """
    K = np.atleast_1d(np.asarray(K, float))
    q = 1.0 - p.beta
    x = np.log(F / K)
    g = F ** q * _phi(q * x)                       # (F^q - K^q) / (q x)
    z = p.nu * x * g / p.alpha
    i0 = p.alpha / g * _z_over_x(z, p.rho)
    return i0 * (1.0 + _i1(F, K, p) * T)


FORMULAS: dict[str, Callable] = {"hagan": hagan_vol, "obloj": obloj_vol}


def sabr_vol(F: float, K, T: float, p: SabrParams, formula: str = "obloj") -> np.ndarray:
    return FORMULAS[formula](F, K, T, p)


# ---------------------------------------------------------------- ATM parameterization
def alpha_from_atm(atm_vol: float, F: float, T: float, beta: float, rho: float, nu: float) -> float:
    """Invert Hagan (2.18) for alpha: smallest positive real root of the ATM cubic."""
    q = 1.0 - beta
    fq = F ** q
    coeffs = [q * q * T / (24.0 * fq * fq),
              rho * beta * nu * T / (4.0 * fq),
              1.0 + (2.0 - 3.0 * rho * rho) * nu * nu * T / 24.0,
              -atm_vol * fq]
    roots = np.roots(coeffs) if coeffs[0] != 0.0 else np.roots(coeffs[1:])
    real = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0.0]
    return min(real) if real else atm_vol * fq


def atm_vol_from_quotes(F: float, strikes, vols, n_points: int = 3) -> float:
    """ATM vol from a quadratic in log-moneyness through the quotes nearest the forward
    (exact interpolation with the default 3 points; use more points to smooth noisy quotes)."""
    k = np.log(np.asarray(strikes, float) / F)
    v = np.asarray(vols, float)
    idx = np.argsort(np.abs(k))[: max(3, min(n_points, len(k)))]
    deg = 2 if len(idx) >= 3 else 1
    return float(np.polyval(np.polyfit(k[idx], v[idx], deg), 0.0))


# ---------------------------------------------------------------- calibration
@dataclass
class SliceFit:
    expiry: float
    forward: float
    params: SabrParams
    atm_vol: float
    rmse: float                 # root-mean-square vol error (decimal, e.g. 0.001 = 0.1 vol pt)
    max_abs_err: float
    formula: str
    strikes: np.ndarray = field(repr=False)
    market_vols: np.ndarray = field(repr=False)

    def vol(self, K) -> np.ndarray:
        return sabr_vol(self.forward, K, self.expiry, self.params, self.formula)


def calibrate_slice(F: float, T: float, strikes, vols, beta: float = 0.5,
                    weights=None, pin_atm: bool = True, atm_vol: float | None = None,
                    formula: str = "obloj") -> SliceFit:
    """Fit SABR to one expiry's smile.

    pin_atm=True  -> [H] recommendation: alpha tied to the ATM vol, fit (rho, nu) only.
    pin_atm=False -> fit (alpha, rho, nu) freely; ATM is matched only in the least-squares sense.
    weights       -> per-quote weights (e.g. vega, 1/bid-ask); default equal.
    """
    K = np.asarray(strikes, float)
    mv = np.asarray(vols, float)
    w = np.ones_like(K) if weights is None else np.sqrt(np.asarray(weights, float))
    if atm_vol is None:
        atm_vol = atm_vol_from_quotes(F, K, mv)
    f = FORMULAS[formula]

    if pin_atm:
        def unpack(x):
            rho, nu = x
            return SabrParams(alpha_from_atm(atm_vol, F, T, beta, rho, nu), beta, rho, nu)
        starts = [(r, n) for r in (-0.6, -0.2, 0.2) for n in (0.3, 0.8, 1.5)]
        lb, ub = [-RHO_MAX, 1e-4], [RHO_MAX, 5.0]
    else:
        a0 = atm_vol * F ** (1.0 - beta)
        def unpack(x):
            return SabrParams(x[0], beta, x[1], x[2])
        starts = [(a0, r, n) for r in (-0.6, -0.2, 0.2) for n in (0.3, 0.8, 1.5)]
        lb, ub = [1e-6, -RHO_MAX, 1e-4], [np.inf, RHO_MAX, 5.0]

    def resid(x):
        return w * (f(F, K, T, unpack(x)) - mv)

    # coarse start selection, then bounded Levenberg-Marquardt-style refinement
    x0 = min(starts, key=lambda s: float(np.sum(resid(np.array(s)) ** 2)))
    sol = least_squares(resid, np.array(x0), bounds=(lb, ub), x_scale="jac", xtol=1e-12, ftol=1e-12)
    p = unpack(sol.x)
    err = f(F, K, T, p) - mv
    return SliceFit(T, F, p, atm_vol, float(np.sqrt(np.mean(err ** 2))),
                    float(np.max(np.abs(err))), formula, K, mv)


# ---------------------------------------------------------------- surface
class SabrSurface:
    """Collection of per-expiry SABR fits. Between expiries: linear in total variance at
    constant log-moneyness ln(K / F(T)); outside the range: flat in vol at that moneyness."""

    def __init__(self, fits: Sequence[SliceFit], forward_fn: Callable[[float], float]):
        self.fits = sorted(fits, key=lambda s: s.expiry)
        self.forward_fn = forward_fn
        self.expiries = np.array([s.expiry for s in self.fits])

    def _slice_vol(self, s: SliceFit, k: float) -> float:
        return float(s.vol(s.forward * math.exp(k))[0])

    def vol(self, K: float, T: float) -> float:
        k = math.log(K / self.forward_fn(T))
        E = self.expiries
        if T <= E[0]:
            return self._slice_vol(self.fits[0], k)
        if T >= E[-1]:
            return self._slice_vol(self.fits[-1], k)
        i = int(np.searchsorted(E, T)) - 1
        t0, t1 = E[i], E[i + 1]
        w0 = self._slice_vol(self.fits[i], k) ** 2 * t0
        w1 = self._slice_vol(self.fits[i + 1], k) ** 2 * t1
        w = w0 + (w1 - w0) * (T - t0) / (t1 - t0)
        return math.sqrt(w / T)


def fit_surface(slices: Sequence[tuple[float, float, Sequence[float], Sequence[float]]],
                forward_fn: Callable[[float], float], **kw) -> SabrSurface:
    """slices: iterable of (expiry, forward, strikes, implied_vols). kw passed to calibrate_slice."""
    return SabrSurface([calibrate_slice(F, T, K, v, **kw) for T, F, K, v in slices], forward_fn)


# ---------------------------------------------------------------- arbitrage diagnostic
def implied_density(F: float, T: float, p: SabrParams, strikes, formula: str = "obloj") -> np.ndarray:
    """Risk-neutral density d2C/dK2 (undiscounted Black calls) on a uniform strike grid.
    Negative values flag butterfly arbitrage (the pathology discussed in [O])."""
    from binomial_tree import black_price
    K = np.asarray(strikes, float)
    h = K[1] - K[0]
    vols = sabr_vol(F, K, T, p, formula)
    C = np.array([black_price(F, k, T, v, 1.0, True) for k, v in zip(K, vols)])
    return (C[2:] - 2 * C[1:-1] + C[:-2]) / h ** 2
