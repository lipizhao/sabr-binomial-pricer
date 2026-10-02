"""
Forward-based CRR binomial tree for American and European equity options.

Follows the design in "Notes Pricing Model":
  * CRR up/down factors  u = exp(sigma*sqrt(dt)),  d = 1/u
  * Full interest-rate curve and implied borrow-rate curve (continuously
    compounded zero rates as functions of maturity)
  * Discrete cash dividends handled as in Hull (2006): the tree is built on
    the "escrowed" process X = S - PV(dividends paid before expiry); the spot
    at each node is X plus the PV (at that node's time) of dividends still to
    come before expiry.
  * The tree is constructed in terms of the forward price to expiration: node
    values are X-forward(t) * u^(2j - i), so the tree is centered on the
    forward and the risk-neutral probability p = (1 - d) / (u - d) is constant
    even with non-flat rate / borrow curves.
  * European options use the same tree with early exercise disallowed.
  * Optional smoothing (Broadie-Detemple, 1996), on by default - see below.

Why smoothing?
  A plain binomial tree prices the option from its payoff max(K - S, 0) at
  expiry. That payoff has a kink at S = K, and as the number of steps changes,
  the tree's final nodes land on different sides of the kink. The price
  therefore zig-zags between odd and even step counts instead of converging
  smoothly (e.g. 12.100 at 50 steps, 12.072 at 51, 12.056 at 100, 12.087 at 101).

  Smoothing replaces the last tree step with the exact Black formula: at each
  node one step before expiry, the continuation value is the closed-form
  European price with time-to-expiry dt (no early exercise can happen within
  that last dt anyway, apart from at the node itself, which we still check).
  The closed form "knows" the payoff exactly, so the kink no longer causes the
  zig-zag, and the price converges smoothly (12.0745, 12.0758, 12.0707,
  12.0716 for the same step counts). Pass smoothing=False to see the raw tree.

Greek conventions (per the notes):
  delta : change in premium per 1 unit change in the underlying price
  gamma : change in delta per 1 unit change in the underlying price
  vega  : change in premium per 1 percentage point change in implied vol
  theta : change in premium per calendar day (negative = time decay)
  rho   : change in premium per 1 percentage point parallel shift in rates
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

Curve = Callable[[float], float]  # maturity t (years) -> continuously compounded zero rate


def flat_curve(rate: float) -> Curve:
    return lambda t: rate


def interp_curve(tenors: Sequence[float], rates: Sequence[float]) -> Curve:
    """Linear interpolation of zero rates, flat extrapolation."""
    tenors = np.asarray(tenors, float)
    rates = np.asarray(rates, float)
    return lambda t: float(np.interp(t, tenors, rates))


def shifted(curve: Curve, bump: float) -> Curve:
    return lambda t: curve(t) + bump


@dataclass(frozen=True)
class Dividend:
    time: float    # ex-dividend time in years
    amount: float  # cash amount


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_price(F: float, K: float, T: float, sigma: float, df: float, is_call: bool) -> float:
    """Black-76 price on a forward, discount factor df."""
    if T <= 0 or sigma <= 0:
        return df * max((F - K) if is_call else (K - F), 0.0)
    sd = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if is_call:
        return df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    return df * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


def _black_vec(F: np.ndarray, K: float, T: float, sigma: float, df: float, is_call: bool) -> np.ndarray:
    return np.array([black_price(f, K, T, sigma, df, is_call) for f in F])


@dataclass
class OptionSpec:
    spot: float
    strike: float
    expiry: float                 # years
    sigma: float                  # Black (implied) vol, e.g. from a SABR-fitted surface
    is_call: bool
    american: bool = True
    rate_curve: Curve = flat_curve(0.0)
    borrow_curve: Curve = flat_curve(0.0)
    dividends: Sequence[Dividend] = ()

    # --- curve helpers ---
    def df(self, t: float) -> float:
        """Risk-free discount factor to time t."""
        return math.exp(-self.rate_curve(t) * t)

    def borrow_factor(self, t: float) -> float:
        return math.exp(-self.borrow_curve(t) * t)

    def live_dividends(self) -> list[Dividend]:
        return [d for d in self.dividends if 0.0 < d.time <= self.expiry]

    def pv_dividends(self, t: float = 0.0) -> float:
        """PV at time t of dividends with ex-date in (t, expiry]."""
        return sum(d.amount * self.df(d.time) / self.df(t)
                   for d in self.live_dividends() if d.time > t)

    def forward(self) -> float:
        """Forward price of the underlying to expiration."""
        x0 = self.spot - self.pv_dividends(0.0)
        return x0 * self.borrow_factor(self.expiry) / self.df(self.expiry)


def _build_and_roll(option: OptionSpec, steps: int, smoothing: bool):
    """Build the tree and roll it back to today.

    Returns (price_today, nodes_for_greeks, step_length) where nodes_for_greeks
    maps step 1 and step 2 to (spot prices, option values) at those nodes.
    Math symbols follow the guide: u/d = up/down factors, p = up probability.
    """
    expiry, strike, vol = option.expiry, option.strike, option.sigma
    step_length = expiry / steps
    u = math.exp(vol * math.sqrt(step_length))
    d = 1.0 / u
    p = (1.0 - d) / (u - d)        # keeps X / forward(X) a martingale (constant p)

    step_times = np.linspace(0.0, expiry, steps + 1)
    discount_factors = np.array([option.df(t) for t in step_times])
    borrow_factors = np.array([option.borrow_factor(t) for t in step_times])
    escrowed_spot = option.spot - option.pv_dividends(0.0)        # X0 = S0 - PV(divs)
    escrowed_forward = escrowed_spot * borrow_factors / discount_factors  # forward of X to each step
    pv_remaining_divs = np.array([option.pv_dividends(t) for t in step_times])
    payoff_sign = 1.0 if option.is_call else -1.0

    def spot_at_step(step: int) -> np.ndarray:
        """Spot price at every node of a step: X node value plus PV of dividends still to come."""
        up_moves = np.arange(step + 1)
        return escrowed_forward[step] * u ** (2 * up_moves - step) + pv_remaining_divs[step]

    def exercise_value(spot: np.ndarray) -> np.ndarray:
        return np.maximum(payoff_sign * (spot - strike), 0.0)

    if smoothing and steps >= 3:
        # Smoothing: start one step before expiry and value each node with the exact
        # Black formula over the final step (see module docstring for why).
        last_step = steps - 1
        escrowed_nodes = escrowed_forward[last_step] * u ** (2 * np.arange(last_step + 1) - last_step)
        forward_to_expiry = escrowed_nodes * escrowed_forward[steps] / escrowed_forward[last_step]
        one_step_discount = discount_factors[steps] / discount_factors[last_step]
        values = _black_vec(forward_to_expiry, strike, step_length, vol, one_step_discount, option.is_call)
        if option.american:
            values = np.maximum(values, exercise_value(spot_at_step(last_step)))
        first_rollback_step = steps - 1
    else:
        # Plain tree: start from the payoff at expiry.
        values = exercise_value(spot_at_step(steps))
        first_rollback_step = steps

    nodes_for_greeks = {}
    for step in range(first_rollback_step - 1, -1, -1):
        one_step_discount = discount_factors[step + 1] / discount_factors[step]
        values = one_step_discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if option.american:
            values = np.maximum(values, exercise_value(spot_at_step(step)))
        if step in (1, 2):
            nodes_for_greeks[step] = (spot_at_step(step), values.copy())
    return float(values[0]), nodes_for_greeks, step_length


def price(option: OptionSpec, steps: int = 500, smoothing: bool = True) -> float:
    """Tree price of the option.

    steps     : number of time steps (500 is a good default; 1000+ for Greeks)
    smoothing : use the exact Black value over the last step to remove the
                odd/even zig-zag in convergence (see module docstring)
    """
    return _build_and_roll(option, steps, smoothing)[0]


def greeks(option: OptionSpec, steps: int = 500, smoothing: bool = True) -> dict[str, float]:
    """Price plus Greeks. Delta, gamma and theta are read off the tree nodes at
    steps 1 and 2 (no extra trees); vega and rho by central bump-and-reprice."""
    price_today, nodes_for_greeks, step_length = _build_and_roll(option, steps, smoothing)
    spots_step1, values_step1 = nodes_for_greeks[1]
    spots_step2, values_step2 = nodes_for_greeks[2]

    delta = (values_step1[1] - values_step1[0]) / (spots_step1[1] - spots_step1[0])
    delta_upper = (values_step2[2] - values_step2[1]) / (spots_step2[2] - spots_step2[1])
    delta_lower = (values_step2[1] - values_step2[0]) / (spots_step2[1] - spots_step2[0])
    gamma = (delta_upper - delta_lower) / (0.5 * (spots_step2[2] - spots_step2[0]))
    # middle node at step 2 is (approximately) today's spot, two steps later
    theta_per_year = (values_step2[1] - price_today) / (2.0 * step_length)

    def reprice(**changes) -> float:
        return price(replace(option, **changes), steps, smoothing)

    bump = 0.01   # 1 vol point for vega, 1% rate shift for rho
    vega = (reprice(sigma=option.sigma + bump) - reprice(sigma=option.sigma - bump)) / 2.0
    rho = (reprice(rate_curve=shifted(option.rate_curve, bump))
           - reprice(rate_curve=shifted(option.rate_curve, -bump))) / 2.0
    return {"price": price_today, "delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta_per_year / 365.0, "rho": rho}
