"""Validation tests for binomial_tree.py (run: python -m pytest -q)."""
import math

import pytest

from binomial_tree import (Dividend, OptionSpec, black_price, flat_curve,
                           greeks, interp_curve, price)


def bs(opt: OptionSpec) -> float:
    """Closed-form Black value on the escrowed forward (exact for Europeans in this model)."""
    return black_price(opt.forward(), opt.strike, opt.expiry, opt.sigma,
                       opt.df(opt.expiry), opt.is_call)


def base(**kw) -> OptionSpec:
    d = dict(spot=100.0, strike=100.0, expiry=1.0, sigma=0.25, is_call=True,
             american=False, rate_curve=flat_curve(0.05))
    d.update(kw)
    return OptionSpec(**d)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_european_matches_black_scholes(is_call, strike):
    opt = base(is_call=is_call, strike=strike)
    assert price(opt, 1000) == pytest.approx(bs(opt), abs=2e-3)


def test_european_with_curves_borrow_and_dividends_matches_black():
    opt = base(is_call=False,
               rate_curve=interp_curve([0.25, 1.0, 2.0], [0.040, 0.045, 0.047]),
               borrow_curve=flat_curve(0.005),
               dividends=[Dividend(0.3, 1.0), Dividend(0.8, 1.0)])
    assert price(opt, 1000) == pytest.approx(bs(opt), abs=2e-3)


def test_put_call_parity_european():
    kw = dict(rate_curve=flat_curve(0.04), borrow_curve=flat_curve(0.01),
              dividends=[Dividend(0.5, 2.0)])
    c = price(base(is_call=True, **kw), 800)
    p = price(base(is_call=False, **kw), 800)
    o = base(**kw)
    assert c - p == pytest.approx(o.df(o.expiry) * (o.forward() - o.strike), abs=2e-3)


def test_american_call_no_dividend_equals_european():
    am = price(base(american=True), 800)
    eu = price(base(american=False), 800)
    assert am == pytest.approx(eu, abs=1e-6)


def test_american_put_bounds():
    am = price(base(is_call=False, american=True, strike=110), 800)
    eu = price(base(is_call=False, american=False, strike=110), 800)
    assert am > eu
    assert am >= 110 - 100


def test_american_call_early_exercise_before_big_dividend():
    kw = dict(dividends=[Dividend(0.5, 8.0)], strike=80.0)
    am = price(base(american=True, **kw), 800)
    eu = price(base(american=False, **kw), 800)
    assert am > eu + 0.1


def test_hull_american_put_reference():
    # Hull, Options Futures & Other Derivatives: S=K=50, r=10%, sigma=40%, T=5/12.
    # Converged American put value ~ 4.28.
    opt = OptionSpec(spot=50, strike=50, expiry=5 / 12, sigma=0.40, is_call=False,
                     american=True, rate_curve=flat_curve(0.10))
    assert price(opt, 2000) == pytest.approx(4.28, abs=0.01)


def test_convergence_is_smooth_with_smoothing():
    opt = base(is_call=False, american=True)
    vals = [price(opt, n) for n in (400, 401, 800, 801)]
    assert max(vals) - min(vals) < 2e-3


def test_greeks_against_bumps():
    opt = base(is_call=False, american=True,
               dividends=[Dividend(0.4, 1.0)], borrow_curve=flat_curve(0.005))
    g = greeks(opt, 1000)
    h = 0.5
    up = price(OptionSpec(**{**opt.__dict__, "spot": opt.spot + h}), 1000)
    dn = price(OptionSpec(**{**opt.__dict__, "spot": opt.spot - h}), 1000)
    assert g["delta"] == pytest.approx((up - dn) / (2 * h), abs=5e-3)
    assert g["gamma"] == pytest.approx((up - 2 * g["price"] + dn) / h**2, abs=2e-3)
    assert g["vega"] > 0 and g["theta"] < 0
