"""Validation tests for sabr.py and surface_pricing.py (run: python -m pytest -q)."""
import math

import numpy as np
import pytest

from binomial_tree import Dividend, OptionSpec, flat_curve, price
from sabr import (SabrParams, alpha_from_atm, calibrate_slice, fit_surface, hagan_vol,
                  implied_density, obloj_vol)
from surface_pricing import implied_vol, theo, with_

F, T = 100.0, 0.5
P = SabrParams(alpha=2.0, beta=0.5, rho=-0.4, nu=0.8)
KS = np.linspace(70, 130, 13)


def test_formulas_agree_at_the_money():
    assert obloj_vol(F, F, T, P)[0] == pytest.approx(hagan_vol(F, F, T, P)[0], rel=1e-12)


def test_formulas_agree_when_beta_is_one():
    p = SabrParams(0.25, 1.0, -0.3, 0.9)
    np.testing.assert_allclose(obloj_vol(F, KS, T, p), hagan_vol(F, KS, T, p), rtol=1e-12)


def test_obloj_is_continuous_as_beta_to_one():
    p1 = SabrParams(0.25, 1.0, -0.3, 0.9)
    pb = SabrParams(0.25 * F ** (1 - 0.9999), 0.9999, -0.3, 0.9)
    np.testing.assert_allclose(obloj_vol(F, KS, T, pb), obloj_vol(F, KS, T, p1), rtol=1e-4)


def test_obloj_nu_zero_is_local_vol_limit():
    p = SabrParams(2.0, 0.5, 0.0, 0.0)
    q = 0.5
    x = np.log(F / KS)
    i0 = np.where(np.abs(x) > 0, x * p.alpha * q / (F ** q - KS ** q + 1e-300), p.alpha * F ** (-q))
    i1 = q * q / 24 * p.alpha ** 2 / (F * KS) ** q
    np.testing.assert_allclose(obloj_vol(F, KS, T, p), i0 * (1 + i1 * T), rtol=1e-10)


def test_alpha_from_atm_round_trip():
    atm = hagan_vol(F, F, T, P)[0]
    assert alpha_from_atm(atm, F, T, P.beta, P.rho, P.nu) == pytest.approx(P.alpha, rel=1e-10)


@pytest.mark.parametrize("formula", ["obloj", "hagan"])
@pytest.mark.parametrize("pin_atm", [True, False])
def test_calibration_recovers_parameters(formula, pin_atm):
    vols = obloj_vol(F, KS, T, P) if formula == "obloj" else hagan_vol(F, KS, T, P)
    fit = calibrate_slice(F, T, KS, vols, beta=0.5, pin_atm=pin_atm, formula=formula)
    assert fit.rmse < 1e-6
    assert fit.params.rho == pytest.approx(P.rho, abs=1e-4)
    assert fit.params.nu == pytest.approx(P.nu, abs=1e-4)
    assert fit.params.alpha == pytest.approx(P.alpha, rel=1e-4)


def test_calibration_is_robust_to_noise():
    rng = np.random.default_rng(0)
    vols = obloj_vol(F, KS, T, P) + rng.normal(0, 0.002, KS.size)   # 0.2 vol-pt noise
    fit = calibrate_slice(F, T, KS, vols, beta=0.5)
    assert fit.rmse < 0.003
    assert fit.params.rho == pytest.approx(P.rho, abs=0.1)
    assert fit.params.nu == pytest.approx(P.nu, abs=0.15)


def test_surface_interpolation():
    fwd = lambda t: 100.0
    s1, s2 = SabrParams(2.0, 0.5, -0.4, 1.0), SabrParams(1.8, 0.5, -0.3, 0.6)
    surf = fit_surface([(0.25, 100.0, KS, obloj_vol(100, KS, 0.25, s1)),
                        (1.0, 100.0, KS, obloj_vol(100, KS, 1.0, s2))], fwd, beta=0.5)
    assert surf.vol(90.0, 0.25) == pytest.approx(obloj_vol(100, 90.0, 0.25, s1)[0], abs=1e-6)
    v0, v1, vm = surf.vol(100, 0.25), surf.vol(100, 1.0), surf.vol(100, 0.6)
    assert min(v0, v1) - 1e-9 <= vm <= max(v0, v1) + 1e-9


def test_density_positive_in_normal_equity_region():
    d = implied_density(F, T, P, np.linspace(50, 200, 301))
    assert d.min() > 0


def test_tree_implied_vol_round_trip_american():
    opt = OptionSpec(100, 95, 0.5, 0.3, is_call=False, american=True,
                     rate_curve=flat_curve(0.04), dividends=[Dividend(0.2, 1.0)])
    mkt = price(opt, 300)
    assert implied_vol(opt, mkt, 300) == pytest.approx(0.3, abs=1e-6)


def test_theo_uses_surface_vol():
    surf = fit_surface([(0.5, 100.0, KS, obloj_vol(100, KS, 0.5, P))], lambda t: 100.0, beta=0.5)
    opt = OptionSpec(100, 90, 0.5, 0.0, is_call=False, american=True)
    out = theo(surf, opt, 300)
    assert out["vol"] == pytest.approx(obloj_vol(100, 90, 0.5, P)[0], abs=1e-6)
    assert out["price"] == pytest.approx(price(with_(opt, sigma=out["vol"]), 300), abs=1e-10)
