"""Glue between the SABR surface and the binomial tree (SABR-plus-tree workflow).

  market option prices --(tree inversion)--> equivalent Black vols --(SABR fit)--> surface
  surface.vol(K, T) --> binomial tree --> theoretical prices and Greeks (American or European)
"""
from __future__ import annotations

from scipy.optimize import brentq

from binomial_tree import OptionSpec, greeks, price
from sabr import SabrSurface


def with_(opt: OptionSpec, **kw) -> OptionSpec:
    return OptionSpec(**{**opt.__dict__, **kw})


def implied_vol(opt: OptionSpec, market_price: float, steps: int = 300,
                lo: float = 1e-3, hi: float = 5.0) -> float:
    """Equivalent Black vol that reproduces market_price in the tree (handles American exercise)."""
    return brentq(lambda s: price(with_(opt, sigma=s), steps) - market_price, lo, hi, xtol=1e-8)


def theo(surface: SabrSurface, opt: OptionSpec, steps: int = 500) -> dict[str, float]:
    """Theoretical price and Greeks with vol read from the fitted SABR surface."""
    vol = surface.vol(opt.strike, opt.expiry)
    out = greeks(with_(opt, sigma=vol), steps)
    return {"vol": vol, **out}
