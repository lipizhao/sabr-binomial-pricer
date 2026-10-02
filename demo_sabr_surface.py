"""
End-to-end demo of the SABR-plus-tree workflow on a synthetic equity market.

Run:  python demo_sabr_surface.py

Steps:
  1. Build a made-up market: American option prices (out-of-the-money puts below
     the forward, calls above), generated from a known SABR surface plus noise
     and rounded to the cent.
  2. Back out the equivalent Black vol of each quote with the binomial tree.
  3. Fit SABR to each expiry (beta fixed, alpha tied to the ATM vol, fit rho and nu)
     and compare the fitted parameters with the ones used to make the data.
  4. Price a dummy American put off the fitted surface (price and Greeks).
Extras: Hagan vs Obloj formula, sensitivity to the choice of beta, an arbitrage
(density) check, and a chart saved as sabr_fit.png.
"""
from dataclasses import replace

import numpy as np

from binomial_tree import Dividend, OptionSpec, flat_curve, interp_curve, price
from sabr import SabrParams, alpha_from_atm, calibrate_slice, fit_surface, implied_density, sabr_vol
from surface_pricing import implied_vol, theo

TREE_STEPS = 300            # steps used when inverting quotes (speed matters: ~900 tree runs)
PRICING_STEPS = 1000        # steps for the final dummy option
BETA = 0.5                  # SABR beta: fixed, not fitted (see guide section 4.4)
QUOTE_NOISE = 0.0015        # 0.15 vol points of random noise added to the synthetic vols

# "True" SABR smile per expiry, used only to manufacture the synthetic market.
# expiry in years -> (ATM vol, rho, nu)
TRUE_SMILES = {
    1 / 12: (0.260, -0.55, 2.2),
    0.25:   (0.245, -0.50, 1.4),
    0.50:   (0.235, -0.45, 1.0),
    1.00:   (0.225, -0.40, 0.7),
}


def make_market_template() -> OptionSpec:
    """Stock, rates, borrow and dividends shared by every option in the demo.
    Strike, expiry, call/put and vol are filled in per option with replace()."""
    return OptionSpec(
        spot=100.0, strike=100.0, expiry=1.0, sigma=0.25, is_call=False, american=True,
        rate_curve=interp_curve([0.08, 0.25, 0.5, 1.0, 2.0], [0.0435, 0.0430, 0.0420, 0.0400, 0.0385]),
        borrow_curve=flat_curve(0.0050),
        dividends=[Dividend(time, 0.75) for time in (0.15, 0.40, 0.65, 0.90)],
    )


def build_synthetic_market(market_template, forward_of, random_generator):
    """Steps 1-2: returns a list of (expiry, forward, strikes, tree_implied_vols)
    and the true SABR parameters per expiry."""
    print("Step 1-2: synthetic American quotes -> tree-implied Black vols")
    expiry_slices, true_params_by_expiry = [], {}
    for expiry, (atm_vol, rho, nu) in TRUE_SMILES.items():
        forward = forward_of(expiry)
        alpha = alpha_from_atm(atm_vol, forward, expiry, BETA, rho, nu)
        true_params = SabrParams(alpha, BETA, rho, nu)
        true_params_by_expiry[expiry] = true_params

        # strikes widen with maturity, rounded to $0.50
        log_moneyness_width = 0.12 + 0.30 * np.sqrt(expiry)
        log_moneyness = np.linspace(-log_moneyness_width, 0.7 * log_moneyness_width, 15)
        strikes = np.unique(np.round(forward * np.exp(log_moneyness) * 2) / 2)

        noisy_vols = sabr_vol(forward, strikes, expiry, true_params) \
            + random_generator.normal(0, QUOTE_NOISE, strikes.size)

        tree_implied_vols = []
        for strike, vol in zip(strikes, noisy_vols):
            quoted_option = replace(market_template, strike=strike, expiry=expiry,
                                    is_call=bool(strike >= forward))      # OTM side
            market_price = round(price(replace(quoted_option, sigma=vol), TREE_STEPS), 2)
            tree_implied_vols.append(implied_vol(quoted_option, max(market_price, 0.01), TREE_STEPS))

        expiry_slices.append((expiry, forward, strikes, np.array(tree_implied_vols)))
        print(f"  T={expiry:5.3f}  F={forward:8.4f}  {len(strikes)} strikes "
              f"{strikes[0]:.1f}-{strikes[-1]:.1f}")
    return expiry_slices, true_params_by_expiry


def report_fit(surface, true_params_by_expiry) -> None:
    print("\nStep 3: SABR fit per expiry (beta=0.5 fixed, alpha from ATM, Obloj formula)")
    print("  expiry   ATM vol   alpha    rho(fit/true)    nu(fit/true)   RMSE(vol pts)  max err")
    for slice_fit in surface.fits:
        true_params = true_params_by_expiry[slice_fit.expiry]
        fitted = slice_fit.params
        print(f"  {slice_fit.expiry:6.3f}  {slice_fit.atm_vol:7.4f}  {fitted.alpha:7.4f}  "
              f"{fitted.rho:6.3f} /{true_params.rho:6.3f}   {fitted.nu:6.3f} /{true_params.nu:6.3f}   "
              f"{100 * slice_fit.rmse:8.3f}     {100 * slice_fit.max_abs_err:6.3f}")


def compare_hagan_and_obloj(expiry_slices) -> None:
    print("\nHagan (2002) vs Obloj (2008) formula, RMSE in vol pts:")
    for expiry, forward, strikes, market_vols in expiry_slices:
        hagan_rmse = calibrate_slice(forward, expiry, strikes, market_vols, beta=BETA, formula="hagan").rmse
        obloj_rmse = calibrate_slice(forward, expiry, strikes, market_vols, beta=BETA, formula="obloj").rmse
        print(f"  T={expiry:5.3f}   hagan {100 * hagan_rmse:.4f}   obloj {100 * obloj_rmse:.4f}")


def show_beta_sensitivity(expiry_slices) -> None:
    print("\nBeta sensitivity on the 6M slice (Hagan: beta is poorly identified from one smile):")
    expiry, forward, strikes, market_vols = expiry_slices[2]
    for beta in (0.0, 0.3, 0.5, 0.7, 1.0):
        slice_fit = calibrate_slice(forward, expiry, strikes, market_vols, beta=beta)
        print(f"  beta={beta:.1f}  rho={slice_fit.params.rho:6.3f}  nu={slice_fit.params.nu:6.3f}  "
              f"RMSE={100 * slice_fit.rmse:.3f} vol pts")


def check_densities(surface) -> None:
    print("\nDensity check (min d2C/dK2 over 0.3F-2.5F, should be > 0):")
    for slice_fit in surface.fits:
        strike_grid = np.linspace(0.3, 2.5, 441) * slice_fit.forward
        density = implied_density(slice_fit.forward, slice_fit.expiry, slice_fit.params, strike_grid)
        status = "OK" if density.min() > 0 else "ARBITRAGE"
        print(f"  T={slice_fit.expiry:5.3f}  min density {density.min():.2e}  {status}")


def price_dummy_put(surface, market_template, forward_of) -> None:
    print("\nStep 4: dummy American put off the fitted surface")
    american_put = replace(market_template, strike=97.5, expiry=0.40, is_call=False, american=True)
    result = theo(surface, american_put, steps=PRICING_STEPS)
    european_put = replace(american_put, sigma=result["vol"], american=False)
    european_price = price(european_put, PRICING_STEPS)
    print(f"  S=100, K=97.5, T=0.40y (between the 3M and 6M slices), forward {forward_of(0.40):.4f}")
    for output_name, output_value in result.items():
        print(f"  {output_name:<8} {output_value:10.4f}")
    print(f"  European {european_price:10.4f}   early-exercise premium "
          f"{result['price'] - european_price:.4f}")


def save_chart(surface, file_name: str = "sabr_fit.png") -> None:
    """One panel per expiry: market vols (circles) vs SABR fit (line)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed - chart skipped)")
        return
    surface_color, ink, muted_ink, grid_color = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    fit_color, market_color = "#2a78d6", "#eb6834"
    figure, axes = plt.subplots(1, len(surface.fits), figsize=(14, 3.8), sharey=True, facecolor=surface_color)
    for axis, slice_fit in zip(axes, surface.fits):
        axis.set_facecolor(surface_color)
        fine_strikes = np.linspace(slice_fit.strikes.min(), slice_fit.strikes.max(), 200)
        axis.plot(fine_strikes / slice_fit.forward, 100 * slice_fit.vol(fine_strikes),
                  color=fit_color, lw=2, label="SABR fit")
        axis.plot(slice_fit.strikes / slice_fit.forward, 100 * slice_fit.market_vols, "o", ms=6,
                  mfc="none", mec=market_color, mew=1.5, label="Market (tree-implied)")
        axis.set_title(f"T = {slice_fit.expiry:.2f}y   rho {slice_fit.params.rho:.2f}, "
                       f"nu {slice_fit.params.nu:.2f}", color=ink, fontsize=10, loc="left")
        axis.set_xlabel("Strike / forward", color=muted_ink, fontsize=9)
        axis.grid(color=grid_color, lw=0.8)
        axis.tick_params(colors=muted_ink, labelsize=8)
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Implied vol (%)", color=muted_ink, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=ink)
    figure.suptitle("SABR fit by expiry (beta = 0.5, Obloj formula)", color=ink, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(file_name, dpi=150, facecolor=surface_color)
    print(f"\nChart saved: {file_name}")


def main() -> None:
    random_generator = np.random.default_rng(42)       # fixed seed -> same numbers every run
    market_template = make_market_template()

    def forward_of(expiry: float) -> float:
        return replace(market_template, expiry=expiry).forward()

    expiry_slices, true_params_by_expiry = build_synthetic_market(market_template, forward_of, random_generator)
    surface = fit_surface(expiry_slices, forward_of, beta=BETA)
    report_fit(surface, true_params_by_expiry)
    compare_hagan_and_obloj(expiry_slices)
    show_beta_sensitivity(expiry_slices)
    check_densities(surface)
    price_dummy_put(surface, market_template, forward_of)
    save_chart(surface)


if __name__ == "__main__":
    main()
