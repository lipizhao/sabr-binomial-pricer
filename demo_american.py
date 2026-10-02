"""Price one dummy American put with the binomial tree and print its Greeks.

Run:  python demo_american.py

What it shows:
  1. The American put's price and Greeks.
  2. The same option priced as European, and the early-exercise premium
     (American minus European = value of being allowed to exercise early).
  3. How the price converges as the number of tree steps grows, with and
     without smoothing (see the "Why smoothing?" note in binomial_tree.py).
"""
from dataclasses import replace

from binomial_tree import Dividend, OptionSpec, flat_curve, greeks, interp_curve, price

TREE_STEPS = 1000                                         # steps used for the headline numbers
CONVERGENCE_STEPS = (50, 51, 100, 101, 250, 500, 1000, 2000)  # odd/even pairs expose the zig-zag


def make_dummy_american_put() -> OptionSpec:
    """A made-up 9-month American put on a $100 stock paying three $0.75 dividends."""
    return OptionSpec(
        spot=100.0,
        strike=105.0,
        expiry=0.75,                      # years
        sigma=0.28,                       # Black vol; in production this comes from the SABR surface
        is_call=False,
        american=True,
        rate_curve=interp_curve([0.25, 0.5, 1.0], [0.0430, 0.0420, 0.0400]),  # zero rates by maturity
        borrow_curve=flat_curve(0.0050),  # stock borrow cost
        dividends=[Dividend(time=0.20, amount=0.75),
                   Dividend(time=0.45, amount=0.75),
                   Dividend(time=0.70, amount=0.75)],
    )


def print_price_and_greeks(american_put: OptionSpec) -> None:
    greek_values = greeks(american_put, steps=TREE_STEPS)
    european_put = replace(american_put, american=False)   # same option, early exercise disallowed
    european_price = price(european_put, TREE_STEPS)
    early_exercise_premium = greek_values["price"] - european_price

    print("Dummy American put: S=100, K=105, T=0.75y, vol=28%, 3 x $0.75 dividends")
    print(f"  forward to expiry      {american_put.forward():10.4f}")
    for greek_name, greek_value in greek_values.items():
        print(f"  {greek_name:<22} {greek_value:10.4f}")
    print(f"  European (same tree)   {european_price:10.4f}")
    print(f"  early-exercise premium {early_exercise_premium:10.4f}")


def print_convergence_table(american_put: OptionSpec) -> None:
    """Without smoothing the price zig-zags between odd and even step counts;
    with smoothing it settles down quickly."""
    print("\nConvergence (American price):")
    print("  steps   no smoothing   smoothed")
    for step_count in CONVERGENCE_STEPS:
        raw_price = price(american_put, step_count, smoothing=False)
        smoothed_price = price(american_put, step_count, smoothing=True)
        print(f"  {step_count:5d}   {raw_price:12.5f}   {smoothed_price:10.5f}")


def main() -> None:
    american_put = make_dummy_american_put()
    print_price_and_greeks(american_put)
    print_convergence_table(american_put)


if __name__ == "__main__":
    main()
