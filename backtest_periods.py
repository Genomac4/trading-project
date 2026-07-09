import pandas as pd
import numpy as np
from explore_constituents import get_sp500_membership
from backtest_composite import (load_all_prices, precompute, run_backtest,
                                metrics, spy_benchmark, cached_fscore, cached_value)

# Extend factor precompute back for the older windows
def precompute_extended(prices, year_start=2005, year_end=2027):
    print("Precomputing factors (extended years)...")
    tickers = sorted(prices.columns)
    years = list(range(year_start, year_end))
    for n, ticker in enumerate(tickers, 1):
        for year in years:
            cached_fscore(ticker, year)
            cached_value(ticker, year)
        if n % 100 == 0:
            print(f"  {n}/{len(tickers)} done")
    print("Done.\n")


PERIODS = [
    ("2007-2012 (crisis+recovery)", "2007-01-01", "2012-12-31"),
    ("2013-2018 (mid bull)",        "2013-01-01", "2018-12-31"),
    ("2019-2026 (covid+AI bull)",   "2019-01-01", "2026-06-30"),
]

CONFIGS = [
    ("MOMENTUM",       1.0, 0.0, 0.0),
    ("MOM+QUALITY",    0.5, 0.5, 0.0),
    ("MOM+QUAL+VALUE", 1/3, 1/3, 1/3),
]

if __name__ == "__main__":
    print("Loading prices...")
    prices = load_all_prices()
    print(f"Loaded {prices.shape[1]} tickers")
    membership = get_sp500_membership()
    precompute_extended(prices)

    for pname, start, end in PERIODS:
        print(f"\n{'#'*70}\nPERIOD: {pname}\n{'#'*70}")
        rows = []
        idx = None
        for label, wm, wq, wv in CONFIGS:
            res = run_backtest(prices, membership, start=start, end=end,
                               w_mom=wm, w_qual=wq, w_val=wv, label=label)
            if len(res) == 0:
                print(f"  {label}: no tradable months (membership too thin)")
                continue
            m = metrics(res["strat_return"])
            m["label"] = label
            rows.append(m)
            idx = res.index

        if idx is None:
            print("  Skipping period — no results.")
            continue

        spy = spy_benchmark(start, end).loc[idx[0]:idx[-1]]
        ms = metrics(spy)
        ms["label"] = "SPY"
        rows.append(ms)

        print(f"\n{'STRATEGY':<18}{'Annual':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>8}")
        print("-" * 51)
        for m in rows:
            print(f"{m['label']:<18}{m['annual']:>8.1%}{m['vol']:>7.1%}"
                  f"{m['sharpe']:>8.2f}{m['max_dd']:>8.1%}")