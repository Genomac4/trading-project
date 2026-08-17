import pandas as pd
from factors import piotroski_fscore, momentum_12_1, trend_signal
from data_layer import get_valuation


# NOTE ON THESE WEIGHTS — READ BEFORE TRUSTING THEM.
# The two weight sets below are NOT the output of a weight optimizer and were
# NOT fit to maximize any in-sample metric. They are lifted directly from
# multi-period walk-forward backtests (see backtest_periods.py /
# backtest_composite.py):
#   - TRENDING (SPY above its 200d MA): a momentum+quality tilt earned Sharpe
#     0.96 vs SPY's 0.88 over 2019-2026.
#   - CHOPPY (SPY below its 200d MA): NOT backtest-derived — see below.
# Because they are picked from how factors behaved across distinct historical
# regimes rather than tuned to a single sample, treat them as a robust prior,
# not a fitted optimum. The overfitting risk is real but deliberately bounded —
# there are only two coarse regimes and no per-parameter search.
#
# CHOPPY IS A JUDGMENT CALL, NOT A BACKTEST RESULT (audited 2026-08-16).
# This comment used to claim a trend/quality tilt had the best Sharpe over the
# choppy 2013-2018 stretch. That is false. In that period FULL (all four
# factors, 0.3/0.3/0.2/0.2) beat MOM+QUAL+TREND on Sharpe both before and after
# the backtest look-ahead fix: 0.71 vs 0.59 pre-fix, 0.68 vs 0.54 post-fix.
# Two separate gaps, both still open:
#   1. The trend/quality config that WAS tested is 0.4/0.4/0.0/0.2. The live
#      CHOPPY vector below is 0.3/0.4/0.0/0.3 — a different vector that has
#      never been run through a backtest at all.
#   2. Even the tested 0.4/0.4/0.0/0.2 was not the best config in 2013-2018;
#      FULL was.
# Consequence: if SPY closes below its 200d MA, unvalidated weights go live
# silently. It has never fired — all 11 scan snapshots to date show TRENDING —
# so this is a latent risk, not a current one. Do not describe CHOPPY as
# backtest-backed until either the live vector is actually tested or the
# weights are changed to something that was.
REGIME_WEIGHTS = {
    "TRENDING": {"momentum": 0.5, "quality": 0.5, "value": 0.0, "trend": 0.0},
    # UN-BACKTESTED — see the note above before trusting or changing these.
    "CHOPPY":   {"momentum": 0.3, "quality": 0.4, "value": 0.0, "trend": 0.3},
}


def get_regime_weights(spy_series):
    """
    Pick the factor-weight set for the current market regime, defined by SPY
    relative to its own 200-day moving average.

    Parameters
    ----------
    spy_series : pd.Series
        SPY closing prices indexed by date (ascending). Only the tail is used.

    Returns
    -------
    dict with:
        regime       : "TRENDING" or "CHOPPY"
        trending     : bool (SPY above its 200d MA)
        pct_vs_200d  : float, SPY's % distance from its 200d MA
        weights      : dict of {momentum, quality, value, trend} weights

    The TRENDING weights come from the multi-period backtest documented above,
    not from optimization. The CHOPPY weights are un-backtested — read the note
    at REGIME_WEIGHTS before relying on them.
    """
    spy_series = spy_series.dropna()
    ma200 = spy_series.rolling(200).mean().iloc[-1]
    spot = spy_series.iloc[-1]
    trending = bool(spot > ma200)
    regime = "TRENDING" if trending else "CHOPPY"
    return {
        "regime": regime,
        "trending": trending,
        "pct_vs_200d": float((spot / ma200) - 1),
        "weights": dict(REGIME_WEIGHTS[regime]),
    }


def screen(tickers):
    """
    Run all factors across a list of tickers and assemble
    the results into one ranked table.
    """
    rows = []

    for ticker in tickers:
        ticker = ticker.upper()
        print(f"\n{'='*40}\nProcessing {ticker}...")

        # Run each factor, defensively — if one fails, skip the stock
        try:
            fscore = piotroski_fscore(ticker)
            mom = momentum_12_1(ticker)
            trend = trend_signal(ticker)
            val = get_valuation(ticker)
        except Exception as e:
            print(f"[{ticker}] Error: {e} — skipping")
            continue

        # Pull the key numbers out of each factor's result
        row = {
            "Ticker": ticker,
            "F_Score": fscore["score"] if fscore else None,
            "F_Available": fscore["available"] if fscore else None,
            "Momentum_12_1": mom["momentum_12_1"] if mom else None,
            "Pct_vs_200d": trend["pct_vs_200d"] if trend else None,
            "Golden_Cross": trend["golden_cross"] if trend else None,
            "EV_EBITDA": val.get("ev_ebitda") if val else None,
            "Forward_PE": val.get("forward_pe") if val else None,
            "Analyst_Target": val.get("analyst_target") if val else None,
            "Analyst_Rating": val.get("analyst_rating") if val else None,
        }
        rows.append(row)

    # Build the results table
    df = pd.DataFrame(rows)
    return df

def add_composite(df, weights=None):
    """
    Add a composite score that combines factors into one ranking.
    Each factor is z-scored (normalized), sign-adjusted so higher
    is always better, weighted, and summed.
    """
    if weights is None:
        weights = {
            "F_Score": 0.30,        # quality
            "Momentum_12_1": 0.25,  # trend
            "Pct_vs_200d": 0.15,    # trend confirmation
            "EV_EBITDA": 0.20,      # value (cheaper = better)
            "Forward_PE": 0.10,     # value (cheaper = better)
        }

    lower_is_better = {"EV_EBITDA", "Forward_PE"}

    df = df.copy()
    composite = pd.Series(0.0, index=df.index)

    for factor, weight in weights.items():
        col = df[factor].astype(float)
        mean = col.mean()
        std = col.std()
        if std == 0 or pd.isna(std):
            z = pd.Series(0.0, index=df.index)
        else:
            z = (col - mean) / std
        if factor in lower_is_better:
            z = -z
        z = z.fillna(0)
        composite += weight * z

    df["Composite"] = composite
    df = df.sort_values("Composite", ascending=False)
    return df
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    universe = ["CEG", "VST", "NRG", "TLN", "COST", "AAPL"]

    results = screen(universe)
    results = add_composite(results)

    print("\n\n========== SCREENER RESULTS (ranked) ==========")
    print(results[["Ticker", "Composite", "F_Score", "Momentum_12_1",
                   "Pct_vs_200d", "EV_EBITDA", "Forward_PE"]])