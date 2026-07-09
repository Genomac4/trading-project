import pandas as pd
from factors import piotroski_fscore, momentum_12_1, trend_signal
from data_layer import get_valuation


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