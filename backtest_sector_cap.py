"""Backtest the sector-diversification cap across the three regime periods.

Head-to-head on identical MOM+QUALITY composites (momentum 0.5 / quality 0.5,
top 30, winsorized z-scores as in the live scan) where the ONLY difference is
the post-ranking sector cap:

  - no cap  (pure composite rank — the validated concentrated selection)
  - cap 3   (max 3 names per sector)
  - cap 2   (max 2 names per sector — what the live Diversified track uses)

The cap mirrors dashboard.select_top exactly: walk the composite-sorted list
top-down, skip a name once its sector already has `cap` picks, stop at top_n.
It never touches the composite or its ordering.

LIMITATIONS — flagged honestly:
1. Sector data is CURRENT, not point-in-time: get_sector returns the ticker's
   sector as classified today (EODHD General.Sector). Sectors are mostly stable
   over a company's life but reclassifications happen (e.g. the 2018 GICS
   Communication Services reshuffle), so early-period assignments approximate.
2. EODHD tags ~85 long-delisted names (LEH, EMC, YHOO, TWX, ...) as sector
   'Other'. Lumping them into one capped pseudo-sector would starve the capped
   configs of delisted names — exactly the names that dominate 2007-2012
   membership — so SECTOR_OVERRIDES below maps them manually from public
   knowledge of each company. That mapping is a judgment call, not vendor data.
3. The few tickers still unresolvable after the overrides are EXEMPT from the
   cap (each treated as its own sector) rather than bucketed together.
"""
import pandas as pd
import numpy as np
from explore_constituents import get_sp500_membership, members_on_date
from backtest_composite import (load_all_prices, asof_fscore, metrics,
                                spy_benchmark, drop_ticker_cache)
from data_layer import get_sector

TOP_N = 30
W_MOM, W_QUAL = 0.5, 0.5
COST_PER_TRADE = 0.0005

PERIODS = [
    ("2007-2012 (crisis+recovery)", "2007-01-01", "2012-12-31"),
    ("2013-2018 (mid bull)",        "2013-01-01", "2018-12-31"),
    ("2019-2026 (covid+AI bull)",   "2019-01-01", "2026-06-30"),
]

# label -> sector cap (None = no cap)
CONFIGS = [
    ("MOM+QUALITY (no cap)", None),
    ("MOM+QUALITY (cap 3)",  3),
    ("MOM+QUALITY (cap 2)",  2),
]

# Manual sectors for delisted names EODHD tags 'Other'/'NA' (see LIMITATIONS).
# Uses the same Morningstar-style sector names EODHD returns for live tickers.
SECTOR_OVERRIDES = {
    "AET": "Healthcare", "ANDV": "Energy", "ANR": "Energy",
    "APOL": "Consumer Defensive", "ARG": "Basic Materials",
    "AVP": "Consumer Defensive", "BCR": "Healthcare", "BIVV": "Healthcare",
    "BMC": "Technology", "BRCM": "Technology", "BXLT": "Healthcare",
    "CAM_OLD": "Energy", "CA_OLD": "Technology", "CBE": "Industrials",
    "CCE": "Consumer Defensive", "CFN": "Healthcare",
    "CMCSK": "Communication Services", "COL": "Industrials",
    "COV": "Healthcare", "CPGX": "Energy", "CSRA": "Technology",
    "CVC": "Communication Services", "CVH": "Healthcare",
    "DELL_OLD": "Technology", "DF": "Consumer Defensive",
    "DTV": "Communication Services", "EMC": "Technology",
    "ESRX": "Healthcare", "EVHC": "Healthcare", "FDO": "Consumer Defensive",
    "GAS": "Utilities", "GG": "Basic Materials", "GGP": "Real Estate",
    "GMCR": "Consumer Defensive", "GR": "Industrials", "HAR": "Technology",
    "HCBK": "Financial Services", "HNZ": "Consumer Defensive",
    "HOT": "Consumer Cyclical", "HSH": "Consumer Defensive",
    "HSP": "Healthcare", "JOY": "Industrials", "LEH": "Financial Services",
    "LLTC": "Technology", "LO": "Consumer Defensive",
    "LVLT": "Communication Services", "LXK": "Technology",
    "MHS": "Healthcare", "MJN": "Consumer Defensive", "MOLX": "Technology",
    "NFX": "Energy", "NVLS": "Technology", "NYX_OLD": "Financial Services",
    "PCL_OLD": "Real Estate", "PCP": "Industrials",
    "PETM": "Consumer Cyclical", "PGN": "Utilities", "POM_OLD": "Utilities",
    "QCP": "Real Estate", "RAI": "Consumer Defensive",
    "RAL_OLD": "Consumer Defensive", "RDC": "Energy", "RHT": "Technology",
    "SCG": "Utilities", "SHLD_OLD": "Consumer Cyclical",
    "SIAL": "Basic Materials", "SNDK_OLD": "Technology",
    "SNI": "Communication Services", "SPLS_OLD": "Consumer Cyclical",
    "STJ": "Healthcare", "SVU": "Consumer Defensive",
    "SWY": "Consumer Defensive", "TEG": "Utilities", "TE_OLD1": "Utilities",
    "TFCF": "Communication Services", "TFCFA": "Communication Services",
    "TIE": "Basic Materials", "TSS": "Technology",
    "TWC": "Communication Services", "TWX": "Communication Services",
    "VIAB": "Communication Services", "VSM": "Basic Materials",
    "WCG": "Healthcare", "WFM": "Consumer Defensive",
    "WIN": "Communication Services", "YHOO": "Communication Services",
}

# EODHD returns GICS-style names for a couple of tickers; fold them into the
# Morningstar-style buckets used everywhere else.
SECTOR_NORMALIZE = {
    "Financials": "Financial Services",
    "Consumer Discretionary": "Consumer Cyclical",
}

# Sentinel values meaning "sector genuinely unknown" — cap-exempt.
UNRESOLVED = {"Other", "NA", "nan", "Unknown", ""}


def build_sector_map(tickers):
    """ticker -> sector, cache-first via get_sector, with delisted overrides
    and name normalization. Unresolved tickers map to a unique per-ticker
    bucket so the cap never groups them."""
    sectors = {}
    unresolved = []
    for t in tickers:
        sec = SECTOR_OVERRIDES.get(t.upper())
        if sec is None:
            sec = get_sector(t)
            sec = SECTOR_NORMALIZE.get(sec, sec)
        if not isinstance(sec, str) or sec in UNRESOLVED:
            unresolved.append(t)
            sec = f"__unresolved_{t}"
        sectors[t] = sec
    return sectors, unresolved


def winsorize(s, lower=0.01, upper=0.99):
    """Same 1st/99th-percentile capping the live scan applies before z-scoring."""
    valid = s.dropna()
    if len(valid) < 2:
        return s
    lo, hi = valid.quantile(lower), valid.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def zscore(series):
    s = series.astype(float)
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def select_with_cap(df, top_n, cap, sectors):
    """dashboard.select_top logic on a composite-sorted frame."""
    if cap is None:
        return list(df["ticker"].head(top_n))
    counts, keep = {}, []
    for _, row in df.iterrows():
        sec = sectors.get(row["ticker"], f"__unresolved_{row['ticker']}")
        if counts.get(sec, 0) >= cap:
            continue
        counts[sec] = counts.get(sec, 0) + 1
        keep.append(row["ticker"])
        if len(keep) >= top_n:
            break
    return keep


def run_backtest_cap(prices, membership, sectors, cap, start, end,
                     top_n=TOP_N, cost_per_trade=COST_PER_TRADE):
    """MOM+QUALITY monthly backtest identical to backtest_composite.run_backtest
    (12-1 momentum, annual PIT F-score, same delisting handling and costs) but
    with winsorized z-scores and an optional post-ranking sector cap."""
    all_daily = {t: prices[t] for t in prices.columns}
    monthly = prices.resample("ME").last().loc[start:end]
    dates = monthly.index

    portfolio_value = 10000.0
    history = []
    prev_holdings = set()

    for i in range(12, len(dates) - 1):
        today = dates[i]
        next_date = dates[i + 1]
        index_members = members_on_date(membership, today)

        rows = []
        for ticker in index_members:
            if ticker not in monthly.columns:
                continue
            p_12mo = monthly[ticker].iloc[i - 12]
            p_1mo = monthly[ticker].iloc[i - 1]
            p_now = monthly[ticker].iloc[i]
            if pd.isna(p_12mo) or pd.isna(p_1mo) or pd.isna(p_now) \
                    or p_12mo <= 0 or p_now <= 0:
                continue
            mom = (p_1mo / p_12mo) - 1
            # As-of the date being simulated — never a fixed date in its future.
            fs = asof_fscore(ticker, today)
            rows.append({"ticker": ticker, "momentum": mom, "fscore": fs})

        df = pd.DataFrame(rows).dropna(subset=["fscore"])
        if len(df) < top_n:
            continue

        df["composite"] = (W_MOM * zscore(winsorize(df["momentum"]))
                           + W_QUAL * zscore(winsorize(df["fscore"])))
        df = df.sort_values("composite", ascending=False)

        holdings = set(select_with_cap(df, top_n, cap, sectors))
        trades = holdings.symmetric_difference(prev_holdings)
        cost = len(trades) * cost_per_trade * (portfolio_value / top_n)
        portfolio_value -= cost
        prev_holdings = holdings

        rets = []
        for ticker in holdings:
            p_now = monthly[ticker].iloc[i]
            p_next = monthly[ticker].iloc[i + 1]
            if pd.notna(p_now) and pd.notna(p_next) and p_now > 0:
                rets.append((p_next / p_now) - 1)
            elif pd.notna(p_now) and pd.isna(p_next):
                daily = all_daily.get(ticker)
                if daily is not None and len(daily.dropna()) > 0:
                    clean = daily.dropna()
                    actual = (clean.iloc[-1] / p_now) - 1
                    if len(clean) >= 126 and (clean.iloc[-1] / clean.iloc[-126] - 1) < -0.50:
                        rets.append((1 + actual) * 0.70 - 1)
                    else:
                        rets.append(actual)
                else:
                    rets.append(-0.30)
        strat_return = np.mean(rets) if rets else 0.0
        portfolio_value *= (1 + strat_return)
        history.append({"date": next_date, "strategy": portfolio_value,
                        "strat_return": strat_return})

    return pd.DataFrame(history).set_index("date")


def precompute_fscores(prices, probe="2015-06-30"):
    """Warm each ticker's as-of F-score timeline, then release its raw payload."""
    print("Precomputing annual F-scores...")
    tickers = sorted(prices.columns)
    for n, ticker in enumerate(tickers, 1):
        asof_fscore(ticker, probe)
        drop_ticker_cache(ticker)
        if n % 100 == 0:
            print(f"  {n}/{len(tickers)} done")
    print("Done.\n")


if __name__ == "__main__":
    print("Loading prices...")
    prices = load_all_prices()
    print(f"Loaded {prices.shape[1]} tickers")
    membership = get_sp500_membership()
    precompute_fscores(prices)

    print("Loading sectors (cache-first, with delisted overrides)...")
    sectors, unresolved = build_sector_map(prices.columns)
    print(f"{len(sectors)} tickers, {len(unresolved)} unresolved (cap-exempt): "
          f"{', '.join(unresolved)}\n")

    for pname, start, end in PERIODS:
        print(f"\n{'#'*70}\nPERIOD: {pname}\n{'#'*70}")
        rows = []
        idx = None
        for label, cap in CONFIGS:
            res = run_backtest_cap(prices, membership, sectors, cap, start, end)
            if len(res) == 0:
                print(f"  {label}: no tradable months")
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

        print(f"\n{'STRATEGY':<24}{'Annual':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>8}")
        print("-" * 57)
        for m in rows:
            print(f"{m['label']:<24}{m['annual']:>8.1%}{m['vol']:>7.1%}"
                  f"{m['sharpe']:>8.2f}{m['max_dd']:>8.1%}")
