import os
import numpy as np
import pandas as pd
from explore_constituents import get_sp500_membership, members_on_date

PRICE_DIR = "data/universe_prices"

def delisting_return(ticker, monthly, i, all_daily):
    """
    Compute the return for a stock that had a price at month i but
    not at month i+1 (it left / delisted during the hold).
    Cites: Shumway (1997), Shumway & Warther (1999) — -30% convention.
    """
    p_now = monthly[ticker].iloc[i]
    if pd.isna(p_now) or p_now <= 0:
        return 0.0

    daily = all_daily.get(ticker)
    if daily is None or len(daily) == 0:
        return -0.30

    last_price = daily.dropna().iloc[-1]
    actual_return = (last_price / p_now) - 1

    crash = False
    clean = daily.dropna()
    if len(clean) >= 126:
        ret_6mo = (clean.iloc[-1] / clean.iloc[-126]) - 1
        if ret_6mo < -0.50:
            crash = True

    if crash:
        return (1 + actual_return) * (1 - 0.30) - 1
    else:
        return actual_return
def load_all_prices():
    """Load every cached universe price file into one big DataFrame."""
    all_series = {}
    for fname in os.listdir(PRICE_DIR):
        if not fname.endswith(".csv"):
            continue
        ticker = fname[:-4]
        df = pd.read_csv(f"{PRICE_DIR}/{fname}", index_col=0, parse_dates=True)
        if "Close" in df.columns and len(df) > 0:
            all_series[ticker] = df["Close"]
    prices = pd.DataFrame(all_series).sort_index()
    return prices


def run_real_backtest(top_n=30, cost_per_trade=0.0005,
                      start="2017-01-01", end="2026-06-30"):
    print("Loading prices...")
    prices = load_all_prices()
    all_daily = {t: prices[t] for t in prices.columns}
    print(f"Loaded {prices.shape[1]} tickers, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")

    print("Loading S&P 500 membership history...")
    membership = get_sp500_membership()

    # Month-end prices
    monthly = prices.resample("ME").last()
    monthly = monthly.loc[start:end]
    dates = monthly.index

    portfolio_value = 10000.0
    history = []
    prev_holdings = set()

    for i in range(12, len(dates) - 1):
        today = dates[i]
        next_date = dates[i + 1]

        # Who was REALLY in the S&P 500 on this date?
        index_members = members_on_date(membership, today)

        # Rank eligible members by 12-1 momentum (past data only)
        mom = {}
        for ticker in index_members:
            if ticker not in monthly.columns:
                continue
            p_12mo = monthly[ticker].iloc[i - 12]
            p_1mo = monthly[ticker].iloc[i - 1]
            if pd.notna(p_12mo) and pd.notna(p_1mo) and p_12mo > 0:
                mom[ticker] = (p_1mo / p_12mo) - 1

        if len(mom) < top_n:
            continue

        ranked = sorted(mom, key=mom.get, reverse=True)
        holdings = set(ranked[:top_n])

        # Transaction costs on changes
        trades = holdings.symmetric_difference(prev_holdings)
        cost = len(trades) * cost_per_trade * (portfolio_value / top_n)
        portfolio_value -= cost
        prev_holdings = holdings

        # This month's return (hold today -> next_date)
        rets = []
        for ticker in holdings:
            p_now = monthly[ticker].iloc[i]
            p_next = monthly[ticker].iloc[i + 1]
            if pd.notna(p_now) and pd.notna(p_next) and p_now > 0:
                rets.append((p_next / p_now) - 1)
            elif pd.notna(p_now) and pd.isna(p_next):
                rets.append(delisting_return(ticker, monthly, i, all_daily))
        strat_return = np.mean(rets) if rets else 0.0
        portfolio_value *= (1 + strat_return)

        history.append({
            "date": next_date,
            "strategy": portfolio_value,
            "strat_return": strat_return,
            "n_eligible": len(mom),
        })

    return pd.DataFrame(history).set_index("date")


def performance_metrics(returns_series, label):
    r = returns_series.dropna()
    n_months = len(r)
    total_return = (1 + r).prod() - 1
    annual_return = (1 + total_return) ** (12 / n_months) - 1
    annual_vol = r.std() * np.sqrt(12)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    equity = (1 + r).cumprod()
    max_dd = ((equity - equity.cummax()) / equity.cummax()).min()
    print(f"\n--- {label} ---")
    print(f"  Total return:      {total_return:.1%}")
    print(f"  Annualized return: {annual_return:.1%}")
    print(f"  Annualized vol:    {annual_vol:.1%}")
    print(f"  Sharpe ratio:      {sharpe:.2f}")
    print(f"  Max drawdown:      {max_dd:.1%}")
def spy_benchmark(start, end):
    """Buy-and-hold SPY over the same window, as monthly returns."""
    import requests
    from config import EODHD_API_KEY

    url = (f"https://eodhd.com/api/eod/SPY.US"
           f"?api_token={EODHD_API_KEY}&fmt=json&period=d")
    data = requests.get(url, timeout=30).json()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[["adjusted_close"]].rename(
        columns={"adjusted_close": "Close"})

    monthly = df["Close"].resample("ME").last().loc[start:end]
    returns = monthly.pct_change().dropna()
    return returns

if __name__ == "__main__":
    start, end = "2017-01-01", "2026-06-30"

    # Strategy
    results = run_real_backtest(top_n=30, start=start, end=end)
    print(f"\nMonths tested: {len(results)}")

    # Align SPY to the same dates the strategy actually traded
    spy_rets = spy_benchmark(start, end)
    spy_rets = spy_rets.loc[results.index[0]:results.index[-1]]

    # Build SPY equity curve from $10,000
    spy_value = 10000 * (1 + spy_rets).cumprod()

    print(f"\nFinal strategy value: ${results['strategy'].iloc[-1]:,.0f}")
    print(f"Final SPY value:      ${spy_value.iloc[-1]:,.0f}")

    performance_metrics(results["strat_return"], "MOMENTUM top-30 (survivorship-free)")
    performance_metrics(spy_rets, "SPY (buy & hold)")

    # The verdict
    strat_total = results["strategy"].iloc[-1] / 10000 - 1
    spy_total = spy_value.iloc[-1] / 10000 - 1
    print(f"\n{'='*45}")
    print(f"VERDICT")
    print(f"  Strategy total return: {strat_total:.1%}")
    print(f"  SPY total return:      {spy_total:.1%}")
    print(f"  Excess (alpha):        {strat_total - spy_total:+.1%}")
    print(f"{'='*45}")