import pandas as pd
from data_layer import get_prices

# Diversified multi-sector universe for backtesting.
# ~30 large-caps across every major sector — debuggable size.
UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    # Financials
    "JPM", "BAC", "GS", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK",
    # Consumer
    "AMZN", "WMT", "COST", "PG", "KO",
    # Energy / Power
    "XOM", "CVX", "CEG", "NEE",
    # Industrials
    "CAT", "BA", "HON", "UPS",
    # Communications / Other
    "DIS", "T",
]


def pull_universe(tickers):
    """Pull and cache price history for every ticker in the universe."""
    data = {}
    failed = []
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...")
        try:
            prices = get_prices(ticker)
            if prices is not None and len(prices) > 0:
                data[ticker] = prices["Close"]
            else:
                failed.append(ticker)
        except Exception as e:
            print(f"  Failed: {e}")
            failed.append(ticker)

    print(f"\nPulled {len(data)} stocks successfully.")
    if failed:
        print(f"Failed: {failed}")
    return data
import numpy as np


def run_backtest(price_data, top_n=5, cost_per_trade=0.0005):
    """
    Momentum backtest, monthly rebalance.
    - price_data: dict of {ticker: Close price Series}
    - top_n: how many stocks to hold each month
    - cost_per_trade: transaction cost as a fraction (0.0005 = 0.05%)
    Returns a DataFrame of monthly portfolio values vs equal-weight benchmark.
    """
    # Combine all price series into one aligned DataFrame (dates x tickers)
    prices = pd.DataFrame(price_data)
    prices = prices.sort_index()

    # Resample to month-end prices — we rebalance monthly
    monthly = prices.resample("ME").last()

    # We need 12 months of history to compute momentum, so start at month 12
    dates = monthly.index
    start_idx = 12

    portfolio_value = 10000.0
    benchmark_value = 10000.0
    history = []

    prev_holdings = set()

    for i in range(start_idx, len(dates) - 1):
        today = dates[i]
        next_date = dates[i + 1]

        # --- RANK by momentum, using ONLY past data (no look-ahead) ---
        # 12-1 momentum: price 1 month ago / price 12 months ago - 1
        mom = {}
        for ticker in monthly.columns:
            p_12mo = monthly[ticker].iloc[i - 12]
            p_1mo = monthly[ticker].iloc[i - 1]
            # Only rank stocks that have valid data for both points
            if pd.notna(p_12mo) and pd.notna(p_1mo) and p_12mo > 0:
                mom[ticker] = (p_1mo / p_12mo) - 1

        if len(mom) < top_n:
            continue  # not enough eligible stocks yet

        # Pick the top N by momentum
        ranked = sorted(mom, key=mom.get, reverse=True)
        holdings = set(ranked[:top_n])

        # --- Transaction costs: pay for stocks we buy or sell this rebalance ---
        trades = holdings.symmetric_difference(prev_holdings)
        cost = len(trades) * cost_per_trade * (portfolio_value / top_n)
        portfolio_value -= cost
        prev_holdings = holdings

        # --- Compute this month's return: hold from today to next_date ---
        rets = []
        for ticker in holdings:
            p_now = monthly[ticker].iloc[i]
            p_next = monthly[ticker].iloc[i + 1]
            if pd.notna(p_now) and pd.notna(p_next) and p_now > 0:
                rets.append((p_next / p_now) - 1)
        strat_return = np.mean(rets) if rets else 0.0
        portfolio_value *= (1 + strat_return)

        # --- Benchmark: equal-weight ALL stocks (buy-and-hold everything) ---
        bench_rets = []
        for ticker in monthly.columns:
            p_now = monthly[ticker].iloc[i]
            p_next = monthly[ticker].iloc[i + 1]
            if pd.notna(p_now) and pd.notna(p_next) and p_now > 0:
                bench_rets.append((p_next / p_now) - 1)
        bench_return = np.mean(bench_rets) if bench_rets else 0.0
        benchmark_value *= (1 + bench_return)

        history.append({
            "date": next_date,
            "strategy": portfolio_value,
            "benchmark": benchmark_value,
            "strat_return": strat_return,
            "bench_return": bench_return,
            "holdings": ", ".join(ranked[:top_n]),
        })

    return pd.DataFrame(history).set_index("date")

if __name__ == "__main__":
    prices = pull_universe(UNIVERSE)
    results = run_backtest(prices, top_n=5, cost_per_trade=0.0005)

    print("\n\n========== BACKTEST RESULTS ==========")
    print(f"Period: {results.index[0].date()} to {results.index[-1].date()}")
    print(f"Months tested: {len(results)}")
    print(f"\nFinal strategy value:   ${results['strategy'].iloc[-1]:,.0f}")
    print(f"Final benchmark value:  ${results['benchmark'].iloc[-1]:,.0f}")
    print(f"\nStrategy total return:  {(results['strategy'].iloc[-1]/10000 - 1):.1%}")
    print(f"Benchmark total return: {(results['benchmark'].iloc[-1]/10000 - 1):.1%}")