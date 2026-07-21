import os
import numpy as np
import pandas as pd
from explore_constituents import get_sp500_membership, members_on_date
from pit_factors import pit_fscore, pit_value_inputs, pit_fscore_quarterly_yoy

PRICE_DIR = "data/universe_prices"

_fscore_cache = {}
_fscore_q_cache = {}
_value_cache = {}


def cached_fscore(ticker, year):
    key = (ticker, year)
    if key not in _fscore_cache:
        try:
            _fscore_cache[key] = pit_fscore(ticker, f"{year}-06-30")
        except Exception:
            _fscore_cache[key] = None
    return _fscore_cache[key]


def cached_fscore_quarterly(ticker, year):
    """Quarterly-YoY quality analog of cached_fscore. Same as-of (June 30 of the
    rebalance year) and same (ticker, year) cache key, so the only difference vs
    cached_fscore is the annual F-score -> quarterly-YoY F-score input."""
    key = (ticker, year)
    if key not in _fscore_q_cache:
        try:
            _fscore_q_cache[key] = pit_fscore_quarterly_yoy(ticker, f"{year}-06-30")
        except Exception:
            _fscore_q_cache[key] = None
    return _fscore_q_cache[key]


def cached_value(ticker, year):
    key = (ticker, year)
    if key not in _value_cache:
        try:
            _value_cache[key] = pit_value_inputs(ticker, f"{year}-06-30")
        except Exception:
            _value_cache[key] = None
    return _value_cache[key]


def load_all_prices():
    all_series = {}
    for fname in os.listdir(PRICE_DIR):
        if not fname.endswith(".csv"):
            continue
        ticker = fname[:-4]
        df = pd.read_csv(f"{PRICE_DIR}/{fname}", index_col=0, parse_dates=True)
        if "Close" in df.columns and len(df) > 0:
            all_series[ticker] = df["Close"]
    return pd.DataFrame(all_series).sort_index()


def precompute(prices, year_start=2005, year_end=2027):
    print("Precomputing fundamentals factors...")
    tickers = sorted(prices.columns)
    years = list(range(year_start, year_end))
    for n, ticker in enumerate(tickers, 1):
        for year in years:
            cached_fscore(ticker, year)
            cached_fscore_quarterly(ticker, year)
            cached_value(ticker, year)
        if n % 100 == 0:
            print(f"  {n}/{len(tickers)} done")
    print("Done.\n")


def compute_trend_table(prices, monthly_dates):
    """Pct vs 200-day MA for every ticker at each month-end. Past data only."""
    print("Precomputing trend (pct vs 200d MA)...")
    ma200 = prices.rolling(200).mean()
    # Align daily MA to month-end dates
    px_m = prices.resample("ME").last()
    ma_m = ma200.resample("ME").last()
    trend = (px_m / ma_m) - 1
    print("Done.\n")
    return trend  # DataFrame: month-end x ticker


def zscore(series):
    s = series.astype(float)
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def run_backtest(prices, membership, trend_table, top_n=30, cost_per_trade=0.0005,
                 start="2017-01-01", end="2026-06-30",
                 w_mom=1.0, w_qual=0.0, w_val=0.0, w_trend=0.0, label="",
                 fscore_fn=cached_fscore):
    all_daily = {t: prices[t] for t in prices.columns}
    monthly = prices.resample("ME").last().loc[start:end]
    trend_m = trend_table.loc[start:end]
    dates = monthly.index

    portfolio_value = 10000.0
    history = []
    prev_holdings = set()

    for i in range(12, len(dates) - 1):
        today = dates[i]
        next_date = dates[i + 1]
        index_members = members_on_date(membership, today)
        year = today.year

        rows = []
        for ticker in index_members:
            if ticker not in monthly.columns:
                continue
            p_12mo = monthly[ticker].iloc[i - 12]
            p_1mo = monthly[ticker].iloc[i - 1]
            p_now = monthly[ticker].iloc[i]
            if pd.isna(p_12mo) or pd.isna(p_1mo) or pd.isna(p_now) or p_12mo <= 0 or p_now <= 0:
                continue
            mom = (p_1mo / p_12mo) - 1
            fs = fscore_fn(ticker, year)
            vi = cached_value(ticker, year)
            ey = None
            if vi is not None:
                mktcap = p_now * vi["shares"]
                if mktcap > 0:
                    ey = vi["net_income"] / mktcap
            tr = None
            if ticker in trend_m.columns and today in trend_m.index:
                tv = trend_m.loc[today, ticker]
                if pd.notna(tv):
                    tr = tv
            rows.append({"ticker": ticker, "momentum": mom,
                         "fscore": fs, "earn_yield": ey, "trend": tr})

        df = pd.DataFrame(rows)
        need = []
        if w_qual > 0:
            need.append("fscore")
        if w_val > 0:
            need.append("earn_yield")
        if w_trend > 0:
            need.append("trend")
        if need:
            df = df.dropna(subset=need)
        if len(df) < top_n:
            continue

        comp = w_mom * zscore(df["momentum"])
        if w_qual > 0:
            comp = comp + w_qual * zscore(df["fscore"])
        if w_val > 0:
            comp = comp + w_val * zscore(df["earn_yield"])
        if w_trend > 0:
            comp = comp + w_trend * zscore(df["trend"])
        df["composite"] = comp
        df = df.sort_values("composite", ascending=False)

        holdings = set(df["ticker"].head(top_n))
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


def metrics(returns_series):
    r = returns_series.dropna()
    n = len(r)
    total = (1 + r).prod() - 1
    annual = (1 + total) ** (12 / n) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = annual / vol if vol > 0 else 0
    equity = (1 + r).cumprod()
    max_dd = ((equity - equity.cummax()) / equity.cummax()).min()
    return {"total": total, "annual": annual, "vol": vol,
            "sharpe": sharpe, "max_dd": max_dd}


def spy_benchmark(start, end):
    import requests
    from config import EODHD_API_KEY
    url = f"https://eodhd.com/api/eod/SPY.US?api_token={EODHD_API_KEY}&fmt=json&period=d"
    data = requests.get(url, timeout=30).json()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[["adjusted_close"]]
    monthly = df["adjusted_close"].resample("ME").last().loc[start:end]
    return monthly.pct_change().dropna()


PERIODS = [
    ("2007-2012 (crisis+recovery)", "2007-01-01", "2012-12-31"),
    ("2013-2018 (mid bull)",        "2013-01-01", "2018-12-31"),
    ("2019-2026 (covid+AI bull)",   "2019-01-01", "2026-06-30"),
]

# Each config carries which F-score feeds the quality weight. All existing
# configs use the annual F-score (cached_fscore) exactly as before; the added
# "MOM+QUALITY (quarterly)" is identical to "MOM+QUALITY" except it uses the
# quarterly-YoY F-score, so the two are a clean head-to-head on quality input.
CONFIGS = [
    ("MOMENTUM",                1.0,  0.0,  0.0,  0.0, cached_fscore),
    ("MOM+QUALITY",             0.5,  0.5,  0.0,  0.0, cached_fscore),
    ("MOM+QUALITY (quarterly)", 0.5,  0.5,  0.0,  0.0, cached_fscore_quarterly),
    ("MOM+QUAL+TREND",          0.4,  0.4,  0.0,  0.2, cached_fscore),
    ("FULL (all four)",         0.3,  0.3,  0.2,  0.2, cached_fscore),
]

if __name__ == "__main__":
    print("Loading prices...")
    prices = load_all_prices()
    print(f"Loaded {prices.shape[1]} tickers")
    membership = get_sp500_membership()
    precompute(prices)
    monthly_dates = prices.resample("ME").last().index
    trend_table = compute_trend_table(prices, monthly_dates)

    for pname, start, end in PERIODS:
        print(f"\n{'#'*70}\nPERIOD: {pname}\n{'#'*70}")
        rows = []
        idx = None
        for label, wm, wq, wv, wt, fscore_fn in CONFIGS:
            res = run_backtest(prices, membership, trend_table,
                               start=start, end=end,
                               w_mom=wm, w_qual=wq, w_val=wv, w_trend=wt,
                               label=label, fscore_fn=fscore_fn)
            if len(res) == 0:
                continue
            m = metrics(res["strat_return"])
            m["label"] = label
            rows.append(m)
            idx = res.index

        if idx is None:
            continue

        spy = spy_benchmark(start, end).loc[idx[0]:idx[-1]]
        ms = metrics(spy)
        ms["label"] = "SPY"
        rows.append(ms)

        print(f"\n{'STRATEGY':<26}{'Annual':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>8}")
        print("-" * 59)
        for m in rows:
            print(f"{m['label']:<26}{m['annual']:>8.1%}{m['vol']:>7.1%}"
                  f"{m['sharpe']:>8.2f}{m['max_dd']:>8.1%}")