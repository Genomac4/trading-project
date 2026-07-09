import requests
import pandas as pd
from config import EODHD_API_KEY


def get_eodhd_prices(ticker):
    """Pull daily adjusted close history from EODHD for one ticker."""
    url = (f"https://eodhd.com/api/eod/{ticker}.US"
           f"?api_token={EODHD_API_KEY}&fmt=json&period=d")
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[{ticker}] failed: status {resp.status_code}")
        return None
    data = resp.json()
    if not data:
        print(f"[{ticker}] no data returned")
        return None
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


# Test on a LIVE stock and a DEAD/left stock
for ticker in ["AAPL", "TWX", "SWY", "JOY"]:
    print(f"\n=== {ticker} ===")
    df = get_eodhd_prices(ticker)
    if df is not None:
        print(f"  {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Latest close: {df['adjusted_close'].iloc[-1]}")