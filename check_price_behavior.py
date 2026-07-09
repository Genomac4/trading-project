import pandas as pd
import os

PRICE_DIR = "data/universe_prices"

# Known cases to compare:
#   Acquisitions (got paid a premium) — expect stable/up into exit
#   Bankruptcies (failed)             — expect a crash into exit
cases = {
    "AET":  "Aetna — ACQUIRED by CVS (premium)",
    "ALXN": "Alexion — ACQUIRED by AstraZeneca (premium)",
    "ATVI": "Activision — ACQUIRED by Microsoft (premium)",
    "ANR":  "Alpha Natural Resources — BANKRUPTCY",
    "BBBY_old": "Bed Bath & Beyond — BANKRUPTCY",
    "AVP":  "Avon — distressed sale",
}

for ticker, label in cases.items():
    path = f"{PRICE_DIR}/{ticker}.csv"
    if not os.path.exists(path):
        print(f"\n{ticker}: no price file ({label})")
        continue

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    close = df["Close"].dropna()
    if len(close) < 130:
        print(f"\n{ticker}: too little data ({label})")
        continue

    last_price = close.iloc[-1]
    price_1mo_before = close.iloc[-21]    # ~1 month before last trade
    price_6mo_before = close.iloc[-126]   # ~6 months before last trade

    ret_1mo = (last_price / price_1mo_before) - 1
    ret_6mo = (last_price / price_6mo_before) - 1

    print(f"\n=== {ticker}: {label} ===")
    print(f"  Last trade date:  {close.index[-1].date()}")
    print(f"  Last price:       ${last_price:.2f}")
    print(f"  Return last 1mo:  {ret_1mo:+.1%}")
    print(f"  Return last 6mo:  {ret_6mo:+.1%}")