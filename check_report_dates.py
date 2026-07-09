import requests
import pandas as pd
from config import EODHD_API_KEY

ticker = "CEG"
url = f"https://eodhd.com/api/fundamentals/{ticker}.US?api_token={EODHD_API_KEY}&fmt=json"
data = requests.get(url).json()

# Earnings history holds the reportDate (8-K filing date) per period
earnings = data.get("Earnings", {}).get("History", {})

print(f"{ticker} — earnings periods with report dates:\n")
rows = []
for period, info in earnings.items():
    rows.append({
        "fiscal_period": period,
        "reportDate": info.get("reportDate"),
        "epsActual": info.get("epsActual"),
    })

df = pd.DataFrame(rows)
df["fiscal_period"] = pd.to_datetime(df["fiscal_period"], errors="coerce")
df["reportDate"] = pd.to_datetime(df["reportDate"], errors="coerce")
df = df.sort_values("fiscal_period", ascending=False)

# Show the gap between fiscal period end and when it was reported
df["lag_days"] = (df["reportDate"] - df["fiscal_period"]).dt.days
print(df.head(16).to_string(index=False))

print(f"\nMedian filing lag: {df['lag_days'].median():.0f} days")
print(f"Range: {df['lag_days'].min():.0f} to {df['lag_days'].max():.0f} days")