import yfinance as yf

# Companies that were major S&P 500 members but no longer exist

dead_companies = {
    "LEH": "Lehman Brothers (bankrupt 2008)",
    "BSC": "Bear Stearns (collapsed 2008)",
    "WAMUQ": "Washington Mutual (failed 2008)",
    "ENRNQ": "Enron (fraud, bankrupt 2001)",
    "GM-OLD": "Old General Motors (bankrupt 2009)"
}

print("Trying to pull data for companies that no longer exist:\n")

for ticker, name in dead_companies.items():
    data = yf.download(ticker, period="10y", interval="1d", progress=False)
    print(f"{ticker} ({name}): {len(data)} rows of data")
