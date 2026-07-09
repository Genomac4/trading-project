import yfinance as yf
import pandas as pd

# Pull price data for CEG since its 2022 IPO
ticker = "CEG"
data = yf.download(ticker, period="10y", interval="1d")

# Flatten the messy MultiIndex columns into simple names
data.columns = data.columns.get_level_values(0)

# Save the clean data to the data folder
data.to_csv(f"data/{ticker}_prices.csv")

# Show what we got
print("First 5 rows:")
print(data.head())
print("\nLast 5 rows:")
print(data.tail())
print("\nTotal rows:", len(data))
print("Columns:", list(data.columns))
print(f"\nData saved to data/{ticker}_prices.csv")