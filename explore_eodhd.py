import requests
from config import EODHD_API_KEY

url = f"https://eodhd.com/api/fundamentals/CEG.US?api_token={EODHD_API_KEY}&fmt=json"
data = requests.get(url).json()

# Look at the most recent year of each statement and list available fields
income = data["Financials"]["Income_Statement"]["yearly"]
balance = data["Financials"]["Balance_Sheet"]["yearly"]
cashflow = data["Financials"]["Cash_Flow"]["yearly"]

latest = sorted(income.keys(), reverse=True)[0]

print("=== INCOME STATEMENT fields ===")
for k in income[latest].keys():
    print(" ", k)

print("\n=== BALANCE SHEET fields ===")
for k in balance[latest].keys():
    print(" ", k)

print("\n=== CASH FLOW fields ===")
for k in cashflow[latest].keys():
    print(" ", k)