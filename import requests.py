import requests
from config import EODHD_API_KEY

ticker = "MSFT.US"
url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={EODHD_API_KEY}&fmt=json"

data = requests.get(url).json()

# EODHD nests financial statements under Financials
income = data["Financials"]["Income_Statement"]["yearly"]
balance = data["Financials"]["Balance_Sheet"]["yearly"]
cashflow = data["Financials"]["Cash_Flow"]["yearly"]

print("MSFT fundamentals via EODHD (most recent 6 years):\n")

for date in sorted(income.keys(), reverse=True)[:6]:
    inc = income[date]
    bal = balance.get(date, {})
    cf = cashflow.get(date, {})

    revenue = inc.get("totalRevenue")
    net_income = inc.get("netIncome")
    gross_profit = inc.get("grossProfit")
    total_assets = bal.get("totalAssets")
    operating_cf = cf.get("totalCashFromOperatingActivities")

    print(f"=== {date[:4]} ===")
    print(f"  Revenue:        {revenue}")
    print(f"  Net Income:     {net_income}")
    print(f"  Gross Profit:   {gross_profit}")
    print(f"  Total Assets:   {total_assets}")
    print(f"  Operating CF:   {operating_cf}")
    print()