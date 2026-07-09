import requests
from config import EODHD_API_KEY

def test_fundamentals(ticker):
    url = f"https://eodhd.com/api/fundamentals/{ticker}.US?api_token={EODHD_API_KEY}&fmt=json"
    data = requests.get(url).json()

    income = data["Financials"]["Income_Statement"]["yearly"]
    balance = data["Financials"]["Balance_Sheet"]["yearly"]
    cashflow = data["Financials"]["Cash_Flow"]["yearly"]

    print(f"\n========== {ticker} ==========")
    for date in sorted(income.keys(), reverse=True)[:6]:
        inc = income[date]
        bal = balance.get(date, {})
        cf = cashflow.get(date, {})

        rev = inc.get("totalRevenue")
        ni = inc.get("netIncome")
        gp = inc.get("grossProfit")
        ta = bal.get("totalAssets")
        ocf = cf.get("totalCashFromOperatingActivities")

        print(f"{date[:4]}: Rev={rev} | NI={ni} | GP={gp} | Assets={ta} | OCF={ocf}")


for ticker in ["MSFT", "CEG", "COST"]:
    test_fundamentals(ticker)