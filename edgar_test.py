import requests
import pandas as pd

headers = {
    "User-Agent": "Eugene McEnery genomac44@gmail.com"
}


def get_cik(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=headers).json()
    for entry in data.values():
        if entry["ticker"] == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    return None


def get_all_facts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return requests.get(url, headers=headers).json()


def get_annual_figures(gaap, tag):
    if tag not in gaap:
        return {}
    results = {}
    for item in gaap[tag]["units"]["USD"]:
        if item.get("form") == "10-K" and item.get("fp") == "FY" and "frame" in item:
            year = item["end"][:4]
            results[year] = item["val"]
    return results


def get_metric(gaap, tag_priority_list):
    for tag in tag_priority_list:
        figures = get_annual_figures(gaap, tag)
        if figures:
            return figures
    return {}

METRICS = {
    "Revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    ],
    "NetIncome": ["NetIncomeLoss"],
    "TotalAssets": ["Assets"],
    "TotalLiabilities": ["Liabilities"],
    "OperatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities"],
}


def pull_fundamentals(ticker):
    """Pull fundamentals and return them as a clean DataFrame."""
    cik = get_cik(ticker)
    if not cik:
        print(f"Could not find CIK for {ticker}")
        return None

    data = get_all_facts(cik)
    gaap = data["facts"]["us-gaap"]

    # Build a dictionary of {metric: {year: value}}
    all_data = {}
    for label, tags in METRICS.items():
        all_data[label] = get_metric(gaap, tags)

    # Convert to a DataFrame: years as rows, metrics as columns
    df = pd.DataFrame(all_data)
    df.index.name = "Year"
    df = df.sort_index()

    # Add the ticker as a column for when we combine many companies
    df["Ticker"] = ticker.upper()

    return df


# Test it
ticker = "CEG
df = pull_fundamentals(ticker)

print(df)

# Save to CSV
df.to_csv(f"{ticker}_fundamentals.csv")
print(f"\nSaved to {ticker}_fundamentals.csv")
