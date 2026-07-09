import requests
import pandas as pd
from datetime import datetime
from config import EODHD_API_KEY


def get_sp500_membership():
    """
    Pull the full historical S&P 500 membership table from EODHD.
    Returns a DataFrame: one row per stock-stint with StartDate/EndDate.
    """
    url = f"https://eodhd.com/api/fundamentals/GSPC.INDX?api_token={EODHD_API_KEY}&fmt=json&historical=1"
    data = requests.get(url).json()

    htc = data.get("HistoricalTickerComponents", {})
    rows = []
    for rec in htc.values():
        rows.append({
            "ticker": rec.get("Code"),
            "name": rec.get("Name"),
            "start": rec.get("StartDate"),
            "end": rec.get("EndDate"),
            "is_delisted": rec.get("IsDelisted"),
        })

    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    return df


def members_on_date(membership_df, date):
    """
    Return the set of tickers that were S&P 500 members on a given date.
    A stock counts if: start <= date AND (end is null OR end > date).
    """
    date = pd.Timestamp(date)
    in_index = (
        (membership_df["start"] <= date) &
        (membership_df["end"].isna() | (membership_df["end"] > date))
    )
    return set(membership_df.loc[in_index, "ticker"].dropna())


if __name__ == "__main__":
    membership = get_sp500_membership()
    print(f"Total membership records: {len(membership)}")
    print(f"Stocks that were ever in the index: {membership['ticker'].nunique()}")
    print(f"Currently still in (no end date): {membership['end'].isna().sum()}")
    print(f"Left the index (have end date): {membership['end'].notna().sum()}")

    # Test the as-of-date machinery on a few historical dates
    for test_date in ["2017-12-31", "2020-03-31", "2023-06-30"]:
        members = members_on_date(membership, test_date)
        print(f"\nMembers on {test_date}: {len(members)} stocks")
        # Show a few that LEFT later — proof we're capturing the dead
        sample = list(members)[:8]
        print(f"  Sample: {sample}")