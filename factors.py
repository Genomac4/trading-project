import pandas as pd
from data_layer import get_fundamentals
from data_layer import get_fundamentals, get_prices, get_valuation

def piotroski_fscore(ticker):
    """
    Compute the Piotroski F-score (0-9) for a ticker.
    Each signal returns 1, 0, or None (if data is missing).
    Score is reported as 'points out of available signals'.
    """
    df = get_fundamentals(ticker)
    if df is None or len(df) < 2:
        print(f"[{ticker}] Not enough data for F-score")
        return None

    # Find the most recent year that has the core fields populated,
    # then pair it with the year immediately before it.
    core_fields = ["NetIncome", "TotalAssets", "OperatingCashFlow"]

    curr = None
    prev = None
    for i in range(len(df) - 1, 0, -1):
        this_year = df.iloc[i]
        year_before = df.iloc[i - 1]
        if all(pd.notna(this_year[f]) for f in core_fields):
            curr = this_year
            prev = year_before
            curr_year = df.index[i]
            prev_year = df.index[i - 1]
            break

    if curr is None:
        print(f"[{ticker}] No year with complete core data found")
        return None

    print(f"[{ticker}] Comparing {curr_year} (current) vs {prev_year} (prior)")

    # We'll collect each signal's result here
    signals = {}
   
    # ===== GROUP 1: PROFITABILITY =====

    # Signal 1: ROA positive this year
    # ROA = Net Income / Total Assets
    roa_curr = None
    if pd.notna(curr["NetIncome"]) and pd.notna(curr["TotalAssets"]) and curr["TotalAssets"] != 0:
        roa_curr = curr["NetIncome"] / curr["TotalAssets"]
        signals["1_ROA_positive"] = 1 if roa_curr > 0 else 0
    else:
        signals["1_ROA_positive"] = None

    # Signal 2: Operating cash flow positive
    if pd.notna(curr["OperatingCashFlow"]):
        signals["2_CFO_positive"] = 1 if curr["OperatingCashFlow"] > 0 else 0
    else:
        signals["2_CFO_positive"] = None

    # Signal 3: ROA improving vs last year
    roa_prev = None
    if pd.notna(prev["NetIncome"]) and pd.notna(prev["TotalAssets"]) and prev["TotalAssets"] != 0:
        roa_prev = prev["NetIncome"] / prev["TotalAssets"]
    if roa_curr is not None and roa_prev is not None:
        signals["3_ROA_improving"] = 1 if roa_curr > roa_prev else 0
    else:
        signals["3_ROA_improving"] = None

    # Signal 4: Accruals — operating cash flow return exceeds ROA
    # (cash earnings are higher quality than accounting earnings)
    if pd.notna(curr["OperatingCashFlow"]) and pd.notna(curr["TotalAssets"]) and curr["TotalAssets"] != 0 and roa_curr is not None:
        cfo_roa = curr["OperatingCashFlow"] / curr["TotalAssets"]
        signals["4_accruals"] = 1 if cfo_roa > roa_curr else 0
    else:
        signals["4_accruals"] = None
# ===== GROUP 2: LEVERAGE & LIQUIDITY =====

    # Signal 5: Long-term debt ratio decreasing (less leverage = healthier)
    # Compare LongTermDebt / TotalAssets this year vs last year
    if (pd.notna(curr["LongTermDebt"]) and pd.notna(curr["TotalAssets"]) and curr["TotalAssets"] != 0
            and pd.notna(prev["LongTermDebt"]) and pd.notna(prev["TotalAssets"]) and prev["TotalAssets"] != 0):
        debt_ratio_curr = curr["LongTermDebt"] / curr["TotalAssets"]
        debt_ratio_prev = prev["LongTermDebt"] / prev["TotalAssets"]
        signals["5_leverage_down"] = 1 if debt_ratio_curr < debt_ratio_prev else 0
    else:
        signals["5_leverage_down"] = None

    # Signal 6: Current ratio improving (better short-term liquidity)
    # Current ratio = Current Assets / Current Liabilities
    if (pd.notna(curr["CurrentAssets"]) and pd.notna(curr["CurrentLiabilities"]) and curr["CurrentLiabilities"] != 0
            and pd.notna(prev["CurrentAssets"]) and pd.notna(prev["CurrentLiabilities"]) and prev["CurrentLiabilities"] != 0):
        cr_curr = curr["CurrentAssets"] / curr["CurrentLiabilities"]
        cr_prev = prev["CurrentAssets"] / prev["CurrentLiabilities"]
        signals["6_current_ratio_up"] = 1 if cr_curr > cr_prev else 0
    else:
        signals["6_current_ratio_up"] = None

    # Signal 7: No new shares issued (no dilution)
    # Shares outstanding this year <= last year
    if pd.notna(curr["SharesOutstanding"]) and pd.notna(prev["SharesOutstanding"]):
        signals["7_no_dilution"] = 1 if curr["SharesOutstanding"] <= prev["SharesOutstanding"] else 0
    else:
        signals["7_no_dilution"] = None
        # ===== GROUP 3: OPERATING EFFICIENCY =====

    # Signal 8: Gross margin improving
    # Gross profit is often not reported directly — derive it:
    # Gross Profit = Revenue - CostOfRevenue
    def gross_margin(row):
        if pd.notna(row.get("GrossProfit")) and pd.notna(row.get("Revenue")) and row["Revenue"] != 0:
            return row["GrossProfit"] / row["Revenue"]
        if (pd.notna(row.get("Revenue")) and pd.notna(row.get("CostOfRevenue"))
                and row["Revenue"] != 0):
            return (row["Revenue"] - row["CostOfRevenue"]) / row["Revenue"]
        return None

    gm_curr = gross_margin(curr)
    gm_prev = gross_margin(prev)
    if gm_curr is not None and gm_prev is not None:
        signals["8_gross_margin_up"] = 1 if gm_curr > gm_prev else 0
    else:
        signals["8_gross_margin_up"] = None

    # Signal 9: Asset turnover improving (more revenue per dollar of assets)
    # Asset turnover = Revenue / Total Assets
    if (pd.notna(curr["Revenue"]) and pd.notna(curr["TotalAssets"]) and curr["TotalAssets"] != 0
            and pd.notna(prev["Revenue"]) and pd.notna(prev["TotalAssets"]) and prev["TotalAssets"] != 0):
        at_curr = curr["Revenue"] / curr["TotalAssets"]
        at_prev = prev["Revenue"] / prev["TotalAssets"]
        signals["9_asset_turnover_up"] = 1 if at_curr > at_prev else 0
    else:
        signals["9_asset_turnover_up"] = None
    # ===== TALLY THE SCORE =====
    # Count how many signals scored 1, and how many were computable
    computable = [v for v in signals.values() if v is not None]
    score = sum(computable)
    available = len(computable)
    missing = 9 - available

    result = {
        "ticker": ticker.upper(),
        "score": score,
        "available": available,
        "missing": missing,
        "signals": signals,
    }

    print(f"[{ticker}] F-Score: {score}/{available}" + (f" ({missing} signals unavailable)" if missing else ""))
    return result
from data_layer import get_prices


def momentum_12_1(ticker):
    """
    Compute 12-1 month momentum: the return from ~12 months ago
    to ~1 month ago, skipping the most recent month to avoid
    short-term reversal noise.
    """
    prices = get_prices(ticker)
    if prices is None or len(prices) < 252:
        print(f"[{ticker}] Not enough price history for momentum")
        return None

    # Trading days: ~21 per month, ~252 per year.
    # We need the price ~252 days ago (12 months) and ~21 days ago (1 month).
    close = prices["Close"]

    price_12mo_ago = close.iloc[-252]   # ~12 months back
    price_1mo_ago = close.iloc[-21]     # ~1 month back

    momentum = (price_1mo_ago / price_12mo_ago) - 1

    print(f"[{ticker}] 12-1 momentum: {momentum:.1%} "
          f"(${price_12mo_ago:.2f} → ${price_1mo_ago:.2f})")

    return {
        "ticker": ticker.upper(),
        "momentum_12_1": momentum,
        "price_12mo_ago": float(price_12mo_ago),
        "price_1mo_ago": float(price_1mo_ago),
    }
def trend_signal(ticker):
    """
    Compute trend indicators from price data:
    - 50-day and 200-day moving averages
    - whether price is above/below the 200-day (long-term trend)
    - whether 50-day is above 200-day (golden cross = bullish)
    """
    prices = get_prices(ticker)
    if prices is None or len(prices) < 200:
        print(f"[{ticker}] Not enough price history for trend")
        return None

    close = prices["Close"]
    current_price = close.iloc[-1]

    # Moving averages: the mean of the last N closing prices
    ma_50 = close.iloc[-50:].mean()
    ma_200 = close.iloc[-200:].mean()

    # How far is price above or below its 200-day trend line?
    pct_vs_200 = (current_price / ma_200) - 1

    # Golden cross: short-term trend above long-term trend = bullish
    golden_cross = ma_50 > ma_200

    # Is price above its long-term trend at all?
    above_200 = current_price > ma_200

    print(f"[{ticker}] Price ${current_price:.2f} | "
          f"50d ${ma_50:.2f} | 200d ${ma_200:.2f} | "
          f"{pct_vs_200:+.1%} vs 200d | "
          f"{'GOLDEN CROSS' if golden_cross else 'below 50<200'}")

    return {
        "ticker": ticker.upper(),
        "price": float(current_price),
        "ma_50": float(ma_50),
        "ma_200": float(ma_200),
        "pct_vs_200d": float(pct_vs_200),
        "above_200d": bool(above_200),
        "golden_cross": bool(golden_cross),
    }
def value_factor(ticker, your_target=None):
    """
    Valuation metrics using EODHD's professional calculations:
    trailing & forward P/E, EV/EBITDA, EV/Revenue, PEG.
    Plus analyst consensus as a reference layer (not a ranking input).
    Optional your_target: your own price target to compare vs consensus.
    """
    val = get_valuation(ticker)
    if val is None:
        print(f"[{ticker}] No valuation data")
        return None

    price = get_prices(ticker)["Close"].iloc[-1]

    # Core valuation multiples (professional, pre-computed)
    print(f"\n[{ticker}] === VALUATION ===")
    print(f"  Price:           ${price:.2f}")
    print(f"  Trailing P/E:    {val['trailing_pe']:.1f}x" if val['trailing_pe'] else "  Trailing P/E:    n/a")
    print(f"  Forward P/E:     {val['forward_pe']:.1f}x" if val['forward_pe'] else "  Forward P/E:     n/a")
    print(f"  EV/EBITDA:       {val['ev_ebitda']:.1f}x" if val['ev_ebitda'] else "  EV/EBITDA:       n/a")
    print(f"  EV/Revenue:      {val['ev_revenue']:.1f}x" if val['ev_revenue'] else "  EV/Revenue:      n/a")
    print(f"  PEG:             {val['peg']:.1f}" if val['peg'] else "  PEG:             n/a")

    # Forward earnings growth implied by estimates
    eps_now = val.get("eps_est_current_yr")
    eps_next = val.get("eps_est_next_yr")
    if eps_now and eps_next and eps_now > 0:
        implied_growth = (eps_next / eps_now) - 1
        print(f"  Implied EPS growth (next yr): {implied_growth:.1%}")

    # Analyst consensus — REFERENCE ONLY, not a ranking factor
    target = val.get("analyst_target")
    rating = val.get("analyst_rating")
    if target and price:
        upside = (target / price) - 1
        print(f"\n[{ticker}] --- Analyst reference (sell-side consensus) ---")
        print(f"  Consensus target: ${target:.0f}  ({upside:+.1%} vs current)")
        print(f"  Rating: {rating:.2f}/5  "
              f"(SB:{val.get('strong_buy')} B:{val.get('buy')} "
              f"H:{val.get('hold')} S:{val.get('sell')})")

    # Your own view vs consensus (if you provide a target)
    if your_target and price:
        your_upside = (your_target / price) - 1
        print(f"\n[{ticker}] --- Your view ---")
        print(f"  Your target: ${your_target:.0f}  ({your_upside:+.1%} vs current)")
        if target:
            gap = (your_target / target) - 1
            print(f"  vs consensus: {gap:+.1%} "
                  f"({'more bullish' if gap > 0 else 'more bearish'} than the Street)")

    return val

    # --- Market cap ---
    if pd.isna(shares) or shares <= 0:
        print(f"[{ticker}] No shares outstanding — cannot value")
        return None
    market_cap = price * shares

    # --- Net debt (approximate): use LongTermDebt minus cash if available ---
    long_term_debt = latest.get("LongTermDebt")
    net_debt = long_term_debt if pd.notna(long_term_debt) else 0
    enterprise_value = market_cap + net_debt

    result = {"ticker": ticker.upper(), "price": float(price),
              "market_cap": float(market_cap),
              "enterprise_value": float(enterprise_value)}

    # --- EV / EBITDA ---
    if pd.notna(ebitda) and ebitda > 0:
        result["ev_ebitda"] = enterprise_value / ebitda
    else:
        result["ev_ebitda"] = None

    # --- P / E ---
    if pd.notna(net_income) and net_income > 0:
        result["pe"] = market_cap / net_income
    else:
        result["pe"] = None

    # --- P / FCF ---
    if pd.notna(fcf) and fcf > 0:
        result["p_fcf"] = market_cap / fcf
    else:
        result["p_fcf"] = None

    # Print a clean summary
    def fmt(x):
        return f"{x:.1f}x" if x is not None else "n/a"
    print(f"[{ticker}] Price ${price:.2f} | MktCap ${market_cap/1e9:.1f}B | "
          f"EV ${enterprise_value/1e9:.1f}B")
    print(f"[{ticker}] EV/EBITDA {fmt(result['ev_ebitda'])} | "
          f"P/E {fmt(result['pe'])} | P/FCF {fmt(result['p_fcf'])}")

    return result
# ---- Test ----
if __name__ == "__main__":
    print(momentum_12_1("CEG"))
    print()
    print(trend_signal("CEG"))
    print()
    result = piotroski_fscore("CEG")
    print()
    for signal, value in result["signals"].items():
        print(f"  {signal}: {value}")
        print()
    value_factor("CEG")
    print()
    print(get_valuation("CEG"))
    print()
    value_factor("CEG", your_target=375)