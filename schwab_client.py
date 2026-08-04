"""Read-only Schwab account access for the dashboard.

READ-ONLY BY DESIGN. This module only ever issues GET requests: account
numbers, balances, and positions. It contains no code capable of placing,
modifying, or cancelling an order, and none may be added. Order-capable
methods on the underlying schwab-py client (place_order, replace_order,
cancel_order, ...) are never called and never exposed from here — callers
should import these functions, not build their own client.

Auth is a one-time browser flow run from the terminal:

    python schwab_client.py login

which stores a token at TOKEN_PATH (gitignored — it is a credential).
The dashboard only ever loads that token file; if it is missing or the
7-day refresh token has expired, every function raises SchwabAuthError
with re-login instructions instead of crashing.
"""
import os
import sys

from config import SCHWAB_APP_KEY, SCHWAB_APP_SECRET, SCHWAB_CALLBACK_URL

TOKEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "schwab_token.json")
RELOGIN_HINT = ("Run `python schwab_client.py login` in a terminal to "
                "(re)connect your Schwab account, then reload this page.")


class SchwabAuthError(Exception):
    """Token missing/expired/invalid — user needs to redo the login flow."""


def _client():
    """Client from the stored token file only — never starts a login flow
    (the dashboard must not try to open a browser from inside Streamlit)."""
    from schwab.auth import client_from_token_file
    if not os.path.exists(TOKEN_PATH):
        raise SchwabAuthError(f"No Schwab token found. {RELOGIN_HINT}")
    try:
        return client_from_token_file(TOKEN_PATH, SCHWAB_APP_KEY, SCHWAB_APP_SECRET)
    except Exception as e:
        raise SchwabAuthError(f"Could not load the Schwab token ({e}). {RELOGIN_HINT}")


def _get_json(resp, what):
    """Unwrap a schwab-py GET response, mapping auth failures to SchwabAuthError."""
    if resp.status_code in (400, 401, 403):
        raise SchwabAuthError(
            f"Schwab rejected the {what} request (HTTP {resp.status_code}) — "
            f"the session has likely expired. {RELOGIN_HINT}")
    if resp.status_code != 200:
        raise RuntimeError(f"Schwab {what} request failed: HTTP {resp.status_code} "
                           f"{resp.text[:200]}")
    return resp.json()


def get_accounts():
    """Account numbers on file: list of {'accountNumber', 'hashValue'} dicts.
    (GET /accounts/accountNumbers)"""
    c = _client()
    return _get_json(c.get_account_numbers(), "account numbers")


def _all_accounts_with_positions():
    """Every account with its positions attached. (GET /accounts?fields=positions)"""
    c = _client()
    return _get_json(
        c.get_accounts(fields=c.Account.Fields.POSITIONS), "accounts/positions")


def _f(d, *keys):
    """First present-and-not-None float among keys, else None."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def get_positions():
    """Holdings across all accounts as a list of dicts:
    ticker, asset_type, quantity, market_value, cost_basis, unrealized_pnl,
    unrealized_pnl_pct, account (last 4 of the account number).

    Quantity is long minus short so a short position shows negative. Cost
    basis is averagePrice x quantity; unrealized P&L is market value minus
    cost basis (falls back to Schwab's longOpenProfitLoss if averagePrice
    is missing). Any field Schwab omits comes back None — callers render
    those as N/A rather than crashing.

    OPTION positions: Schwab reports marketValue with the 100x contract
    multiplier baked in but averagePrice per share, so the basis is scaled
    by 100 to match (verified against Schwab's own longOpenProfitLoss,
    which equals marketValue - averagePrice*qty*100 to the cent)."""
    out = []
    for acct in _all_accounts_with_positions():
        sa = acct.get("securitiesAccount", {})
        acct_tail = str(sa.get("accountNumber", ""))[-4:]
        for pos in sa.get("positions", []):
            inst = pos.get("instrument", {})
            qty = (_f(pos, "longQuantity") or 0.0) - (_f(pos, "shortQuantity") or 0.0)
            mv = _f(pos, "marketValue")
            avg = _f(pos, "averagePrice")
            mult = 100.0 if inst.get("assetType") == "OPTION" else 1.0
            basis = avg * qty * mult if (avg is not None and qty) else None
            pnl = (mv - basis) if (mv is not None and basis is not None) else \
                _f(pos, "longOpenProfitLoss")
            pnl_pct = (pnl / abs(basis)) if (pnl is not None and basis) else None
            out.append({
                "ticker": str(inst.get("symbol", "")).strip().upper(),
                "asset_type": inst.get("assetType", ""),
                "description": inst.get("description", ""),
                "underlying": inst.get("underlyingSymbol", ""),
                "quantity": qty,
                "market_value": mv,
                "cost_basis": basis,
                "unrealized_pnl": pnl,
                "unrealized_pnl_pct": pnl_pct,
                "account": acct_tail,
            })
    return out


def get_balances():
    """Per-account balances: list of dicts with account (last 4), type,
    total_value (liquidation value), cash, buying_power. Buying power falls
    back to cash available for trading on cash accounts."""
    out = []
    for acct in _all_accounts_with_positions():
        sa = acct.get("securitiesAccount", {})
        cur = sa.get("currentBalances", {})
        out.append({
            "account": str(sa.get("accountNumber", ""))[-4:],
            "type": sa.get("type", ""),
            "total_value": _f(cur, "liquidationValue", "totalValue"),
            "cash": _f(cur, "cashBalance", "totalCash"),
            "buying_power": _f(cur, "buyingPower", "cashAvailableForTrading",
                               "availableFunds"),
        })
    return out


def get_option_chain(symbol, from_date=None, to_date=None, strike_count=None,
                     calls_only=True):
    """Raw option-chain JSON for one underlying. (GET /marketdata/chains)

    Read-only market data — same firewall as everything else here: a GET,
    no order capability. from_date/to_date (datetime.date) bound the expiries
    and strike_count limits strikes around ATM, keeping payloads small."""
    c = _client()
    kwargs = {}
    if calls_only:
        kwargs["contract_type"] = c.Options.ContractType.CALL
    if from_date is not None:
        kwargs["from_date"] = from_date
    if to_date is not None:
        kwargs["to_date"] = to_date
    if strike_count is not None:
        kwargs["strike_count"] = strike_count
    return _get_json(c.get_option_chain(symbol, **kwargs), "option chain")


def login():
    """One-time interactive OAuth: opens a browser to Schwab's login page and
    captures the callback, writing the token to TOKEN_PATH. Falls back to the
    copy/paste manual flow if the local callback server can't run."""
    from schwab.auth import client_from_login_flow, client_from_manual_flow
    print(f"Starting Schwab OAuth. Callback URL: {SCHWAB_CALLBACK_URL}")
    print("A browser window will open — log in to Schwab and approve access.")
    try:
        client_from_login_flow(SCHWAB_APP_KEY, SCHWAB_APP_SECRET,
                               SCHWAB_CALLBACK_URL, TOKEN_PATH)
    except Exception as e:
        print(f"\nBrowser flow failed ({e}); falling back to the manual flow.")
        print("Log in in the browser, approve, then paste the FULL redirect URL "
              "from the address bar (it will start with the callback URL and "
              "look like an error page — that's expected).")
        client_from_manual_flow(SCHWAB_APP_KEY, SCHWAB_APP_SECRET,
                                SCHWAB_CALLBACK_URL, TOKEN_PATH)
    print(f"\nToken saved to {TOKEN_PATH} (gitignored).")
    accts = get_accounts()
    print(f"Connected — {len(accts)} account(s): "
          + ", ".join("..." + str(a.get("accountNumber", ""))[-4:] for a in accts))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "login":
        login()
    else:
        print(__doc__)
        print("Usage: python schwab_client.py login")
