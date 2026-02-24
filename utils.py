import re
import pandas as pd


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Strip whitespace from string columns
    for col in df.columns:
        df[col] = df[col].str.strip() if df[col].dtype == object else df[col]

    # Numeric columns
    for col in ["Quantity", "Price", "Interest", "Amount", "Commission", "Fee"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Date columns
    for col in ["TradeDate", "SettledDate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Normalise Action / RecordType
    df["Action"] = df["Action"].fillna("").str.strip()
    df["RecordType"] = df["RecordType"].fillna("").str.strip()
    df["Symbol"] = df["Symbol"].fillna("").str.strip()

    # Parse option fields from Description
    df = _parse_option_fields(df)

    return df


def _parse_option_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows that are option trades (Symbol is blank and Description starts
    with CALL/PUT), extract: OptionUnderlying, OptionType, OptionExpiry,
    OptionStrike, IsOption, OptionStatus.
    """
    opt_re = re.compile(
        r"^(CALL|PUT)\s+(\w+)\s+([\d/]+)\s+([\d.]+)",
        re.IGNORECASE,
    )
    expired_re = re.compile(r"OPTION EXPIRATION", re.IGNORECASE)
    exercised_re = re.compile(r"\bEXERCISED\b", re.IGNORECASE)
    closing_re = re.compile(r"CLOSING CONTRACT", re.IGNORECASE)
    opening_re = re.compile(r"OPEN CONTRACT", re.IGNORECASE)

    underlying, opt_type, expiry, strike, is_opt, status = [], [], [], [], [], []

    for _, row in df.iterrows():
        desc = str(row.get("Description", ""))
        m = opt_re.match(desc)
        if m:
            is_opt.append(True)
            opt_type.append(m.group(1).upper())
            underlying.append(m.group(2).upper())
            expiry.append(m.group(3))
            strike.append(float(m.group(4)))
            if expired_re.search(desc):
                status.append("Expired")
            elif exercised_re.search(desc):
                status.append("Exercised")
            elif closing_re.search(desc):
                status.append("Closed")
            elif opening_re.search(desc):
                status.append("Opened")
            else:
                status.append("Unknown")
        else:
            is_opt.append(False)
            opt_type.append(None)
            underlying.append(None)
            expiry.append(None)
            strike.append(None)
            status.append(None)

    df["IsOption"] = is_opt
    df["OptionType"] = opt_type
    df["OptionUnderlying"] = underlying
    df["OptionExpiry"] = expiry
    df["OptionStrike"] = strike
    df["OptionStatus"] = status
    return df


# ---------------------------------------------------------------------------
# Realised P&L — average cost method
# ---------------------------------------------------------------------------

def compute_realized_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk stock trades chronologically per symbol and compute realised P&L
    for each SELL row using the average-cost method.
    Returns the trades DataFrame with extra columns:
        AvgCostBasis, RealizedPnL, RealizedPnLPerShare
    """
    trades = df[
        (df["RecordType"] == "Trade") & (~df["IsOption"])
    ].copy()
    _order = {"BUY": 0, "SELL": 1, "Other": 2, "": 2}
    trades["_sort_priority"] = trades["Action"].map(_order).fillna(2).astype(int)
    trades = trades.sort_values(["TradeDate", "_sort_priority"], kind="stable")

    results = []
    # cost_basis[symbol] = (total_shares, total_cost)
    cost_basis: dict[str, tuple[float, float]] = {}

    for idx, row in trades.iterrows():
        sym = row["Symbol"]
        qty = row["Quantity"]   # positive = buy, negative = sell
        price = row["Price"]
        action = row["Action"].upper()

        if sym not in cost_basis:
            cost_basis[sym] = (0.0, 0.0)

        shares, total_cost = cost_basis[sym]
        avg_cost = total_cost / shares if shares else 0.0
        realized = 0.0

        if action == "BUY" and qty > 0:
            shares += qty
            total_cost += qty * price
            avg_cost = total_cost / shares if shares else 0.0

        elif action == "SELL" and qty < 0:
            sell_qty = abs(qty)
            if shares > 0:
                realized = sell_qty * (price - avg_cost)
            shares = max(0.0, shares - sell_qty)
            total_cost = shares * avg_cost

        # Handle free stock transfers (Other, no price) treated as BUY @ 0
        elif action in ("", "OTHER") and qty > 0 and price == 0:
            shares += qty
            # cost stays as-is (unknown basis, treated as $0)

        cost_basis[sym] = (shares, total_cost)
        results.append({
            "idx": idx,
            "AvgCostBasis": avg_cost,
            "RealizedPnL": realized,
            "RemainingShares": shares,
        })

    pnl_df = pd.DataFrame(results).set_index("idx")
    trades = trades.join(pnl_df, how="left")
    return trades


def compute_option_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match option open/close/expired rows by contract key and compute P&L.
    Returns one row per contract lifecycle with P&L.
    """
    opts = df[df["IsOption"]].copy()
    # Within the same date, BUY (open) must be processed before SELL (close)
    # and SELL before Other (expiry). Use a numeric priority key.
    _order = {"BUY": 0, "SELL": 1, "Other": 2, "": 2}
    opts["_sort_priority"] = opts["Action"].map(_order).fillna(2).astype(int)
    opts = opts.sort_values(["TradeDate", "_sort_priority"], kind="stable")

    # Build a contract key: underlying + expiry + strike + type
    opts["ContractKey"] = (
        opts["OptionUnderlying"].fillna("") + "|"
        + opts["OptionExpiry"].fillna("") + "|"
        + opts["OptionStrike"].astype(str) + "|"
        + opts["OptionType"].fillna("")
    )

    records = []
    open_positions: dict[str, list] = {}  # key -> list of open rows

    for _, row in opts.iterrows():
        key = row["ContractKey"]
        action = row["Action"].upper()
        status = row["OptionStatus"]
        contracts = abs(row["Quantity"])
        amount = row["Amount"]  # negative for buys, positive for sells
        date = row["TradeDate"]

        if key not in open_positions:
            open_positions[key] = []

        if action == "BUY" and status == "Opened":
            open_positions[key].append({
                "OpenDate": date,
                "OpenContracts": contracts,
                "OpenCost": amount,  # negative (cash out)
                "ContractKey": key,
                "Underlying": row["OptionUnderlying"],
                "OptionType": row["OptionType"],
                "OptionExpiry": row["OptionExpiry"],
                "OptionStrike": row["OptionStrike"],
            })

        elif action == "SELL" and status == "Closed":
            remaining = contracts
            while remaining > 0 and open_positions.get(key):
                opener = open_positions[key][0]
                fill = min(remaining, opener["OpenContracts"])
                ratio = fill / opener["OpenContracts"]
                cost_alloc = opener["OpenCost"] * ratio  # negative
                proceeds = amount * (fill / contracts)   # positive
                pnl = proceeds + cost_alloc              # proceeds - abs(cost)

                records.append({
                    "Underlying": opener["Underlying"],
                    "OptionType": opener["OptionType"],
                    "OptionExpiry": opener["OptionExpiry"],
                    "OptionStrike": opener["OptionStrike"],
                    "OpenDate": opener["OpenDate"],
                    "CloseDate": date,
                    "Contracts": fill,
                    "OpenCost": abs(cost_alloc),
                    "CloseProceeds": proceeds,
                    "RealizedPnL": pnl,
                    "Outcome": "Closed",
                })

                opener["OpenContracts"] -= fill
                opener["OpenCost"] -= cost_alloc
                if opener["OpenContracts"] <= 0:
                    open_positions[key].pop(0)
                remaining -= fill

        elif status in ("Expired", "Exercised"):
            outcome_label = "Expired Worthless" if status == "Expired" else "Exercised"
            while open_positions.get(key):
                opener = open_positions[key].pop(0)
                records.append({
                    "Underlying": opener["Underlying"],
                    "OptionType": opener["OptionType"],
                    "OptionExpiry": opener["OptionExpiry"],
                    "OptionStrike": opener["OptionStrike"],
                    "OpenDate": opener["OpenDate"],
                    "CloseDate": date,
                    "Contracts": opener["OpenContracts"],
                    "OpenCost": abs(opener["OpenCost"]),
                    "CloseProceeds": 0.0,
                    "RealizedPnL": opener["OpenCost"],  # negative (premium cost)
                    "Outcome": outcome_label,
                })

    # Still-open options
    for key, openers in open_positions.items():
        for opener in openers:
            records.append({
                "Underlying": opener["Underlying"],
                "OptionType": opener["OptionType"],
                "OptionExpiry": opener["OptionExpiry"],
                "OptionStrike": opener["OptionStrike"],
                "OpenDate": opener["OpenDate"],
                "CloseDate": None,
                "Contracts": opener["OpenContracts"],
                "OpenCost": abs(opener["OpenCost"]),
                "CloseProceeds": None,
                "RealizedPnL": None,
                "Outcome": "Still Open",
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def get_dividends(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Action"] == "Dividend"].copy()


def get_margin_interest(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["Action"] == "Interest") & (df["Amount"] < 0)
    return df[mask].copy()


def get_credit_interest(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["Action"] == "Interest") & (df["Amount"] > 0)
    return df[mask].copy()


def get_cash_deposits(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Description"].str.contains("ACH DEPOSIT", case=False, na=False)
    return df[mask].copy()


def get_all_stock_symbols(df: pd.DataFrame) -> list[str]:
    syms = df[
        (df["RecordType"] == "Trade") & (~df["IsOption"]) & (df["Symbol"] != "")
    ]["Symbol"].unique().tolist()
    return sorted(syms)


def get_all_option_underlyings(df: pd.DataFrame) -> list[str]:
    syms = df[df["IsOption"]]["OptionUnderlying"].dropna().unique().tolist()
    return sorted(syms)


# ---------------------------------------------------------------------------
# Per-symbol metrics
# ---------------------------------------------------------------------------

def symbol_metrics(df: pd.DataFrame, symbol: str, stock_pnl: pd.DataFrame | None = None) -> dict:
    stock_trades = stock_pnl if stock_pnl is not None else compute_realized_pnl(df)
    sym_trades = stock_trades[stock_trades["Symbol"] == symbol]

    buys = sym_trades[sym_trades["Action"] == "BUY"]
    sells = sym_trades[sym_trades["Action"] == "SELL"]

    total_bought_shares = buys["Quantity"].sum()
    total_sold_shares = abs(sells["Quantity"].sum())
    remaining_shares = total_bought_shares - total_sold_shares

    avg_buy_price = (
        (buys["Quantity"] * buys["Price"]).sum() / buys["Quantity"].sum()
        if not buys.empty and buys["Quantity"].sum() > 0 else 0.0
    )
    avg_sell_price = (
        (abs(sells["Quantity"]) * sells["Price"]).sum() / abs(sells["Quantity"]).sum()
        if not sells.empty and abs(sells["Quantity"]).sum() > 0 else 0.0
    )

    total_realized_pnl = sells["RealizedPnL"].sum()
    total_invested = (buys["Quantity"] * buys["Price"]).sum()
    total_fees = sym_trades["Fee"].sum()

    winning_sells = sells[sells["RealizedPnL"] > 0]
    losing_sells = sells[sells["RealizedPnL"] < 0]
    win_rate = len(winning_sells) / len(sells) * 100 if len(sells) > 0 else 0.0

    dividends = get_dividends(df)
    sym_divs = dividends[dividends["Symbol"] == symbol]
    total_dividends = sym_divs["Amount"].sum()

    return {
        "symbol": symbol,
        "total_buys": len(buys),
        "total_sells": len(sells),
        "total_bought_shares": total_bought_shares,
        "total_sold_shares": total_sold_shares,
        "remaining_shares": remaining_shares,
        "avg_buy_price": avg_buy_price,
        "avg_sell_price": avg_sell_price,
        "total_invested": total_invested,
        "total_realized_pnl": total_realized_pnl,
        "total_fees": total_fees,
        "win_rate": win_rate,
        "total_dividends": total_dividends,
        "net_pnl": total_realized_pnl + total_dividends - total_fees,
        "best_sell_pnl": sells["RealizedPnL"].max() if not sells.empty else 0.0,
        "worst_sell_pnl": sells["RealizedPnL"].min() if not sells.empty else 0.0,
        "trades": sym_trades,
    }


# ---------------------------------------------------------------------------
# Per-day metrics (optionally filtered by symbol)
# ---------------------------------------------------------------------------

def day_metrics(
    df: pd.DataFrame,
    date: pd.Timestamp,
    symbol: str | None = None,
    stock_pnl: pd.DataFrame | None = None,
    opt_pnl: pd.DataFrame | None = None,
) -> dict:
    day = df[df["TradeDate"].dt.date == date.date()].copy()

    if symbol:
        # For stock rows filter by symbol; for option rows filter by underlying
        stock_mask = (day["Symbol"] == symbol) & (~day["IsOption"])
        opt_mask = day["IsOption"] & (day["OptionUnderlying"] == symbol)
        div_mask = (day["Action"] == "Dividend") & (day["Symbol"] == symbol)
        day = day[stock_mask | opt_mask | div_mask]

    stock_trades_day = day[
        (day["RecordType"] == "Trade") & (~day["IsOption"])
    ]
    option_trades_day = day[
        (day["RecordType"] == "Trade") & day["IsOption"]
    ]
    dividends_day = day[day["Action"] == "Dividend"]
    interest_day = day[day["Action"] == "Interest"]
    other_day = day[day["Action"].isin(["Other"]) & ~day["IsOption"]]

    # Realized P&L on this day requires full history context
    all_stock_pnl = stock_pnl if stock_pnl is not None else compute_realized_pnl(df)
    day_pnl_rows = all_stock_pnl[
        all_stock_pnl["TradeDate"].dt.date == date.date()
    ]
    if symbol:
        day_pnl_rows = day_pnl_rows[day_pnl_rows["Symbol"] == symbol]

    realized_stock_pnl = day_pnl_rows[
        day_pnl_rows["Action"] == "SELL"
    ]["RealizedPnL"].sum()

    # Options P&L on this day
    all_opt_pnl = opt_pnl if opt_pnl is not None else compute_option_pnl(df)
    if not all_opt_pnl.empty:
        day_opt_pnl_rows = all_opt_pnl[
            pd.to_datetime(all_opt_pnl["CloseDate"]).dt.date == date.date()
        ]
        if symbol:
            day_opt_pnl_rows = day_opt_pnl_rows[
                day_opt_pnl_rows["Underlying"] == symbol
            ]
        realized_opt_pnl = day_opt_pnl_rows["RealizedPnL"].sum()
    else:
        day_opt_pnl_rows = pd.DataFrame()
        realized_opt_pnl = 0.0

    total_day_bought = stock_trades_day[stock_trades_day["Action"] == "BUY"]["Amount"].sum()
    total_day_sold = stock_trades_day[stock_trades_day["Action"] == "SELL"]["Amount"].sum()
    total_fees_day = stock_trades_day["Fee"].sum() + option_trades_day["Fee"].sum()
    total_dividends_day = dividends_day["Amount"].sum()
    total_margin_day = interest_day[interest_day["Amount"] < 0]["Amount"].sum()
    total_opt_spent = option_trades_day[option_trades_day["Action"] == "BUY"]["Amount"].sum()
    total_opt_received = option_trades_day[option_trades_day["Action"] == "SELL"]["Amount"].sum()

    net_cash_flow = (
        total_day_sold + total_day_bought   # bought is negative
        + total_opt_received + total_opt_spent
        + total_dividends_day + total_margin_day
    )

    return {
        "date": date.date(),
        "symbol_filter": symbol,
        "stock_trades": stock_trades_day,
        "option_trades": option_trades_day,
        "option_pnl_rows": day_opt_pnl_rows,
        "dividends": dividends_day,
        "interest": interest_day,
        "other": other_day,
        "realized_stock_pnl": realized_stock_pnl,
        "realized_opt_pnl": realized_opt_pnl,
        "total_realized_pnl": realized_stock_pnl + realized_opt_pnl,
        "total_bought_cash": abs(total_day_bought),
        "total_sold_cash": total_day_sold,
        "total_fees": total_fees_day,
        "total_dividends": total_dividends_day,
        "total_margin_interest": total_margin_day,
        "total_opt_spent": abs(total_opt_spent),
        "total_opt_received": total_opt_received,
        "net_cash_flow": net_cash_flow,
    }


# ---------------------------------------------------------------------------
# Monthly P&L for charting
# ---------------------------------------------------------------------------

def monthly_pnl(df: pd.DataFrame) -> pd.DataFrame:
    stock_pnl = compute_realized_pnl(df)
    sells = stock_pnl[stock_pnl["Action"] == "SELL"].copy()
    sells["Month"] = sells["TradeDate"].dt.to_period("M")
    monthly_stock = sells.groupby("Month")["RealizedPnL"].sum().reset_index()
    monthly_stock.columns = ["Month", "StockPnL"]

    opt_pnl = compute_option_pnl(df)
    if not opt_pnl.empty:
        closed_opts = opt_pnl[opt_pnl["CloseDate"].notna()].copy()
        closed_opts["Month"] = pd.to_datetime(closed_opts["CloseDate"]).dt.to_period("M")
        monthly_opt = closed_opts.groupby("Month")["RealizedPnL"].sum().reset_index()
        monthly_opt.columns = ["Month", "OptionPnL"]
    else:
        monthly_opt = pd.DataFrame(columns=["Month", "OptionPnL"])

    divs = get_dividends(df).copy()
    divs["Month"] = divs["TradeDate"].dt.to_period("M")
    monthly_div = divs.groupby("Month")["Amount"].sum().reset_index()
    monthly_div.columns = ["Month", "Dividends"]

    interest = get_margin_interest(df).copy()
    interest["Month"] = interest["TradeDate"].dt.to_period("M")
    monthly_int = interest.groupby("Month")["Amount"].sum().reset_index()
    monthly_int.columns = ["Month", "MarginInterest"]

    merged = monthly_stock.merge(monthly_opt, on="Month", how="outer")
    merged = merged.merge(monthly_div, on="Month", how="outer")
    merged = merged.merge(monthly_int, on="Month", how="outer")
    merged = merged.fillna(0.0)
    merged["TotalPnL"] = (
        merged["StockPnL"] + merged["OptionPnL"]
        + merged["Dividends"] + merged["MarginInterest"]
    )
    merged = merged.sort_values("Month")
    merged["Month"] = merged["Month"].astype(str)
    return merged


# ---------------------------------------------------------------------------
# Daily P&L (for calendar view)
# ---------------------------------------------------------------------------

def compute_daily_pnl(
    df: pd.DataFrame,
    stock_pnl: pd.DataFrame | None = None,
    opt_pnl: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per active trading day with P&L broken down by type."""
    sp = stock_pnl if stock_pnl is not None else compute_realized_pnl(df)
    op = opt_pnl if opt_pnl is not None else compute_option_pnl(df)

    sells = sp[sp["Action"] == "SELL"].copy()
    sells["_d"] = sells["TradeDate"].dt.date
    daily_stock = sells.groupby("_d")["RealizedPnL"].sum().rename("StockPnL")

    if not op.empty:
        closed = op[op["CloseDate"].notna()].copy()
        closed["_d"] = pd.to_datetime(closed["CloseDate"]).dt.date
        daily_opt = closed.groupby("_d")["RealizedPnL"].sum().rename("OptionPnL")
    else:
        daily_opt = pd.Series(dtype=float, name="OptionPnL")

    divs = get_dividends(df).copy()
    divs["_d"] = divs["TradeDate"].dt.date
    daily_div = divs.groupby("_d")["Amount"].sum().rename("Dividends")

    merged = (
        pd.concat([daily_stock, daily_opt, daily_div], axis=1)
        .fillna(0.0)
        .reset_index()
        .rename(columns={"_d": "Date"})
    )
    merged["TotalPnL"] = merged["StockPnL"] + merged["OptionPnL"] + merged["Dividends"]
    merged["Date"] = pd.to_datetime(merged["Date"])
    return merged.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Account overview
# ---------------------------------------------------------------------------

def account_overview(df: pd.DataFrame) -> dict:
    stock_pnl = compute_realized_pnl(df)
    total_stock_pnl = stock_pnl[stock_pnl["Action"] == "SELL"]["RealizedPnL"].sum()

    opt_pnl = compute_option_pnl(df)
    total_opt_pnl = (
        opt_pnl[opt_pnl["RealizedPnL"].notna()]["RealizedPnL"].sum()
        if not opt_pnl.empty else 0.0
    )

    total_dividends = get_dividends(df)["Amount"].sum()
    total_margin = get_margin_interest(df)["Amount"].sum()
    total_credit = get_credit_interest(df)["Amount"].sum()
    total_fees = df["Fee"].sum()
    total_deposits = get_cash_deposits(df)["Amount"].sum()

    return {
        "total_stock_pnl": total_stock_pnl,
        "total_option_pnl": total_opt_pnl,
        "total_dividends": total_dividends,
        "total_margin_interest": total_margin,
        "total_credit_interest": total_credit,
        "total_fees": total_fees,
        "total_deposits": total_deposits,
        "net_pnl": total_stock_pnl + total_opt_pnl + total_dividends + total_margin + total_credit - total_fees,
    }


# ---------------------------------------------------------------------------
# Performance Analysis helpers
# ---------------------------------------------------------------------------

def _max_consecutive(series: pd.Series, target: int) -> int:
    """Count the longest run of `target` values in a boolean-like Series."""
    best = cur = 0
    for v in series:
        if v == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def build_unified_trade_log(
    stock_pnl: pd.DataFrame,
    opt_pnl: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine stock sells and closed option contracts into one trade log.
    Columns: CloseDate, Symbol, TradeType, RealizedPnL, AmountDeployed,
             HoldDays, Outcome
    """
    rows = []

    # ---- stocks -----------------------------------------------------------
    sells = stock_pnl[stock_pnl["Action"] == "SELL"].copy()

    # FIFO hold-time: for each symbol maintain a queue of (buy_date, qty)
    buy_queue: dict[str, list] = {}
    for _, r in stock_pnl.sort_values("TradeDate").iterrows():
        sym = r["Symbol"]
        if sym not in buy_queue:
            buy_queue[sym] = []
        if r["Action"] == "BUY" and r["Quantity"] > 0:
            buy_queue[sym].append({"date": r["TradeDate"], "qty": r["Quantity"]})

    # reset queue for second pass
    buy_queue2: dict[str, list] = {}
    for _, r in stock_pnl.sort_values("TradeDate").iterrows():
        sym = r["Symbol"]
        if sym not in buy_queue2:
            buy_queue2[sym] = []
        if r["Action"] == "BUY" and r["Quantity"] > 0:
            buy_queue2[sym].append({"date": r["TradeDate"], "qty": r["Quantity"]})
        elif r["Action"] == "SELL":
            remaining = abs(r["Quantity"])
            w_days, consumed = 0.0, 0.0
            q = buy_queue2.get(sym, [])
            while remaining > 0 and q:
                b = q[0]
                fill = min(remaining, b["qty"])
                w_days += ((r["TradeDate"] - b["date"]).days) * fill
                consumed += fill
                b["qty"] -= fill
                remaining -= fill
                if b["qty"] <= 0:
                    q.pop(0)
            hold = w_days / consumed if consumed > 0 else 0
            deployed = abs(r["Quantity"]) * r["AvgCostBasis"]
            rows.append({
                "CloseDate":      r["TradeDate"],
                "Symbol":         sym,
                "TradeType":      "Stock",
                "RealizedPnL":    r["RealizedPnL"],
                "AmountDeployed": deployed,
                "HoldDays":       hold,
                "Outcome":        "Win" if r["RealizedPnL"] > 0 else "Loss",
            })

    # ---- options ----------------------------------------------------------
    closed_opts = opt_pnl[opt_pnl["Outcome"].isin(
        ["Closed", "Expired Worthless", "Exercised"]
    )].copy()
    for _, r in closed_opts.iterrows():
        open_dt  = pd.to_datetime(r["OpenDate"])
        close_dt = pd.to_datetime(r["CloseDate"])
        hold = (close_dt - open_dt).days if pd.notna(open_dt) and pd.notna(close_dt) else 0
        rows.append({
            "CloseDate":      close_dt,
            "Symbol":         r["Underlying"],
            "TradeType":      "Option",
            "RealizedPnL":    r["RealizedPnL"],
            "AmountDeployed": r["OpenCost"],
            "HoldDays":       hold,
            "Outcome":        r["Outcome"] if r["Outcome"] != "Closed"
                              else ("Win" if r["RealizedPnL"] > 0 else "Loss"),
        })

    tlog = pd.DataFrame(rows)
    if tlog.empty:
        return tlog
    tlog["CloseDate"] = pd.to_datetime(tlog["CloseDate"])
    tlog["DayOfWeek"] = tlog["CloseDate"].dt.day_name()
    tlog["Month"]     = tlog["CloseDate"].dt.month
    tlog["Year"]      = tlog["CloseDate"].dt.year
    tlog["DayOfWeekN"] = tlog["CloseDate"].dt.dayofweek   # 0=Mon
    return tlog.sort_values("CloseDate").reset_index(drop=True)


def compute_core_stats(tlog: pd.DataFrame) -> dict:
    """Key performance metrics across all closed trades."""
    if tlog.empty:
        return {}

    closed = tlog[tlog["Outcome"].isin(["Win", "Loss"])].copy()
    wins   = closed[closed["Outcome"] == "Win"]
    losses = closed[closed["Outcome"] == "Loss"]

    n_total   = len(closed)
    n_wins    = len(wins)
    n_losses  = len(losses)
    win_rate  = n_wins / n_total * 100 if n_total else 0

    avg_win   = wins["RealizedPnL"].mean()   if n_wins   else 0.0
    avg_loss  = losses["RealizedPnL"].mean() if n_losses else 0.0  # negative
    gross_win = wins["RealizedPnL"].sum()
    gross_loss= losses["RealizedPnL"].sum()  # negative

    profit_factor = abs(gross_win / gross_loss) if gross_loss != 0 else float("inf")
    expectancy    = closed["RealizedPnL"].mean() if n_total else 0.0
    rr_ratio      = avg_win / abs(avg_loss) if avg_loss != 0 else float("inf")

    outcomes_bin = (closed["RealizedPnL"] > 0).astype(int)
    max_consec_wins   = _max_consecutive(outcomes_bin, 1)
    max_consec_losses = _max_consecutive(outcomes_bin, 0)

    avg_hold_win  = wins["HoldDays"].mean()   if n_wins   else 0.0
    avg_hold_loss = losses["HoldDays"].mean() if n_losses else 0.0

    return {
        "n_total": n_total, "n_wins": n_wins, "n_losses": n_losses,
        "win_rate": win_rate,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "gross_win": gross_win, "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "rr_ratio": rr_ratio,
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "avg_hold_win": avg_hold_win,
        "avg_hold_loss": avg_hold_loss,
        "best_trade": closed["RealizedPnL"].max(),
        "worst_trade": closed["RealizedPnL"].min(),
        "total_pnl": closed["RealizedPnL"].sum(),
    }


def rolling_win_rate(tlog: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling win rate over `window` trades."""
    closed = tlog[tlog["Outcome"].isin(["Win", "Loss"])].copy()
    closed["IsWin"] = (closed["RealizedPnL"] > 0).astype(float)
    closed["RollingWR"] = closed["IsWin"].rolling(window, min_periods=window).mean() * 100
    closed["TradeIndex"] = range(len(closed))
    return closed[["CloseDate", "TradeIndex", "RollingWR", "RealizedPnL",
                    "Symbol", "TradeType"]].dropna(subset=["RollingWR"])


def dow_month_heatmap_data(tlog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (dow_df, month_df):
      dow_df:   DayOfWeek × {AvgPnL, WinRate, Count}
      month_df: Month     × {AvgPnL, WinRate, Count}
    """
    closed = tlog[tlog["Outcome"].isin(["Win", "Loss"])].copy()
    closed["IsWin"] = (closed["RealizedPnL"] > 0).astype(float)

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    dow_df = (
        closed.groupby("DayOfWeek")
        .agg(AvgPnL=("RealizedPnL", "mean"),
             TotalPnL=("RealizedPnL", "sum"),
             WinRate=("IsWin", "mean"),
             Count=("RealizedPnL", "count"))
        .reindex(dow_order)
        .reset_index()
    )
    dow_df["WinRate"] *= 100

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    month_df = (
        closed.groupby("Month")
        .agg(AvgPnL=("RealizedPnL", "mean"),
             TotalPnL=("RealizedPnL", "sum"),
             WinRate=("IsWin", "mean"),
             Count=("RealizedPnL", "count"))
        .reset_index()
    )
    month_df["WinRate"] *= 100
    month_df["MonthName"] = month_df["Month"].apply(lambda m: month_names[m - 1])
    return dow_df, month_df


def symbol_ranking(
    tlog: pd.DataFrame,
    opt_pnl: pd.DataFrame,
) -> pd.DataFrame:
    """Per-symbol stats combining stocks and options."""
    closed = tlog[tlog["Outcome"].isin(["Win", "Loss"])].copy()

    def stats(g):
        wins   = g[g["RealizedPnL"] > 0]
        losses = g[g["RealizedPnL"] < 0]
        gl     = losses["RealizedPnL"].sum()
        gw     = wins["RealizedPnL"].sum()
        pf     = abs(gw / gl) if gl != 0 else float("inf")
        return pd.Series({
            "Trades":         len(g),
            "WinRate":        len(wins) / len(g) * 100,
            "TotalPnL":       g["RealizedPnL"].sum(),
            "AvgPnL":         g["RealizedPnL"].mean(),
            "AvgWin":         wins["RealizedPnL"].mean() if len(wins) else 0,
            "AvgLoss":        losses["RealizedPnL"].mean() if len(losses) else 0,
            "ProfitFactor":   pf,
            "BestTrade":      g["RealizedPnL"].max(),
            "WorstTrade":     g["RealizedPnL"].min(),
            "AvgHoldDays":    g["HoldDays"].mean(),
        })

    ranking = closed.groupby("Symbol").apply(stats, include_groups=False).reset_index()
    ranking = ranking.sort_values("TotalPnL", ascending=False)
    return ranking


# ---------------------------------------------------------------------------
# Equity curve & drawdown
# ---------------------------------------------------------------------------

def compute_equity_curve(daily_pnl: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative P&L equity curve and drawdown from daily_pnl.
    Returns DataFrame with: Date, TotalPnL, CumPnL, HWM, Drawdown, DrawdownPct
    """
    curve = daily_pnl[["Date", "TotalPnL", "StockPnL", "OptionPnL", "Dividends"]].copy()
    curve = curve.sort_values("Date").reset_index(drop=True)
    curve["CumPnL"] = curve["TotalPnL"].cumsum()
    curve["HWM"] = curve["CumPnL"].cummax()          # high-water mark
    curve["Drawdown"] = curve["CumPnL"] - curve["HWM"]  # <= 0
    curve["DrawdownPct"] = 0.0
    mask = curve["HWM"] > 0
    curve.loc[mask, "DrawdownPct"] = (
        curve.loc[mask, "Drawdown"] / curve.loc[mask, "HWM"] * 100
    )
    return curve


def max_drawdown_stats(curve: pd.DataFrame) -> dict:
    """Return key drawdown statistics from the equity curve."""
    if curve.empty:
        return {}
    max_dd = curve["Drawdown"].min()
    max_dd_pct = curve["DrawdownPct"].min()

    # Longest consecutive drawdown duration (in days)
    max_dur = 0
    cur_start = None
    for _, row in curve.iterrows():
        if row["Drawdown"] < 0:
            if cur_start is None:
                cur_start = row["Date"]
            dur = (row["Date"] - cur_start).days + 1
            max_dur = max(max_dur, dur)
        else:
            cur_start = None

    return {
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_duration_days": max_dur,
        "current_drawdown": curve["Drawdown"].iloc[-1],
        "total_pnl": curve["CumPnL"].iloc[-1],
    }


# ---------------------------------------------------------------------------
# Open positions
# ---------------------------------------------------------------------------

def get_open_positions(stock_pnl: pd.DataFrame) -> pd.DataFrame:
    """
    Derive currently open stock positions (RemainingShares > 0) from the
    realized P&L DataFrame.  Returns Symbol, Shares, AvgCostBasis, TotalCost.
    """
    if stock_pnl.empty:
        return pd.DataFrame()

    latest = (
        stock_pnl
        .sort_values("TradeDate")
        .groupby("Symbol")
        .last()
        .reset_index()
    )
    open_pos = latest[latest["RemainingShares"] > 0][
        ["Symbol", "RemainingShares", "AvgCostBasis", "TradeDate"]
    ].copy()
    open_pos.columns = ["Symbol", "Shares", "AvgCostBasis", "LastTradeDate"]
    open_pos["TotalCost"] = open_pos["Shares"] * open_pos["AvgCostBasis"]
    open_pos = open_pos.sort_values("TotalCost", ascending=False).reset_index(drop=True)
    return open_pos


# ---------------------------------------------------------------------------
# Tax summary
# ---------------------------------------------------------------------------

def compute_tax_summary(trade_log: pd.DataFrame) -> dict:
    """
    Classify closed trades as short-term (<365 days hold) or long-term (≥365 days).
    Returns dict with stats and DataFrames for each class.
    """
    closed = trade_log[trade_log["Outcome"].isin(["Win", "Loss"])].copy()
    st_df = closed[closed["HoldDays"] < 365].copy()
    lt_df = closed[closed["HoldDays"] >= 365].copy()

    def _stats(g: pd.DataFrame) -> dict:
        if g.empty:
            return {"count": 0, "gains": 0.0, "losses": 0.0, "net": 0.0}
        gains  = g[g["RealizedPnL"] > 0]["RealizedPnL"].sum()
        losses = g[g["RealizedPnL"] < 0]["RealizedPnL"].sum()
        return {
            "count":  len(g),
            "gains":  gains,
            "losses": losses,
            "net":    g["RealizedPnL"].sum(),
        }

    return {
        "short_term":    _stats(st_df),
        "long_term":     _stats(lt_df),
        "short_term_df": st_df,
        "long_term_df":  lt_df,
    }


# ---------------------------------------------------------------------------
# SPY benchmark comparison
# ---------------------------------------------------------------------------

def fetch_spy_comparison(
    deposits_df: pd.DataFrame,
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Simulate investing each ACH deposit into SPY on the deposit date.
    Returns DataFrame with Date, SPYValue, TotalDeposited, SPYGain columns
    representing the hypothetical SPY portfolio value over time.
    """
    try:
        import yfinance as yf

        start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end_str   = pd.Timestamp(end_date).strftime("%Y-%m-%d")

        raw = yf.download("SPY", start=start_str, end=end_str,
                          progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()

        # Handle multi-level column (yfinance may return MultiIndex)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        spy_close = raw["Close"].squeeze()
        spy_close.index = pd.to_datetime(spy_close.index).normalize()

        # Deposits (ACH only, positive amounts)
        deps = deposits_df[deposits_df["Amount"] > 0].copy()
        deps["TradeDate"] = pd.to_datetime(deps["TradeDate"]).dt.normalize()
        deps = deps.sort_values("TradeDate")

        spy_shares    = 0.0
        total_deposited = 0.0
        dep_records   = list(deps[["TradeDate", "Amount"]].itertuples(index=False))
        dep_idx       = 0
        n_deps        = len(dep_records)

        records = []
        for date, price_val in spy_close.items():
            date_norm = pd.Timestamp(date).normalize()
            price = float(price_val)

            # Apply all deposits on or before this trading day
            while dep_idx < n_deps and dep_records[dep_idx].TradeDate <= date_norm:
                dep_amount = dep_records[dep_idx].Amount
                # Find nearest SPY price >= deposit date
                avail = spy_close.index[spy_close.index >= dep_records[dep_idx].TradeDate]
                if len(avail):
                    buy_price = float(spy_close[avail[0]])
                    spy_shares    += dep_amount / buy_price
                    total_deposited += dep_amount
                dep_idx += 1

            records.append({
                "Date":           date_norm,
                "SPYPrice":       price,
                "SPYShares":      spy_shares,
                "SPYValue":       spy_shares * price,
                "TotalDeposited": total_deposited,
                "SPYGain":        spy_shares * price - total_deposited,
            })

        result = pd.DataFrame(records)
        result["Date"] = pd.to_datetime(result["Date"])
        return result

    except Exception:
        return pd.DataFrame()


def options_deep_dive(opt_pnl: pd.DataFrame) -> dict:
    """Options-specific analytics."""
    if opt_pnl.empty:
        return {}

    opt = opt_pnl.copy()
    opt["OpenDate"]  = pd.to_datetime(opt["OpenDate"])
    opt["ExpiryDate"] = pd.to_datetime(
        opt["OptionExpiry"], format="%m/%d/%y", errors="coerce"
    )
    opt["DTE"] = (opt["ExpiryDate"] - opt["OpenDate"]).dt.days

    closed   = opt[opt["Outcome"].isin(["Closed", "Expired Worthless", "Exercised"])]
    won      = closed[closed["RealizedPnL"] > 0]
    lost     = closed[closed["RealizedPnL"] <= 0]
    expired  = opt[opt["Outcome"] == "Expired Worthless"]
    still_open = opt[opt["Outcome"] == "Still Open"]

    # Premium capture rate on winning trades
    cap_rate = (
        (won["CloseProceeds"].fillna(0) / won["OpenCost"]).replace([float("inf")], 0).mean() * 100
        if not won.empty else 0.0
    )

    # DTE buckets
    dte_bins   = [0, 3, 7, 14, 30, 60, 120, 999]
    dte_labels = ["0-3d","4-7d","8-14d","15-30d","31-60d","61-120d","120d+"]
    opt["DTEBucket"] = pd.cut(
        opt["DTE"].fillna(0), bins=dte_bins, labels=dte_labels, right=True
    )
    dte_stats = (
        opt[opt["Outcome"].isin(["Closed","Expired Worthless","Exercised"])]
        .groupby("DTEBucket", observed=True)
        .agg(
            Count=("RealizedPnL", "count"),
            TotalPnL=("RealizedPnL", "sum"),
            WinRate=("RealizedPnL", lambda x: (x > 0).mean() * 100),
            AvgPnL=("RealizedPnL", "mean"),
        )
        .reset_index()
    )

    return {
        "n_expired":           len(expired),
        "lost_to_expiry":      expired["RealizedPnL"].sum(),
        "premium_capture_pct": cap_rate,
        "avg_dte_winners":     won["DTE"].mean() if not won.empty else 0,
        "avg_dte_losers":      lost["DTE"].mean() if not lost.empty else 0,
        "dte_stats":           dte_stats,
        "opt_with_dte":        opt[opt["DTE"].notna()].copy(),
        "still_open_risk":     still_open["OpenCost"].sum(),
    }
