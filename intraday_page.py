"""Intra-Day Trading page — imported and called from app.py."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Helper: detect same-day round-trip trades
# ---------------------------------------------------------------------------

def compute_day_trades(df: pd.DataFrame, opt_pnl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect same-day round-trip trades (buy + sell same symbol same calendar date).

    Stock detection: group raw df (RecordType==Trade & ~IsOption) by (Symbol, TradeDate).
    A group qualifies if it contains both BUY and SELL rows.
      - day_qty  = min(sum_buys, abs(sum_sells))
      - avg_buy  = weighted average buy price
      - avg_sell = weighted average sell price
      - pnl      = day_qty * (avg_sell - avg_buy)

    Option detection: opt_pnl_df rows where OpenDate == CloseDate and RealizedPnL is not null.

    Returns DataFrame with columns:
        TradeDate, Symbol, Type, DayQty, AvgBuy, AvgSell, RealizedPnL, Outcome
    """
    records = []

    # ── Stock round-trips ────────────────────────────────────────────────────
    stock_trades = df[
        (df["RecordType"] == "Trade") & (~df["IsOption"])
    ].copy()

    for (sym, date), grp in stock_trades.groupby(["Symbol", "TradeDate"]):
        buys  = grp[grp["Action"].str.upper() == "BUY"]
        sells = grp[grp["Action"].str.upper() == "SELL"]

        if buys.empty or sells.empty:
            continue

        sum_buys  = buys["Quantity"].sum()           # positive
        sum_sells = abs(sells["Quantity"].sum())      # magnitude of negative qty

        if sum_buys <= 0 or sum_sells <= 0:
            continue

        day_qty  = min(sum_buys, sum_sells)
        avg_buy  = (buys["Quantity"] * buys["Price"]).sum() / sum_buys
        avg_sell = (abs(sells["Quantity"]) * sells["Price"]).sum() / sum_sells
        pnl      = day_qty * (avg_sell - avg_buy)

        records.append({
            "TradeDate":   date,
            "Symbol":      sym,
            "Type":        "Stock",
            "DayQty":      day_qty,
            "AvgBuy":      avg_buy,
            "AvgSell":     avg_sell,
            "RealizedPnL": pnl,
            "Outcome":     "Win" if pnl > 0 else ("Loss" if pnl < 0 else "Break-Even"),
        })

    # ── Option round-trips ───────────────────────────────────────────────────
    if not opt_pnl_df.empty and "OpenDate" in opt_pnl_df.columns:
        opt_day = opt_pnl_df[
            (opt_pnl_df["OpenDate"] == opt_pnl_df["CloseDate"])
            & opt_pnl_df["RealizedPnL"].notna()
        ].copy()

        for _, row in opt_day.iterrows():
            pnl       = row["RealizedPnL"]
            sym       = row.get("Underlying", "")
            contracts = row.get("Contracts", 1)
            open_cost = row.get("OpenCost", 0.0)
            proceeds  = row.get("CloseProceeds", 0.0)

            records.append({
                "TradeDate":   row["CloseDate"],
                "Symbol":      sym,
                "Type":        "Option",
                "DayQty":      contracts,
                "AvgBuy":      open_cost / max(contracts, 1),
                "AvgSell":     proceeds / max(contracts, 1),
                "RealizedPnL": pnl,
                "Outcome":     "Win" if pnl > 0 else ("Loss" if pnl < 0 else "Break-Even"),
            })

    if not records:
        return pd.DataFrame(columns=[
            "TradeDate", "Symbol", "Type", "DayQty",
            "AvgBuy", "AvgSell", "RealizedPnL", "Outcome",
        ])

    result = pd.DataFrame(records)
    result["TradeDate"] = pd.to_datetime(result["TradeDate"])
    return result.sort_values("TradeDate").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------

def render(df: pd.DataFrame, stock_pnl_df: pd.DataFrame,
           opt_pnl_df: pd.DataFrame, fmt) -> None:
    st.title("Intra-Day Trading")
    st.caption(
        "Same-day round trips: symbols where both a buy and a sell occurred on the same "
        "calendar date. Since the CSV contains dates only (no timestamps), all same-day "
        "buy+sell pairs are classified as intra-day trades."
    )

    day_trades = compute_day_trades(df, opt_pnl_df)

    if day_trades.empty:
        st.info("No same-day round-trip trades found in your data.")
        return

    # ════════════════════════════════════════════════════════════════════════
    # 1. Overview KPIs
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Overview")

    total   = len(day_trades)
    wins    = (day_trades["Outcome"] == "Win").sum()
    wr      = wins / total * 100 if total else 0.0
    tot_pnl = day_trades["RealizedPnL"].sum()
    avg_pnl = day_trades["RealizedPnL"].mean() if total else 0.0
    best    = day_trades["RealizedPnL"].max()
    worst   = day_trades["RealizedPnL"].min()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Day Trades", total)
    c2.metric("Win Rate",          f"{wr:.1f}%")
    c3.metric("Total P&L",         fmt(tot_pnl), delta=f"{tot_pnl:+,.2f}")
    c4.metric("Avg P&L / Trade",   fmt(avg_pnl), delta=f"{avg_pnl:+,.2f}")
    c5.metric("Best Day Trade",    fmt(best))
    c6.metric("Worst Day Trade",   fmt(worst))

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # 2. Day Trades vs Swing Trades
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Day Trades vs Swing Trades")

    # Build lookup set of (Symbol, TradeDate) keys for stock day trades
    dt_stock_rows = day_trades[day_trades["Type"] == "Stock"]
    dt_stock_keys = set(zip(dt_stock_rows["Symbol"], dt_stock_rows["TradeDate"]))

    # Stock sells from stock_pnl_df
    if not stock_pnl_df.empty and "Action" in stock_pnl_df.columns:
        stock_sells = stock_pnl_df[
            stock_pnl_df["Action"].str.upper() == "SELL"
        ].copy()
        if not stock_sells.empty:
            stock_sells["_is_day"] = stock_sells.apply(
                lambda r: (r["Symbol"], r["TradeDate"]) in dt_stock_keys, axis=1
            )
            st_day   = stock_sells[stock_sells["_is_day"]]
            st_swing = stock_sells[~stock_sells["_is_day"]]
        else:
            st_day   = pd.DataFrame()
            st_swing = pd.DataFrame()
    else:
        st_day   = pd.DataFrame()
        st_swing = pd.DataFrame()

    # Option trades split by same-day vs multi-day
    if not opt_pnl_df.empty and "OpenDate" in opt_pnl_df.columns:
        opt_closed = opt_pnl_df[opt_pnl_df["RealizedPnL"].notna()].copy()
        opt_day    = opt_closed[opt_closed["OpenDate"] == opt_closed["CloseDate"]]
        opt_swing  = opt_closed[opt_closed["OpenDate"] != opt_closed["CloseDate"]]
    else:
        opt_day   = pd.DataFrame()
        opt_swing = pd.DataFrame()

    def _stats(sub: pd.DataFrame) -> dict:
        if sub.empty or "RealizedPnL" not in sub.columns:
            return {"count": 0, "total_pnl": 0.0, "win_rate": 0.0}
        n    = len(sub)
        wins_s = (sub["RealizedPnL"] > 0).sum()
        return {
            "count":     n,
            "total_pnl": sub["RealizedPnL"].sum(),
            "win_rate":  wins_s / n * 100,
        }

    s_day_stats   = _stats(st_day)
    s_swing_stats = _stats(st_swing)
    o_day_stats   = _stats(opt_day)
    o_swing_stats = _stats(opt_swing)

    col_sd, col_ss, col_od, col_os = st.columns(4)

    with col_sd:
        st.markdown("**Stock — Day Trades**")
        st.metric("Count",     s_day_stats["count"])
        st.metric("Total P&L", fmt(s_day_stats["total_pnl"]))
        st.metric("Win Rate",  f"{s_day_stats['win_rate']:.1f}%")

    with col_ss:
        st.markdown("**Stock — Swing Trades**")
        st.metric("Count",     s_swing_stats["count"])
        st.metric("Total P&L", fmt(s_swing_stats["total_pnl"]))
        st.metric("Win Rate",  f"{s_swing_stats['win_rate']:.1f}%")

    with col_od:
        st.markdown("**Option — Day Trades**")
        st.metric("Count",     o_day_stats["count"])
        st.metric("Total P&L", fmt(o_day_stats["total_pnl"]))
        st.metric("Win Rate",  f"{o_day_stats['win_rate']:.1f}%")

    with col_os:
        st.markdown("**Option — Swing Trades**")
        st.metric("Count",     o_swing_stats["count"])
        st.metric("Total P&L", fmt(o_swing_stats["total_pnl"]))
        st.metric("Win Rate",  f"{o_swing_stats['win_rate']:.1f}%")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # 3. Frequency & P&L Over Time
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Frequency & P&L Over Time")

    monthly = (
        day_trades.assign(Month=day_trades["TradeDate"].dt.to_period("M"))
        .groupby("Month")
        .agg(Count=("RealizedPnL", "count"), TotalPnL=("RealizedPnL", "sum"))
        .reset_index()
    )
    monthly["MonthStr"] = monthly["Month"].astype(str)

    pnl_marker_colors = [
        "#00c97a" if v >= 0 else "#ff4b4b" for v in monthly["TotalPnL"]
    ]

    fig_freq = go.Figure()
    fig_freq.add_trace(go.Bar(
        x=monthly["MonthStr"],
        y=monthly["Count"],
        name="Trade Count",
        marker_color="#4C9BE8",
        yaxis="y",
        hovertemplate="<b>%{x}</b><br>Trades: %{y}<extra></extra>",
    ))
    fig_freq.add_trace(go.Scatter(
        x=monthly["MonthStr"],
        y=monthly["TotalPnL"],
        name="Total P&L",
        mode="lines+markers",
        line=dict(color="#F4A460", width=2),
        marker=dict(color=pnl_marker_colors, size=8),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>",
    ))
    fig_freq.update_layout(
        title="Monthly Day-Trade Frequency & P&L",
        yaxis=dict(title="# Trades"),
        yaxis2=dict(title="Total P&L ($)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
        height=360,
        margin=dict(t=50),
    )
    st.plotly_chart(fig_freq, width='stretch')

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # 4. Symbol Breakdown
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Symbol Breakdown")

    sym_grp = (
        day_trades.groupby("Symbol")
        .agg(
            Trades=("RealizedPnL", "count"),
            WinRate=("Outcome", lambda x: (x == "Win").mean() * 100),
            TotalPnL=("RealizedPnL", "sum"),
            AvgPnL=("RealizedPnL", "mean"),
            Best=("RealizedPnL", "max"),
            Worst=("RealizedPnL", "min"),
        )
        .reset_index()
        .sort_values("TotalPnL", ascending=False)
    )

    bar_clr = ["#00c97a" if v >= 0 else "#ff4b4b" for v in sym_grp["TotalPnL"]]
    fig_sym = go.Figure(go.Bar(
        x=sym_grp["Symbol"],
        y=sym_grp["TotalPnL"],
        marker_color=bar_clr,
        text=sym_grp["TotalPnL"].map(lambda v: f"${v:,.2f}"),
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Total P&L: $%{y:,.2f}<br>"
            "Trades: %{customdata[0]}<br>"
            "Win Rate: %{customdata[1]:.1f}%<extra></extra>"
        ),
        customdata=sym_grp[["Trades", "WinRate"]].values,
    ))
    fig_sym.update_layout(
        title="Total Day-Trade P&L by Symbol",
        yaxis_title="Total P&L ($)",
        height=360,
        margin=dict(t=40),
    )
    st.plotly_chart(fig_sym, width='stretch')

    # Symbol summary table
    sym_disp = sym_grp.copy()
    sym_disp["WinRate"]  = sym_disp["WinRate"].map(lambda v: f"{v:.1f}%")
    sym_disp["TotalPnL"] = sym_disp["TotalPnL"].map(lambda v: f"${v:,.2f}")
    sym_disp["AvgPnL"]   = sym_disp["AvgPnL"].map(lambda v: f"${v:,.2f}")
    sym_disp["Best"]     = sym_disp["Best"].map(lambda v: f"${v:,.2f}")
    sym_disp["Worst"]    = sym_disp["Worst"].map(lambda v: f"${v:,.2f}")
    sym_disp = sym_disp.rename(columns={
        "WinRate": "Win Rate", "TotalPnL": "Total P&L",
        "AvgPnL": "Avg P&L",
    })
    st.dataframe(sym_disp, width='stretch', hide_index=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # 5. Day Trades Log
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Day Trades Log")

    with st.expander("Filters", expanded=False):
        f1, f2 = st.columns(2)
        with f1:
            sym_opts   = sorted(day_trades["Symbol"].unique().tolist())
            sym_filter = st.multiselect("Symbol", sym_opts)
        with f2:
            type_filter = st.multiselect(
                "Type", ["Stock", "Option"], default=["Stock", "Option"]
            )

    log = day_trades.copy().reset_index(drop=True)
    if sym_filter:
        log = log[log["Symbol"].isin(sym_filter)].reset_index(drop=True)
    if type_filter:
        log = log[log["Type"].isin(type_filter)].reset_index(drop=True)

    # ── By-day summary table ─────────────────────────────────────────────────
    daily_summary = (
        log.groupby(log["TradeDate"].dt.date)
        .agg(
            Trades   =("RealizedPnL", "count"),
            Wins     =("Outcome",     lambda x: (x == "Win").sum()),
            Losses   =("Outcome",     lambda x: (x == "Loss").sum()),
            WinRate  =("Outcome",     lambda x: (x == "Win").mean() * 100),
            TotalPnL =("RealizedPnL", "sum"),
            Best     =("RealizedPnL", "max"),
            Worst    =("RealizedPnL", "min"),
        )
        .reset_index()
        .rename(columns={"TradeDate": "Date"})
        .sort_values("Date", ascending=False)
    )
    raw_day_pnl = daily_summary["TotalPnL"].values
    day_disp = daily_summary.copy()
    day_disp["Date"]     = day_disp["Date"].astype(str)
    day_disp["WinRate"]  = day_disp["WinRate"].map(lambda v: f"{v:.0f}%")
    day_disp["TotalPnL"] = day_disp["TotalPnL"].map(lambda v: f"${v:,.2f}")
    day_disp["Best"]     = day_disp["Best"].map(lambda v: f"${v:,.2f}")
    day_disp["Worst"]    = day_disp["Worst"].map(lambda v: f"${v:,.2f}")
    day_disp = day_disp.rename(columns={
        "WinRate": "Win Rate", "TotalPnL": "Total P&L",
    })

    def _color_day_pnl(col):
        return [
            ("color: #00c97a; font-weight: 600" if v > 0
             else "color: #ff4b4b; font-weight: 600" if v < 0
             else "")
            for v in raw_day_pnl
        ]

    st.dataframe(
        day_disp.style.apply(_color_day_pnl, subset=["Total P&L"], axis=0),
        width='stretch',
        hide_index=True,
    )

    st.markdown("**Individual Trades**")

    # Capture raw P&L floats for coloring before formatting
    raw_pnl = log["RealizedPnL"].values

    log_disp = log.copy()
    log_disp["TradeDate"]   = log_disp["TradeDate"].dt.strftime("%Y-%m-%d")
    log_disp["DayQty"]      = log_disp["DayQty"].map(lambda v: f"{v:,.0f}")
    log_disp["AvgBuy"]      = log_disp["AvgBuy"].map(lambda v: f"${v:,.4f}")
    log_disp["AvgSell"]     = log_disp["AvgSell"].map(lambda v: f"${v:,.4f}")
    log_disp["RealizedPnL"] = log_disp["RealizedPnL"].map(lambda v: f"${v:,.2f}")
    log_disp = log_disp.rename(columns={
        "TradeDate":   "Date",
        "DayQty":      "Qty",
        "AvgBuy":      "Avg Buy",
        "AvgSell":     "Avg Sell",
        "RealizedPnL": "P&L",
    })

    def _color_pnl(col):
        """Color the P&L column using the parallel raw float array."""
        return [
            ("color: #00c97a; font-weight: 600" if v > 0
             else "color: #ff4b4b; font-weight: 600" if v < 0
             else "")
            for v in raw_pnl
        ]

    st.dataframe(
        log_disp.style.apply(_color_pnl, subset=["P&L"], axis=0),
        width='stretch',
        hide_index=True,
    )

    st.download_button(
        label="⬇ Download Day Trades CSV",
        data=log.to_csv(index=False).encode("utf-8"),
        file_name="day_trades.csv",
        mime="text/csv",
    )

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # 6. Selected Day Detail
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Selected Day Detail")

    day_trade_dates = sorted(day_trades["TradeDate"].dt.date.unique().tolist())
    if not day_trade_dates:
        st.info("No day-trade dates available.")
        return

    selected_date = st.date_input(
        "Select a day-trade date",
        value=day_trade_dates[-1],
        min_value=day_trade_dates[0],
        max_value=day_trade_dates[-1],
    )

    day_detail = day_trades[
        day_trades["TradeDate"].dt.date == selected_date
    ].copy()

    if day_detail.empty:
        st.info(f"No day trades recorded on {selected_date}.")
        st.caption(
            "Dates with day trades: "
            + ", ".join(str(d) for d in day_trade_dates)
        )
        return

    # Day-scoped KPIs
    d_n     = len(day_detail)
    d_wins  = (day_detail["Outcome"] == "Win").sum()
    d_wr    = d_wins / d_n * 100
    d_pnl   = day_detail["RealizedPnL"].sum()
    d_avg   = day_detail["RealizedPnL"].mean()
    d_best  = day_detail["RealizedPnL"].max()
    d_worst = day_detail["RealizedPnL"].min()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Trades That Day",  d_n)
    k2.metric("Win Rate",         f"{d_wr:.1f}%")
    k3.metric("Total P&L",        fmt(d_pnl),  delta=f"{d_pnl:+,.2f}")
    k4.metric("Avg P&L / Trade",  fmt(d_avg),  delta=f"{d_avg:+,.2f}")
    k5.metric("Best",             fmt(d_best))
    k6.metric("Worst",            fmt(d_worst))

    selected_ts = pd.Timestamp(selected_date)
    total_day   = day_detail["RealizedPnL"].sum()

    # ── Chart 1: P&L aggregated by underlying symbol ─────────────────────────
    sym_summary = (
        day_detail.groupby("Symbol")
        .agg(TotalPnL=("RealizedPnL", "sum"), Trades=("RealizedPnL", "count"))
        .reset_index()
        .sort_values("TotalPnL", ascending=True)
    )
    n_syms = len(sym_summary)
    clr1 = ["#00c97a" if v >= 0 else "#ff4b4b" for v in sym_summary["TotalPnL"]]

    fig_sym = go.Figure(go.Bar(
        orientation="h",
        y=sym_summary["Symbol"],
        x=sym_summary["TotalPnL"],
        marker_color=clr1,
        text=sym_summary["TotalPnL"].map(lambda v: f"${v:,.2f}"),
        textposition="outside",
        textfont=dict(size=12),
        customdata=sym_summary[["Trades"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>Total P&L: $%{x:,.2f}<br>"
            "Round-trips: %{customdata[0]}<extra></extra>"
        ),
    ))
    fig_sym.update_layout(
        title=f"P&L by Underlying — {selected_date}  |  Day Total: <b>${total_day:,.2f}</b>",
        xaxis=dict(title="P&L ($)", zeroline=True, zerolinecolor="rgba(255,255,255,0.3)"),
        yaxis=dict(title=""),
        bargap=0.4,
        height=max(220, n_syms * 48 + 80),
        margin=dict(l=20, r=140, t=55, b=40),
    )
    st.plotly_chart(fig_sym, width='stretch')

    # ── Chart 2: Individual contract breakdown ────────────────────────────────
    # Stocks: one round-trip per symbol already in day_detail
    stock_indiv = day_detail[day_detail["Type"] == "Stock"][
        ["Symbol", "RealizedPnL"]
    ].rename(columns={"Symbol": "Label"})

    # Options: pull contract details from opt_pnl_df; group same contract rows
    opt_indiv = pd.DataFrame(columns=["Label", "RealizedPnL"])
    if not opt_pnl_df.empty and "OpenDate" in opt_pnl_df.columns:
        opt_today = opt_pnl_df[
            (opt_pnl_df["CloseDate"] == selected_ts)
            & (opt_pnl_df["OpenDate"] == opt_pnl_df["CloseDate"])
            & opt_pnl_df["RealizedPnL"].notna()
        ].copy()
        if not opt_today.empty:
            def _opt_label(row):
                underlying = row.get("Underlying", "")
                strike     = row.get("OptionStrike", "")
                typ        = str(row.get("OptionType", ""))[:1].upper()  # C / P
                expiry     = str(row.get("OptionExpiry", ""))
                return f"{underlying} {strike}{typ} {expiry}"
            opt_today["Label"] = opt_today.apply(_opt_label, axis=1)
            opt_indiv = opt_today[["Label", "RealizedPnL"]]

    # Merge all rows then group by label so the same contract is never split
    indiv = (
        pd.concat([stock_indiv, opt_indiv], ignore_index=True)
        .groupby("Label", as_index=False)
        .agg(RealizedPnL=("RealizedPnL", "sum"))
        .sort_values("RealizedPnL", ascending=True)
    )
    n_indiv = len(indiv)
    clr2 = ["#00c97a" if v >= 0 else "#ff4b4b" for v in indiv["RealizedPnL"]]

    fig_indiv = go.Figure(go.Bar(
        orientation="h",
        y=indiv["Label"],
        x=indiv["RealizedPnL"],
        marker_color=clr2,
        text=indiv["RealizedPnL"].map(lambda v: f"${v:,.2f}"),
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate="<b>%{y}</b><br>Net P&L: $%{x:,.2f}<extra></extra>",
    ))
    fig_indiv.update_layout(
        title=f"P&L by Contract — {selected_date}",
        xaxis=dict(title="P&L ($)", zeroline=True, zerolinecolor="rgba(255,255,255,0.3)"),
        yaxis=dict(title=""),
        bargap=0.4,
        height=max(220, n_indiv * 48 + 80),
        margin=dict(l=20, r=140, t=55, b=40),
    )
    st.plotly_chart(fig_indiv, width='stretch')

    # (b) Raw trade rows for day-traded symbols on that date
    st.markdown(f"**Raw Trade Rows — {selected_date}**")
    day_syms = day_detail["Symbol"].unique().tolist()

    raw_rows = df[
        (df["TradeDate"] == selected_ts)
        & (df["Symbol"].isin(day_syms))
        & (df["RecordType"] == "Trade")
        & (~df["IsOption"])
    ].copy()

    if raw_rows.empty:
        st.caption("No raw stock trade rows found for this date/symbols.")
    else:
        display_cols = [
            c for c in
            ["Symbol", "TradeDate", "Action", "Quantity", "Price", "Amount", "Description"]
            if c in raw_rows.columns
        ]
        raw_disp = raw_rows[display_cols].copy()
        if "TradeDate" in raw_disp.columns:
            raw_disp["TradeDate"] = raw_disp["TradeDate"].dt.strftime("%Y-%m-%d")
        st.dataframe(raw_disp, width='stretch', hide_index=True)
