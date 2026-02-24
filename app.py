import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

import perf_page

from utils import (
    account_overview,
    build_unified_trade_log,
    compute_daily_pnl,
    compute_equity_curve,
    compute_option_pnl,
    compute_realized_pnl,
    compute_tax_summary,
    day_metrics,
    fetch_spy_comparison,
    get_all_option_underlyings,
    get_all_stock_symbols,
    get_cash_deposits,
    get_dividends,
    get_margin_interest,
    get_open_positions,
    load_data,
    max_drawdown_stats,
    monthly_pnl,
    options_deep_dive,
    rolling_win_rate,
    symbol_metrics,
    symbol_ranking,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FirstTrade Analysis",
    page_icon="📈",
    layout="wide",
)

DEFAULT_PATH = "FT_CSV_export.csv"

# ---------------------------------------------------------------------------
# Sidebar: navigation (file upload moved here too)
# ---------------------------------------------------------------------------

st.sidebar.title("📈 FirstTrade Analysis")

st.sidebar.markdown("**Data source**")
uploaded_file = st.sidebar.file_uploader(
    "Upload your FirstTrade CSV",
    type=["csv"],
    help="Export your trading history from FirstTrade and upload it here. "
         "If left empty the default file on disk is used.",
    label_visibility="collapsed",
)
if uploaded_file is not None:
    st.sidebar.success(f"Loaded: {uploaded_file.name}")
else:
    st.sidebar.caption("Using default file on disk")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Account Overview",
        "Symbol Analysis",
        "Daily View",
        "Symbol + Day",
        "Options Analysis",
        "Performance Analysis",
        "Open Positions",
        "Tax Summary",
    ],
)

# ---------------------------------------------------------------------------
# Data loading  (all expensive work cached together by source key)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_all_bytes(content: bytes):
    df_       = load_data(io.BytesIO(content))
    sp_       = compute_realized_pnl(df_)
    op_       = compute_option_pnl(df_)
    dp_       = compute_daily_pnl(df_, stock_pnl=sp_, opt_pnl=op_)
    tl_       = build_unified_trade_log(sp_, op_)
    return df_, sp_, op_, dp_, tl_


@st.cache_data(show_spinner=False)
def _load_all_path(path: str):
    df_       = load_data(path)
    sp_       = compute_realized_pnl(df_)
    op_       = compute_option_pnl(df_)
    dp_       = compute_daily_pnl(df_, stock_pnl=sp_, opt_pnl=op_)
    tl_       = build_unified_trade_log(sp_, op_)
    return df_, sp_, op_, dp_, tl_


with st.spinner("Loading and analysing trading data…"):
    if uploaded_file is not None:
        df, stock_pnl_df, opt_pnl_df, daily_pnl_df, trade_log_df = (
            _load_all_bytes(uploaded_file.getvalue())
        )
    else:
        df, stock_pnl_df, opt_pnl_df, daily_pnl_df, trade_log_df = (
            _load_all_path(DEFAULT_PATH)
        )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

date_min        = df["TradeDate"].min().date()
date_max        = df["TradeDate"].max().date()
all_symbols     = get_all_stock_symbols(df)
all_underlyings = get_all_option_underlyings(df)


def fmt(val: float, prefix="$") -> str:
    return f"{prefix}{val:,.2f}"


def style_pnl(val):
    if isinstance(val, (int, float)):
        color = "#00c97a" if val > 0 else ("#ff4b4b" if val < 0 else "inherit")
        return f"color: {color}; font-weight: 600"
    return ""


def csv_download(df_export: pd.DataFrame, filename: str, label="⬇ Download CSV"):
    st.download_button(
        label=label,
        data=df_export.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Page: Account Overview
# ---------------------------------------------------------------------------

if page == "Account Overview":
    st.title("Account Overview")

    ov      = account_overview(df)
    monthly = monthly_pnl(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stock Realized P&L",   fmt(ov["total_stock_pnl"]),
              delta=f"{ov['total_stock_pnl']:+,.2f}")
    c2.metric("Options Realized P&L", fmt(ov["total_option_pnl"]),
              delta=f"{ov['total_option_pnl']:+,.2f}")
    c3.metric("Dividends Earned",     fmt(ov["total_dividends"]))
    c4.metric("Net P&L (all-in)",     fmt(ov["net_pnl"]),
              delta=f"{ov['net_pnl']:+,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Margin Interest Paid",   fmt(abs(ov["total_margin_interest"])),
              delta=f"{ov['total_margin_interest']:+,.2f}", delta_color="inverse")
    c6.metric("Credit Interest Earned", fmt(ov["total_credit_interest"]))
    c7.metric("Total Fees Paid",        fmt(ov["total_fees"]))
    c8.metric("Cash Deposited (ACH)",   fmt(ov["total_deposits"]))

    st.divider()

    # ── Monthly P&L stacked bar ──────────────────────────────────────────────
    st.subheader("Monthly P&L Breakdown")
    fig = go.Figure()
    colors = {
        "StockPnL": "#4C9BE8", "OptionPnL": "#F4A460",
        "Dividends": "#00c97a", "MarginInterest": "#ff4b4b",
    }
    labels = {
        "StockPnL": "Stock P&L", "OptionPnL": "Options P&L",
        "Dividends": "Dividends", "MarginInterest": "Margin Interest",
    }
    for col in ["StockPnL", "OptionPnL", "Dividends", "MarginInterest"]:
        fig.add_trace(go.Bar(
            x=monthly["Month"], y=monthly[col],
            name=labels[col], marker_color=colors[col],
        ))
    fig.add_trace(go.Scatter(
        x=monthly["Month"], y=monthly["TotalPnL"].cumsum(),
        name="Cumulative Net P&L", mode="lines+markers",
        line=dict(color="white", width=2, dash="dot"), yaxis="y2",
    ))
    fig.update_layout(
        barmode="relative",
        yaxis=dict(title="Monthly P&L ($)"),
        yaxis2=dict(title="Cumulative P&L ($)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420, margin=dict(t=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Monthly breakdown table"):
        display = monthly.copy()
        for col in ["StockPnL", "OptionPnL", "Dividends", "MarginInterest", "TotalPnL"]:
            display[col] = display[col].map(lambda v: f"${v:,.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        csv_download(monthly, "monthly_pnl.csv")

    st.divider()

    # ── Equity Curve & Drawdown ─────────────────────────────────────────────
    st.subheader("Equity Curve & Drawdown")

    eq_curve = compute_equity_curve(daily_pnl_df)
    dd_stats  = max_drawdown_stats(eq_curve)

    if not eq_curve.empty:
        eq1, eq2, eq3 = st.columns(3)
        eq1.metric("Total Realized P&L", fmt(dd_stats.get("total_pnl", 0)))
        eq2.metric("Max Drawdown (from peak P&L)", fmt(dd_stats.get("max_drawdown", 0)),
                   delta=f"{dd_stats.get('max_drawdown', 0):+,.2f}", delta_color="inverse")
        eq3.metric("Max DD Duration", f"{dd_stats.get('max_drawdown_duration_days', 0)} calendar days")

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq_curve["Date"], y=eq_curve["CumPnL"],
            name="Cumulative P&L", mode="lines",
            line=dict(color="#4C9BE8", width=2),
        ))
        fig_eq.add_trace(go.Scatter(
            x=eq_curve["Date"], y=eq_curve["HWM"],
            name="High-Water Mark", mode="lines",
            line=dict(color="#00c97a", width=1.5, dash="dot"),
        ))
        fig_eq.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
        fig_eq.update_layout(
            yaxis_title="Cumulative P&L ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=300, margin=dict(t=40),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=eq_curve["Date"], y=eq_curve["Drawdown"],
            name="Drawdown ($)", mode="lines",
            line=dict(color="#ff4b4b", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,75,75,0.15)",
        ))
        fig_dd.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
        fig_dd.update_layout(
            yaxis_title="Drawdown from Peak ($)",
            height=200, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    st.divider()

    # ── SPY Benchmark Comparison ─────────────────────────────────────────────
    st.subheader("SPY Benchmark Comparison")
    st.caption(
        "Simulates investing each ACH deposit into SPY on the deposit date. "
        "Compares your realized P&L against a passive SPY buy-and-hold strategy."
    )

    if st.button("Load SPY Data", type="primary", key="spy_btn"):
        with st.spinner("Fetching SPY price history from Yahoo Finance…"):
            deposits_df = get_cash_deposits(df)
            spy_df = fetch_spy_comparison(deposits_df, date_min, date_max)
        if spy_df.empty:
            st.error("Could not fetch SPY data. Check your internet connection.")
        else:
            st.session_state["spy_df"] = spy_df

    if "spy_df" in st.session_state:
        spy_df = st.session_state["spy_df"]

        eq_for_spy = eq_curve[["Date", "CumPnL"]].copy()
        eq_for_spy["Date"] = pd.to_datetime(eq_for_spy["Date"])
        spy_merged = spy_df.merge(eq_for_spy, on="Date", how="left").ffill()

        total_dep = spy_df["TotalDeposited"].iloc[-1]
        spy_gain  = spy_df["SPYGain"].iloc[-1]
        your_pnl  = dd_stats.get("total_pnl", 0)

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Deposited",                   fmt(total_dep))
        sc2.metric("Your Realized P&L",                 fmt(your_pnl),
                   delta=f"{your_pnl:+,.2f}")
        sc3.metric("SPY Gain (deposits → SPY)",          fmt(spy_gain),
                   delta=f"{spy_gain:+,.2f}")
        sc4.metric("You vs SPY",                         fmt(your_pnl - spy_gain),
                   delta=f"{your_pnl - spy_gain:+,.2f}")

        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(
            x=spy_merged["Date"], y=spy_merged["CumPnL"].fillna(0),
            name="Your Realized P&L", mode="lines",
            line=dict(color="#4C9BE8", width=2),
        ))
        fig_bench.add_trace(go.Scatter(
            x=spy_merged["Date"], y=spy_merged["SPYGain"],
            name="SPY Buy-and-Hold Gain", mode="lines",
            line=dict(color="#F4A460", width=2),
        ))
        fig_bench.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
        fig_bench.update_layout(
            yaxis_title="Gain ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=320, margin=dict(t=40),
        )
        st.plotly_chart(fig_bench, use_container_width=True)
        st.caption(
            "Note: Your P&L shown is **realized only** (closed trades + dividends). "
            "Unrealized gains on open positions are not included."
        )

    st.divider()

    # ── P&L Calendar ──────────────────────────────────────────────────────────
    st.subheader("P&L Calendar")

    available_years = sorted(daily_pnl_df["Date"].dt.year.unique().tolist(), reverse=True)

    cal_col1, cal_col2 = st.columns([1, 3])
    with cal_col1:
        granularity = st.radio("View by", ["Day", "Week", "Month", "Year"])
    with cal_col2:
        include_dividends = st.checkbox("Include dividends in P&L", value=True)
        breakdown_toggle  = st.checkbox("Show stock / options breakdown in tooltip", value=True)

    cal_df = daily_pnl_df.copy()
    if not include_dividends:
        cal_df["TotalPnL"] = cal_df["StockPnL"] + cal_df["OptionPnL"]

    def pnl_bar(x_vals, y_vals, title, x_label=""):
        clrs = ["#00c97a" if v >= 0 else "#ff4b4b" for v in y_vals]
        fig_b = go.Figure(go.Bar(
            x=x_vals, y=y_vals, marker_color=clrs,
            text=[f"${v:,.0f}" for v in y_vals], textposition="outside",
        ))
        total  = sum(y_vals)
        ctotal = "#00c97a" if total >= 0 else "#ff4b4b"
        fig_b.update_layout(
            title=dict(
                text=f"{title}   <span style='color:{ctotal}'>Total: ${total:,.2f}</span>",
                font=dict(size=14),
            ),
            xaxis_title=x_label, yaxis_title="P&L ($)",
            height=380, margin=dict(t=50, b=60),
        )
        return fig_b

    if granularity == "Day":
        yr_col, _ = st.columns([1, 3])
        with yr_col:
            year = st.selectbox("Year", available_years)

        year_df    = cal_df[cal_df["Date"].dt.year == year]
        start_date = pd.Timestamp(f"{year}-01-01")
        end_date   = pd.Timestamp(f"{year}-12-31")
        all_days   = pd.date_range(start_date, end_date, freq="D")
        ref_monday = start_date - pd.Timedelta(days=start_date.dayofweek)

        grid = pd.DataFrame({"Date": all_days})
        grid = grid.merge(cal_df, on="Date", how="left")
        grid["WeekCol"] = ((grid["Date"] - ref_monday).dt.days // 7).astype(int)
        grid["DayRow"]  = grid["Date"].dt.dayofweek.astype(int)

        n_weeks = int(grid["WeekCol"].max()) + 1
        z       = np.full((7, n_weeks), np.nan)
        hover   = np.full((7, n_weeks), "", dtype=object)

        for _, row in grid.iterrows():
            wr, wc   = int(row["DayRow"]), int(row["WeekCol"])
            date_str = row["Date"].strftime("%b %d, %Y")
            pnl      = row["TotalPnL"]
            if pd.notna(pnl) and pnl != 0.0:
                z[wr, wc] = pnl
                tip = f"<b>{date_str}</b><br>Total P&L: ${pnl:,.2f}"
                if breakdown_toggle:
                    tip += (f"<br>Stock: ${row.get('StockPnL', 0):,.2f}"
                            f"<br>Options: ${row.get('OptionPnL', 0):,.2f}"
                            f"<br>Dividends: ${row.get('Dividends', 0):,.2f}")
                hover[wr, wc] = tip
            else:
                hover[wr, wc] = date_str

        month_ticks, month_labels = [], []
        for m in range(1, 13):
            fd = pd.Timestamp(f"{year}-{m:02d}-01")
            if fd > end_date:
                break
            month_ticks.append(int((fd - ref_monday).days // 7))
            month_labels.append(fd.strftime("%b"))

        abs_max = float(np.nanmax(np.abs(z))) if not np.all(np.isnan(z)) else 1.0

        fig_cal = go.Figure(go.Heatmap(
            z=z, text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorscale=[
                [0.00, "#7f0000"], [0.35, "#ff4b4b"],
                [0.50, "#2d2d3a"], [0.65, "#00c97a"], [1.00, "#004d1f"],
            ],
            zmid=0, zmin=-abs_max, zmax=abs_max,
            showscale=True,
            colorbar=dict(title="P&L ($)", thickness=14, len=0.8),
            xgap=3, ygap=3,
        ))
        fig_cal.update_layout(
            xaxis=dict(tickvals=month_ticks, ticktext=month_labels,
                       side="top", showgrid=False),
            yaxis=dict(
                tickvals=list(range(7)),
                ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                autorange="reversed", showgrid=False,
            ),
            height=240, margin=dict(t=50, b=10, l=55, r=80),
        )
        st.plotly_chart(fig_cal, use_container_width=True)

        active_days = year_df[year_df["TotalPnL"] != 0]
        win_days    = active_days[active_days["TotalPnL"] > 0]
        lose_days   = active_days[active_days["TotalPnL"] < 0]
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Active Days",  len(active_days))
        s2.metric("Winning Days", len(win_days))
        s3.metric("Losing Days",  len(lose_days))
        s4.metric("Best Day",   fmt(active_days["TotalPnL"].max()) if not active_days.empty else "—")
        s5.metric("Worst Day",  fmt(active_days["TotalPnL"].min()) if not active_days.empty else "—")

    elif granularity == "Week":
        yr_col, _ = st.columns([1, 3])
        with yr_col:
            year = st.selectbox("Year", available_years)

        year_df = cal_df[cal_df["Date"].dt.year == year].copy()
        year_df["WeekStart"] = year_df["Date"] - pd.to_timedelta(
            year_df["Date"].dt.dayofweek, unit="D"
        )
        week_grp = year_df.groupby("WeekStart").agg(
            TotalPnL=("TotalPnL", "sum"), StockPnL=("StockPnL", "sum"),
            OptionPnL=("OptionPnL", "sum"), Dividends=("Dividends", "sum"),
        ).reset_index()

        x_labels = [d.strftime("W%W\n%b %d") for d in week_grp["WeekStart"]]
        fig_wk   = pnl_bar(x_labels, week_grp["TotalPnL"].tolist(),
                           f"{year} — Weekly Realized P&L")
        if breakdown_toggle:
            for col, clr, name in [
                ("StockPnL", "#4C9BE8", "Stock"),
                ("OptionPnL", "#F4A460", "Options"),
                ("Dividends", "#aaffcc", "Dividends"),
            ]:
                fig_wk.add_trace(go.Scatter(
                    x=x_labels, y=week_grp[col], mode="lines+markers",
                    name=name, line=dict(color=clr, width=1.5, dash="dot"),
                    visible="legendonly",
                ))
        fig_wk.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_wk, use_container_width=True)

        s1, s2, s3, s4, s5 = st.columns(5)
        pos = week_grp[week_grp["TotalPnL"] > 0]
        neg = week_grp[week_grp["TotalPnL"] < 0]
        s1.metric("Profitable Weeks", len(pos))
        s2.metric("Losing Weeks",     len(neg))
        s3.metric("Best Week",  fmt(week_grp["TotalPnL"].max()))
        s4.metric("Worst Week", fmt(week_grp["TotalPnL"].min()))
        s5.metric("Avg Week",   fmt(week_grp["TotalPnL"].mean()))

    elif granularity == "Month":
        yr_opts = ["All years"] + [str(y) for y in available_years]
        yr_col, _ = st.columns([1, 3])
        with yr_col:
            yr_pick = st.selectbox("Year", yr_opts)

        month_df = cal_df.copy()
        if yr_pick != "All years":
            month_df = month_df[month_df["Date"].dt.year == int(yr_pick)]

        month_df["MonthPeriod"] = month_df["Date"].dt.to_period("M")
        mon_grp = month_df.groupby("MonthPeriod").agg(
            TotalPnL=("TotalPnL", "sum"), StockPnL=("StockPnL", "sum"),
            OptionPnL=("OptionPnL", "sum"), Dividends=("Dividends", "sum"),
        ).reset_index()
        mon_grp["MonthStr"] = mon_grp["MonthPeriod"].astype(str)

        fig_mo = pnl_bar(mon_grp["MonthStr"].tolist(), mon_grp["TotalPnL"].tolist(),
                         "Monthly Realized P&L")
        if breakdown_toggle:
            for col, clr, name in [
                ("StockPnL", "#4C9BE8", "Stock"),
                ("OptionPnL", "#F4A460", "Options"),
                ("Dividends", "#aaffcc", "Dividends"),
            ]:
                fig_mo.add_trace(go.Scatter(
                    x=mon_grp["MonthStr"], y=mon_grp[col],
                    mode="lines+markers", name=name,
                    line=dict(color=clr, width=1.5, dash="dot"),
                    visible="legendonly",
                ))
        st.plotly_chart(fig_mo, use_container_width=True)

        s1, s2, s3, s4 = st.columns(4)
        pos = mon_grp[mon_grp["TotalPnL"] > 0]
        neg = mon_grp[mon_grp["TotalPnL"] < 0]
        s1.metric("Profitable Months", len(pos))
        s2.metric("Losing Months",     len(neg))
        s3.metric("Best Month",  fmt(mon_grp["TotalPnL"].max()))
        s4.metric("Worst Month", fmt(mon_grp["TotalPnL"].min()))

    elif granularity == "Year":
        yr_grp         = cal_df.copy()
        yr_grp["Year"] = yr_grp["Date"].dt.year
        yr_agg = yr_grp.groupby("Year").agg(
            TotalPnL=("TotalPnL", "sum"), StockPnL=("StockPnL", "sum"),
            OptionPnL=("OptionPnL", "sum"), Dividends=("Dividends", "sum"),
        ).reset_index()

        fig_yr = pnl_bar(
            yr_agg["Year"].astype(str).tolist(),
            yr_agg["TotalPnL"].tolist(), "Annual Realized P&L",
        )
        if breakdown_toggle:
            for col, clr, name in [
                ("StockPnL", "#4C9BE8", "Stock"),
                ("OptionPnL", "#F4A460", "Options"),
                ("Dividends", "#aaffcc", "Dividends"),
            ]:
                fig_yr.add_trace(go.Bar(
                    x=yr_agg["Year"].astype(str), y=yr_agg[col],
                    name=name, marker_color=clr, visible="legendonly",
                ))
        st.plotly_chart(fig_yr, use_container_width=True)

        for _, row in yr_agg.iterrows():
            st.markdown(
                f"**{int(row['Year'])}** — "
                f"Total: **{fmt(row['TotalPnL'])}** | "
                f"Stock: {fmt(row['StockPnL'])} | "
                f"Options: {fmt(row['OptionPnL'])} | "
                f"Dividends: {fmt(row['Dividends'])}"
            )


# ---------------------------------------------------------------------------
# Page: Symbol Analysis
# ---------------------------------------------------------------------------

elif page == "Symbol Analysis":
    st.title("Symbol Analysis")

    sym = st.selectbox("Select Symbol", all_symbols)
    if sym:
        m      = symbol_metrics(df, sym, stock_pnl=stock_pnl_df)
        trades = m["trades"]

        st.subheader(f"{sym} — Key Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Realized P&L",             fmt(m["total_realized_pnl"]),
                  delta=f"{m['total_realized_pnl']:+,.2f}")
        c2.metric("Net P&L (after fees+divs)", fmt(m["net_pnl"]),
                  delta=f"{m['net_pnl']:+,.2f}")
        c3.metric("Win Rate",                 f"{m['win_rate']:.1f}%")
        c4.metric("Remaining Shares",         f"{m['remaining_shares']:,.0f}")
        c5.metric("Dividends Received",       fmt(m["total_dividends"]))

        c6, c7, c8, c9, c10 = st.columns(5)
        c6.metric("Avg Buy Price",    fmt(m["avg_buy_price"]))
        c7.metric("Avg Sell Price",   fmt(m["avg_sell_price"]))
        c8.metric("Total Invested",   fmt(m["total_invested"]))
        c9.metric("Best Trade P&L",   fmt(m["best_sell_pnl"]),
                  delta=f"{m['best_sell_pnl']:+,.2f}")
        c10.metric("Worst Trade P&L", fmt(m["worst_sell_pnl"]),
                   delta=f"{m['worst_sell_pnl']:+,.2f}", delta_color="inverse")

        st.divider()

        sells = trades[trades["Action"] == "SELL"].sort_values("TradeDate")
        if not sells.empty:
            sells = sells.copy()
            sells["CumPnL"] = sells["RealizedPnL"].cumsum()
            fig2 = px.area(
                sells, x="TradeDate", y="CumPnL",
                title=f"{sym} — Cumulative Realized P&L",
                labels={"CumPnL": "Cumulative P&L ($)", "TradeDate": "Date"},
                color_discrete_sequence=["#4C9BE8"],
            )
            fig2.update_traces(line_color="#4C9BE8", fillcolor="rgba(76,155,232,0.15)")
            fig2.update_layout(height=300, margin=dict(t=40))
            st.plotly_chart(fig2, use_container_width=True)

            bar_colors = ["#00c97a" if v >= 0 else "#ff4b4b" for v in sells["RealizedPnL"]]
            fig3 = go.Figure(go.Bar(
                x=sells["TradeDate"].dt.strftime("%Y-%m-%d"),
                y=sells["RealizedPnL"],
                marker_color=bar_colors,
                text=[f"${v:,.2f}" for v in sells["RealizedPnL"]],
                textposition="outside",
            ))
            fig3.update_layout(
                title=f"{sym} — P&L per Sell Trade",
                yaxis_title="P&L ($)", height=300, margin=dict(t=40),
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Full Trade History")
        display_cols = ["TradeDate", "Action", "Quantity", "Price", "Amount",
                        "AvgCostBasis", "RealizedPnL", "Fee"]
        show = trades[display_cols].copy()
        show["TradeDate"] = show["TradeDate"].dt.strftime("%Y-%m-%d")
        show = show.rename(columns={
            "AvgCostBasis": "Avg Cost Basis", "RealizedPnL": "Realized P&L",
        })
        st.dataframe(
            show.style.map(style_pnl, subset=["Realized P&L"]),
            use_container_width=True, hide_index=True,
        )
        csv_download(show, f"{sym}_trades.csv")

        divs = get_dividends(df)
        sym_divs = divs[divs["Symbol"] == sym]
        if not sym_divs.empty:
            st.subheader("Dividend History")
            d = sym_divs[["TradeDate", "Amount", "Description"]].copy()
            d["TradeDate"] = d["TradeDate"].dt.strftime("%Y-%m-%d")
            st.dataframe(d, use_container_width=True, hide_index=True)

        # ── Options section ──────────────────────────────────────────────────
        st.divider()
        st.subheader(f"{sym} — Options Activity")

        sym_opts = opt_pnl_df[opt_pnl_df["Underlying"] == sym].copy()

        if sym_opts.empty:
            st.caption("No options traded on this underlying.")
        else:
            closed_opts     = sym_opts[sym_opts["Outcome"] != "Still Open"]
            still_open_opts = sym_opts[sym_opts["Outcome"] == "Still Open"]
            wins_opts       = closed_opts[closed_opts["RealizedPnL"] > 0]
            losses_opts     = closed_opts[closed_opts["RealizedPnL"] <= 0]
            expired_opts    = sym_opts[sym_opts["Outcome"] == "Expired Worthless"]
            opt_win_rate    = len(wins_opts) / len(closed_opts) * 100 if len(closed_opts) > 0 else 0.0
            total_opt_pnl   = closed_opts["RealizedPnL"].fillna(0).sum()
            total_prem_paid = sym_opts["OpenCost"].sum()
            total_prem_rec  = closed_opts["CloseProceeds"].fillna(0).sum()
            avg_hold_days   = (
                (pd.to_datetime(closed_opts["CloseDate"])
                 - pd.to_datetime(closed_opts["OpenDate"])).dt.days.mean()
                if not closed_opts.empty else 0.0
            )
            best_opt  = closed_opts["RealizedPnL"].max() if not closed_opts.empty else 0.0
            worst_opt = closed_opts["RealizedPnL"].min() if not closed_opts.empty else 0.0

            oc1, oc2, oc3, oc4, oc5 = st.columns(5)
            oc1.metric("Options Realized P&L", fmt(total_opt_pnl),
                       delta=f"{total_opt_pnl:+,.2f}")
            oc2.metric("Win Rate (closed)",  f"{opt_win_rate:.1f}%")
            oc3.metric("Total Premium Paid", fmt(total_prem_paid))
            oc4.metric("Premium Recovered",  fmt(total_prem_rec))
            oc5.metric("Avg Hold (days)",    f"{avg_hold_days:.1f}" if avg_hold_days else "—")

            oc6, oc7, oc8, oc9, oc10 = st.columns(5)
            oc6.metric("# Contracts Opened", int(sym_opts["Contracts"].sum()))
            oc7.metric("# Winning Trades",   len(wins_opts))
            oc8.metric("# Losing Trades",    len(losses_opts))
            oc9.metric("Expired Worthless",  len(expired_opts))
            oc10.metric("Best / Worst",      f"{fmt(best_opt)} / {fmt(worst_opt)}")

            if len(still_open_opts) > 0:
                st.info(f"**{len(still_open_opts)}** position(s) still open — "
                        f"total premium at risk: **{fmt(still_open_opts['OpenCost'].sum())}**")

            st.markdown("#### Cumulative P&L — Stock vs Options")
            fig_combo = go.Figure()
            if not sells.empty:
                sc = sells[["TradeDate", "RealizedPnL"]].copy()
                sc["CumPnL"] = sc["RealizedPnL"].cumsum()
                fig_combo.add_trace(go.Scatter(
                    x=sc["TradeDate"], y=sc["CumPnL"],
                    mode="lines+markers", name="Stock P&L",
                    line=dict(color="#4C9BE8", width=2),
                ))
            if not closed_opts.empty:
                oc = closed_opts.copy()
                oc["CloseDate"] = pd.to_datetime(oc["CloseDate"])
                oc = oc.sort_values("CloseDate")
                oc["CumOptPnL"] = oc["RealizedPnL"].cumsum()
                fig_combo.add_trace(go.Scatter(
                    x=oc["CloseDate"], y=oc["CumOptPnL"],
                    mode="lines+markers", name="Options P&L",
                    line=dict(color="#F4A460", width=2),
                ))
            fig_combo.update_layout(
                yaxis_title="Cumulative P&L ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=320, margin=dict(t=40),
            )
            st.plotly_chart(fig_combo, use_container_width=True)

            if not closed_opts.empty:
                closed_sorted = closed_opts.sort_values("OpenDate").copy()
                closed_sorted["Label"] = (
                    closed_sorted["OptionType"].fillna("") + " "
                    + closed_sorted["OptionStrike"].astype(str) + " exp "
                    + closed_sorted["OptionExpiry"].fillna("")
                )
                bar_clr = ["#00c97a" if v >= 0 else "#ff4b4b"
                           for v in closed_sorted["RealizedPnL"]]
                fig_opt_bar = go.Figure(go.Bar(
                    x=closed_sorted["Label"],
                    y=closed_sorted["RealizedPnL"],
                    marker_color=bar_clr,
                    text=[f"${v:,.2f}" for v in closed_sorted["RealizedPnL"]],
                    textposition="outside",
                    customdata=closed_sorted[["OpenDate", "CloseDate", "Contracts",
                                              "OpenCost", "CloseProceeds", "Outcome"]].values,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Open: %{customdata[0]}<br>"
                        "Close: %{customdata[1]}<br>"
                        "Contracts: %{customdata[2]}<br>"
                        "Cost: $%{customdata[3]:,.2f}<br>"
                        "Proceeds: $%{customdata[4]:,.2f}<br>"
                        "Outcome: %{customdata[5]}<br>"
                        "P&L: $%{y:,.2f}<extra></extra>"
                    ),
                ))
                fig_opt_bar.update_layout(
                    title=f"{sym} — P&L per Options Contract",
                    yaxis_title="Realized P&L ($)",
                    xaxis_tickangle=-35,
                    height=380, margin=dict(t=40, b=120),
                )
                st.plotly_chart(fig_opt_bar, use_container_width=True)

            st.markdown("#### All Options Contracts")
            disp_o = sym_opts.copy()
            disp_o["OpenDate"]  = pd.to_datetime(disp_o["OpenDate"]).dt.strftime("%Y-%m-%d")
            disp_o["CloseDate"] = (
                pd.to_datetime(disp_o["CloseDate"]).dt.strftime("%Y-%m-%d").fillna("—")
            )
            exp_o = sym_opts.copy()
            for col in ["OpenCost", "CloseProceeds", "RealizedPnL"]:
                disp_o[col] = disp_o[col].apply(
                    lambda v: f"${v:,.2f}" if pd.notna(v) else "—"
                )
            disp_o = disp_o.rename(columns={
                "OptionType": "Type", "OptionExpiry": "Expiry",
                "OptionStrike": "Strike", "OpenDate": "Open Date",
                "CloseDate": "Close Date", "OpenCost": "Premium Paid",
                "CloseProceeds": "Proceeds", "RealizedPnL": "P&L",
            })
            show_cols = ["Type", "Strike", "Expiry", "Contracts",
                         "Open Date", "Close Date", "Premium Paid", "Proceeds", "P&L", "Outcome"]
            st.dataframe(
                disp_o[show_cols].style.map(style_pnl, subset=["P&L"]),
                use_container_width=True, hide_index=True,
            )
            csv_download(exp_o, f"{sym}_options.csv")


# ---------------------------------------------------------------------------
# Page: Daily View
# ---------------------------------------------------------------------------

elif page == "Daily View":
    st.title("Daily View")

    selected_date = st.date_input(
        "Select Date", value=date_max, min_value=date_min, max_value=date_max,
    )
    dm = day_metrics(df, pd.Timestamp(selected_date),
                     stock_pnl=stock_pnl_df, opt_pnl=opt_pnl_df)

    if (dm["stock_trades"].empty and dm["option_trades"].empty
            and dm["dividends"].empty and dm["interest"].empty):
        st.info("No activity on this date.")
    else:
        st.subheader(f"Summary — {selected_date}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Realized P&L", fmt(dm["total_realized_pnl"]),
                  delta=f"{dm['total_realized_pnl']:+,.2f}")
        c2.metric("Stock P&L",          fmt(dm["realized_stock_pnl"]),
                  delta=f"{dm['realized_stock_pnl']:+,.2f}")
        c3.metric("Options P&L",        fmt(dm["realized_opt_pnl"]),
                  delta=f"{dm['realized_opt_pnl']:+,.2f}")
        c4.metric("Net Cash Flow",      fmt(dm["net_cash_flow"]),
                  delta=f"{dm['net_cash_flow']:+,.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Cash Deployed (buys)", fmt(dm["total_bought_cash"]))
        c6.metric("Cash Received (sells)", fmt(dm["total_sold_cash"]))
        c7.metric("Options Spent",        fmt(dm["total_opt_spent"]))
        c8.metric("Dividends",            fmt(dm["total_dividends"]))

        if dm["total_margin_interest"] < 0:
            st.warning(
                f"Margin interest charged today: **{fmt(abs(dm['total_margin_interest']))}**"
            )

        st.divider()

        if not dm["stock_trades"].empty:
            st.subheader("Stock Trades")
            cols = ["Symbol", "Action", "Quantity", "Price", "Amount", "Fee"]
            st.dataframe(dm["stock_trades"][cols], use_container_width=True, hide_index=True)
            csv_download(dm["stock_trades"][cols], f"{selected_date}_stock_trades.csv")

        if not dm["option_trades"].empty:
            st.subheader("Options Trades")
            ocols = ["OptionUnderlying", "OptionType", "OptionStrike", "OptionExpiry",
                     "OptionStatus", "Action", "Quantity", "Price", "Amount", "Fee"]
            st.dataframe(dm["option_trades"][ocols], use_container_width=True, hide_index=True)
            csv_download(dm["option_trades"][ocols], f"{selected_date}_option_trades.csv")

        if not dm["option_pnl_rows"].empty:
            st.subheader("Options Closed/Expired Today — P&L")
            st.dataframe(
                dm["option_pnl_rows"].style.map(style_pnl, subset=["RealizedPnL"]),
                use_container_width=True, hide_index=True,
            )

        if not dm["dividends"].empty:
            st.subheader("Dividends")
            st.dataframe(
                dm["dividends"][["Symbol", "Amount", "Description"]],
                use_container_width=True, hide_index=True,
            )

        if not dm["interest"].empty:
            st.subheader("Interest")
            st.dataframe(
                dm["interest"][["Amount", "Description"]],
                use_container_width=True, hide_index=True,
            )


# ---------------------------------------------------------------------------
# Page: Symbol + Day
# ---------------------------------------------------------------------------

elif page == "Symbol + Day":
    st.title("Symbol + Day")
    st.caption("All key metrics for a single ticker on a single trading day.")

    col_s, col_d = st.columns(2)
    with col_s:
        sym = st.selectbox("Symbol", all_symbols)
    with col_d:
        selected_date = st.date_input(
            "Date", value=date_max, min_value=date_min, max_value=date_max,
        )

    dm = day_metrics(df, pd.Timestamp(selected_date), symbol=sym,
                     stock_pnl=stock_pnl_df, opt_pnl=opt_pnl_df)

    no_activity = (dm["stock_trades"].empty and dm["option_trades"].empty
                   and dm["dividends"].empty)
    if no_activity:
        st.info(f"No activity for **{sym}** on **{selected_date}**.")
    else:
        st.subheader(f"{sym} on {selected_date} — Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Realized Stock P&L",   fmt(dm["realized_stock_pnl"]),
                  delta=f"{dm['realized_stock_pnl']:+,.2f}")
        c2.metric("Realized Options P&L", fmt(dm["realized_opt_pnl"]),
                  delta=f"{dm['realized_opt_pnl']:+,.2f}")
        c3.metric("Total Realized P&L",   fmt(dm["total_realized_pnl"]),
                  delta=f"{dm['total_realized_pnl']:+,.2f}")
        c4.metric("Cash Deployed",        fmt(dm["total_bought_cash"]))
        c5.metric("Cash Received",        fmt(dm["total_sold_cash"]))

        c6, c7, c8, c9, c10 = st.columns(5)
        c6.metric("Options Spent",    fmt(dm["total_opt_spent"]))
        c7.metric("Options Received", fmt(dm["total_opt_received"]))
        c8.metric("Dividends",        fmt(dm["total_dividends"]))
        c9.metric("Fees",             fmt(dm["total_fees"]))
        c10.metric("Net Cash Flow",   fmt(dm["net_cash_flow"]),
                   delta=f"{dm['net_cash_flow']:+,.2f}")

        st.divider()

        if not dm["stock_trades"].empty:
            st.subheader("Stock Trades")
            day_rows = stock_pnl_df[
                (stock_pnl_df["Symbol"] == sym)
                & (stock_pnl_df["TradeDate"].dt.date == selected_date)
            ]
            cols = ["Action", "Quantity", "Price", "Amount", "AvgCostBasis", "RealizedPnL", "Fee"]
            show = day_rows[cols].copy().rename(
                columns={"AvgCostBasis": "Avg Cost Basis", "RealizedPnL": "Realized P&L"}
            )
            st.dataframe(
                show.style.map(style_pnl, subset=["Realized P&L"]),
                use_container_width=True, hide_index=True,
            )

            if not day_rows.empty:
                buys_day  = day_rows[day_rows["Action"] == "BUY"]
                sells_day = day_rows[day_rows["Action"] == "SELL"]
                wx, wy, wt, wc = [], [], [], []
                for _, r in buys_day.iterrows():
                    wx.append(f"BUY {int(abs(r['Quantity']))}@{r['Price']:.2f}")
                    wy.append(r["Amount"])
                    wt.append(f"${r['Amount']:,.2f}")
                    wc.append("#ff4b4b")
                for _, r in sells_day.iterrows():
                    wx.append(f"SELL {int(abs(r['Quantity']))}@{r['Price']:.2f}")
                    wy.append(r["Amount"])
                    wt.append(f"${r['Amount']:,.2f}")
                    wc.append("#00c97a")
                if wx:
                    fig_wf = go.Figure(go.Bar(
                        x=wx, y=wy, marker_color=wc, text=wt, textposition="outside",
                    ))
                    fig_wf.update_layout(
                        title="Cash Flow per Trade", yaxis_title="Amount ($)",
                        height=300, margin=dict(t=40),
                    )
                    st.plotly_chart(fig_wf, use_container_width=True)

        if not dm["option_trades"].empty:
            st.subheader("Options Activity")
            ocols = ["OptionType", "OptionStrike", "OptionExpiry",
                     "OptionStatus", "Action", "Quantity", "Price", "Amount"]
            st.dataframe(dm["option_trades"][ocols], use_container_width=True, hide_index=True)

        if not dm["option_pnl_rows"].empty:
            st.subheader("Options Closed/Expired — P&L")
            st.dataframe(
                dm["option_pnl_rows"].style.map(style_pnl, subset=["RealizedPnL"]),
                use_container_width=True, hide_index=True,
            )

        if not dm["dividends"].empty:
            st.subheader("Dividends")
            st.dataframe(
                dm["dividends"][["Amount", "Description"]],
                use_container_width=True, hide_index=True,
            )


# ---------------------------------------------------------------------------
# Page: Options Analysis
# ---------------------------------------------------------------------------

elif page == "Options Analysis":
    st.title("Options Analysis")

    opt_pnl = opt_pnl_df

    if opt_pnl.empty:
        st.info("No options data found.")
    else:
        closed     = opt_pnl[opt_pnl["Outcome"] != "Still Open"]
        still_open = opt_pnl[opt_pnl["Outcome"] == "Still Open"]

        total_spent     = opt_pnl["OpenCost"].sum()
        total_recovered = closed["CloseProceeds"].fillna(0).sum()
        total_realized  = closed["RealizedPnL"].fillna(0).sum()
        wins            = closed[closed["RealizedPnL"] > 0]
        losses          = closed[closed["RealizedPnL"] <= 0]
        win_rate        = len(wins) / len(closed) * 100 if len(closed) > 0 else 0.0
        expired         = opt_pnl[opt_pnl["Outcome"] == "Expired Worthless"]
        total_lost      = expired["RealizedPnL"].sum()

        st.subheader("Overall Options Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Options Realized P&L", fmt(total_realized),
                  delta=f"{total_realized:+,.2f}")
        c2.metric("Win Rate (closed trades)",   f"{win_rate:.1f}%")
        c3.metric("Total Premium Spent",        fmt(total_spent))
        c4.metric("Total Premium Recovered",    fmt(total_recovered))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("# Winning Trades",          len(wins))
        c6.metric("# Losing Trades",           len(losses))
        c7.metric("Expired Worthless (count)", len(expired))
        c8.metric("Lost to Expirations",       fmt(abs(total_lost)),
                  delta=f"{total_lost:+,.2f}", delta_color="inverse")

        st.divider()

        st.subheader("P&L by Underlying")
        by_sym = (
            closed.groupby("Underlying")["RealizedPnL"]
            .sum().reset_index().sort_values("RealizedPnL")
        )
        bar_colors = ["#00c97a" if v >= 0 else "#ff4b4b" for v in by_sym["RealizedPnL"]]
        fig_sym = go.Figure(go.Bar(
            x=by_sym["Underlying"], y=by_sym["RealizedPnL"],
            marker_color=bar_colors,
            text=[f"${v:,.2f}" for v in by_sym["RealizedPnL"]],
            textposition="outside",
        ))
        fig_sym.update_layout(yaxis_title="Realized P&L ($)", height=350, margin=dict(t=20))
        st.plotly_chart(fig_sym, use_container_width=True)

        outcome_counts = opt_pnl["Outcome"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]
        fig_pie = px.pie(
            outcome_counts, names="Outcome", values="Count",
            title="Trade Outcomes",
            color_discrete_map={
                "Closed": "#4C9BE8", "Expired Worthless": "#ff4b4b",
                "Still Open": "#F4A460", "Exercised": "#aaffcc",
            },
        )
        fig_pie.update_layout(height=320)

        col_pie, col_bar = st.columns([1, 2])
        with col_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_bar:
            fig_dist = px.histogram(
                closed.copy(), x="RealizedPnL", nbins=20,
                title="P&L Distribution (closed trades)",
                color_discrete_sequence=["#4C9BE8"],
                labels={"RealizedPnL": "Realized P&L ($)"},
            )
            fig_dist.update_layout(height=320, margin=dict(t=40))
            st.plotly_chart(fig_dist, use_container_width=True)

        st.divider()

        filter_underlying = st.selectbox(
            "Filter by Underlying (optional)",
            ["All"] + sorted(opt_pnl["Underlying"].dropna().unique().tolist()),
        )
        filter_outcome = st.selectbox(
            "Filter by Outcome",
            ["All"] + sorted(opt_pnl["Outcome"].dropna().unique().tolist()),
        )

        display_opts = opt_pnl.copy()
        if filter_underlying != "All":
            display_opts = display_opts[display_opts["Underlying"] == filter_underlying]
        if filter_outcome != "All":
            display_opts = display_opts[display_opts["Outcome"] == filter_outcome]

        export_opts = display_opts.copy()
        for col in ["OpenDate", "CloseDate"]:
            display_opts[col] = pd.to_datetime(display_opts[col]).dt.strftime("%Y-%m-%d")
        for col in ["OpenCost", "CloseProceeds", "RealizedPnL"]:
            display_opts[col] = display_opts[col].apply(
                lambda v: f"${v:,.2f}" if pd.notna(v) else "—"
            )

        st.dataframe(display_opts, use_container_width=True, hide_index=True)
        csv_download(export_opts, "options_analysis.csv")

        if not still_open.empty:
            st.subheader("Still Open Positions")
            st.dataframe(still_open, use_container_width=True, hide_index=True)
            csv_download(still_open, "options_still_open.csv")


# ---------------------------------------------------------------------------
# Page: Performance Analysis
# ---------------------------------------------------------------------------

elif page == "Performance Analysis":
    perf_page.render(trade_log_df, opt_pnl_df, fmt)


# ---------------------------------------------------------------------------
# Page: Open Positions
# ---------------------------------------------------------------------------

elif page == "Open Positions":
    st.title("Open Positions")
    st.caption(
        "Current stock holdings derived from your trading history. "
        "Enable the toggle below to fetch live prices from Yahoo Finance."
    )

    open_pos  = get_open_positions(stock_pnl_df)
    open_opts = opt_pnl_df[opt_pnl_df["Outcome"] == "Still Open"].copy()

    # ── Stock positions ──────────────────────────────────────────────────────
    st.subheader("Stock Positions")
    if open_pos.empty:
        st.info("No open stock positions found.")
    else:
        total_cost = open_pos["TotalCost"].sum()

        fetch_live = st.checkbox("Fetch live prices from Yahoo Finance", value=False)
        if fetch_live:
            with st.spinner("Fetching live prices…"):
                try:
                    import yfinance as yf
                    tickers = open_pos["Symbol"].tolist()
                    raw = yf.download(
                        tickers if len(tickers) > 1 else tickers[0],
                        period="1d", progress=False, auto_adjust=True,
                    )
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    if not raw.empty:
                        prices = raw["Close"].iloc[-1]
                        if isinstance(prices, (float, int)):
                            prices = pd.Series([prices], index=tickers)
                        open_pos = open_pos.copy()
                        open_pos["LivePrice"]     = open_pos["Symbol"].map(prices)
                        open_pos["MarketValue"]   = open_pos["Shares"] * open_pos["LivePrice"]
                        open_pos["UnrealizedPnL"] = open_pos["MarketValue"] - open_pos["TotalCost"]
                        open_pos["UnrealizedPct"] = (
                            open_pos["UnrealizedPnL"] / open_pos["TotalCost"].replace(0, np.nan) * 100
                        ).round(2)
                except Exception as e:
                    st.error(f"Could not fetch live prices: {e}")
                    fetch_live = False

        if fetch_live and "MarketValue" in open_pos.columns:
            total_mkt = open_pos["MarketValue"].sum()
            total_unr = open_pos["UnrealizedPnL"].sum()
            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("# Open Positions", len(open_pos))
            oc2.metric("Total Cost Basis", fmt(total_cost))
            oc3.metric("Market Value",     fmt(total_mkt))
            oc4.metric("Unrealized P&L",   fmt(total_unr),
                       delta=f"{total_unr:+,.2f}")
        else:
            oc1, oc2 = st.columns(2)
            oc1.metric("# Open Positions", len(open_pos))
            oc2.metric("Total Cost Basis", fmt(total_cost))

        disp = open_pos.copy()
        disp["LastTradeDate"] = pd.to_datetime(disp["LastTradeDate"]).dt.strftime("%Y-%m-%d")
        disp["Shares"]        = disp["Shares"].map(lambda v: f"{v:,.0f}")
        disp["AvgCostBasis"]  = disp["AvgCostBasis"].map(lambda v: f"${v:,.4f}")
        disp["TotalCost"]     = disp["TotalCost"].map(lambda v: f"${v:,.2f}")
        if fetch_live and "MarketValue" in open_pos.columns:
            disp["LivePrice"]     = open_pos["LivePrice"].map(
                lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
            disp["MarketValue"]   = open_pos["MarketValue"].map(
                lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
            disp["UnrealizedPnL"] = open_pos["UnrealizedPnL"].map(
                lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
            disp["UnrealizedPct"] = open_pos["UnrealizedPct"].map(
                lambda v: f"{v:.1f}%" if pd.notna(v) else "—")

        disp = disp.rename(columns={
            "AvgCostBasis": "Avg Cost/Share",
            "TotalCost":    "Cost Basis",
            "LastTradeDate": "Last Trade",
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)
        csv_download(open_pos, "open_positions.csv")

        # Portfolio pie chart
        fig_pie = px.pie(
            open_pos, names="Symbol", values="TotalCost",
            title="Portfolio Composition (by Cost Basis)",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=420)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Open options ─────────────────────────────────────────────────────────
    st.subheader("Open Option Positions")
    if open_opts.empty:
        st.info("No open options positions.")
    else:
        total_prem_at_risk = open_opts["OpenCost"].sum()
        oo1, oo2 = st.columns(2)
        oo1.metric("# Open Option Positions", len(open_opts))
        oo2.metric("Total Premium at Risk",   fmt(total_prem_at_risk))

        disp_opts = open_opts.copy()
        disp_opts["OpenDate"] = pd.to_datetime(disp_opts["OpenDate"]).dt.strftime("%Y-%m-%d")
        disp_opts["OpenCost"] = disp_opts["OpenCost"].map(lambda v: f"${v:,.2f}")
        show_cols = ["Underlying", "OptionType", "OptionStrike", "OptionExpiry",
                     "OpenDate", "Contracts", "OpenCost"]
        st.dataframe(disp_opts[show_cols], use_container_width=True, hide_index=True)
        csv_download(open_opts[show_cols], "open_options.csv")


# ---------------------------------------------------------------------------
# Page: Tax Summary
# ---------------------------------------------------------------------------

elif page == "Tax Summary":
    st.title("Tax Summary")
    st.caption(
        "Classifies closed trades as **short-term** (<365 days held) or "
        "**long-term** (≥365 days held). "
        "For informational use only — consult a tax professional."
    )

    tax    = compute_tax_summary(trade_log_df)
    st_inf = tax["short_term"]
    lt_inf = tax["long_term"]

    st.subheader("Capital Gains Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Short-Term Trades",   st_inf["count"])
    c2.metric("Short-Term Net",       fmt(st_inf["net"]),
              delta=f"{st_inf['net']:+,.2f}")
    c3.metric("Long-Term Trades",    lt_inf["count"])
    c4.metric("Long-Term Net",        fmt(lt_inf["net"]),
              delta=f"{lt_inf['net']:+,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("ST Gains",   fmt(st_inf["gains"]))
    c6.metric("ST Losses",  fmt(st_inf["losses"]))
    c7.metric("LT Gains",   fmt(lt_inf["gains"]))
    c8.metric("LT Losses",  fmt(lt_inf["losses"]))

    total_net = st_inf["net"] + lt_inf["net"]
    st.metric("Total Net Realized Gain/Loss", fmt(total_net),
              delta=f"{total_net:+,.2f}")

    if st_inf["net"] < 0:
        st.success(
            f"✓ Short-term losses of **{fmt(abs(st_inf['net']))}** "
            "can offset other short-term (ordinary income) gains."
        )
    if lt_inf["net"] > 0:
        st.info(
            "Long-term gains may qualify for a **lower tax rate** than short-term gains. "
            "Consider deferring short-term sells to the 365-day mark where feasible."
        )

    st.divider()

    fig_tax = go.Figure()
    categories = [
        ("ST Gains",   st_inf["gains"],  "#4C9BE8"),
        ("ST Losses",  st_inf["losses"], "#ff4b4b"),
        ("LT Gains",   lt_inf["gains"],  "#00c97a"),
        ("LT Losses",  lt_inf["losses"], "#F4A460"),
    ]
    for name, val, clr in categories:
        fig_tax.add_trace(go.Bar(
            name=name, x=[name], y=[val],
            marker_color=clr,
            text=[f"${val:,.2f}"], textposition="outside",
        ))
    fig_tax.update_layout(
        title="Capital Gains Breakdown",
        yaxis_title="Amount ($)", height=380, margin=dict(t=40),
        showlegend=False,
    )
    st.plotly_chart(fig_tax, use_container_width=True)

    st.divider()

    tab1, tab2 = st.tabs(["Short-Term Trades", "Long-Term Trades"])

    with tab1:
        st.caption("Positions held fewer than 365 days — taxed as ordinary income.")
        if tax["short_term_df"].empty:
            st.info("No short-term trades.")
        else:
            st_disp = tax["short_term_df"][
                ["CloseDate", "Symbol", "TradeType", "RealizedPnL", "HoldDays", "Outcome"]
            ].copy()
            st_disp["CloseDate"]   = st_disp["CloseDate"].dt.strftime("%Y-%m-%d")
            st_disp["HoldDays"]    = st_disp["HoldDays"].map(lambda v: f"{v:.0f}d")
            st_disp["RealizedPnL"] = st_disp["RealizedPnL"].map(lambda v: f"${v:,.2f}")
            st.dataframe(
                st_disp.style.map(style_pnl, subset=["RealizedPnL"]),
                use_container_width=True, hide_index=True,
            )
            csv_download(tax["short_term_df"], "short_term_trades.csv")

    with tab2:
        st.caption("Positions held 365+ days — potentially taxed at lower long-term rate.")
        if tax["long_term_df"].empty:
            st.info("No long-term trades.")
        else:
            lt_disp = tax["long_term_df"][
                ["CloseDate", "Symbol", "TradeType", "RealizedPnL", "HoldDays", "Outcome"]
            ].copy()
            lt_disp["CloseDate"]   = lt_disp["CloseDate"].dt.strftime("%Y-%m-%d")
            lt_disp["HoldDays"]    = lt_disp["HoldDays"].map(lambda v: f"{v:.0f}d")
            lt_disp["RealizedPnL"] = lt_disp["RealizedPnL"].map(lambda v: f"${v:,.2f}")
            st.dataframe(
                lt_disp.style.map(style_pnl, subset=["RealizedPnL"]),
                use_container_width=True, hide_index=True,
            )
            csv_download(tax["long_term_df"], "long_term_trades.csv")
