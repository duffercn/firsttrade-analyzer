"""Performance Analysis page — imported and called from app.py."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    compute_core_stats,
    dow_month_heatmap_data,
    options_deep_dive,
    rolling_win_rate,
    symbol_ranking,
)


def render(trade_log_df, opt_pnl_df, fmt):
    st.title("Performance Analysis")

    if trade_log_df.empty:
        st.info("No closed trades found.")
        return

    # ── filters ─────────────────────────────────────────────────────────────
    with st.expander("Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            type_filter = st.multiselect(
                "Trade type", ["Stock", "Option"],
                default=["Stock", "Option"],
            )
        with fc2:
            available_years = sorted(trade_log_df["Year"].unique().tolist())
            year_filter = st.multiselect(
                "Year", available_years, default=available_years
            )
        with fc3:
            syms = sorted(trade_log_df["Symbol"].unique().tolist())
            sym_filter = st.multiselect("Symbol (leave blank = all)", syms)

    tlog = trade_log_df.copy()
    if type_filter:
        tlog = tlog[tlog["TradeType"].isin(type_filter)]
    if year_filter:
        tlog = tlog[tlog["Year"].isin(year_filter)]
    if sym_filter:
        tlog = tlog[tlog["Symbol"].isin(sym_filter)]

    stats = compute_core_stats(tlog)
    if not stats:
        st.warning("No closed trades match the current filters.")
        return

    # ════════════════════════════════════════════════════════════════════════
    # 1. Core trade statistics
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Core Trade Statistics")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trades",    stats["n_total"])
    c2.metric("Win Rate",        f"{stats['win_rate']:.1f}%")
    c3.metric("Expectancy / trade", fmt(stats["expectancy"]),
              delta=f"{stats['expectancy']:+,.2f}")
    c4.metric("Profit Factor",
              f"{stats['profit_factor']:.2f}" if stats["profit_factor"] != float("inf") else "∞")
    c5.metric("Risk/Reward Ratio",
              f"{stats['rr_ratio']:.2f}" if stats["rr_ratio"] != float("inf") else "∞")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Avg Win",           fmt(stats["avg_win"]))
    c7.metric("Avg Loss",          fmt(stats["avg_loss"]))
    c8.metric("Gross Profit",      fmt(stats["gross_win"]))
    c9.metric("Gross Loss",        fmt(stats["gross_loss"]))
    c10.metric("Net P&L",          fmt(stats["total_pnl"]),
               delta=f"{stats['total_pnl']:+,.2f}")

    c11, c12, c13, c14, c15 = st.columns(5)
    c11.metric("Max Consec. Wins",   stats["max_consec_wins"])
    c12.metric("Max Consec. Losses", stats["max_consec_losses"])
    c13.metric("Avg Hold (wins)",    f"{stats['avg_hold_win']:.1f}d")
    c14.metric("Avg Hold (losses)",  f"{stats['avg_hold_loss']:.1f}d")
    c15.metric("Best / Worst",
               f"{fmt(stats['best_trade'])} / {fmt(stats['worst_trade'])}")

    # Disposition effect callout
    if stats["avg_hold_loss"] > stats["avg_hold_win"] * 1.3:
        st.warning(
            f"⚠️ **Disposition Effect detected**: you hold losing trades "
            f"**{stats['avg_hold_loss']:.0f}d** on average vs "
            f"**{stats['avg_hold_win']:.0f}d** for winners. "
            "Classic sign of cutting winners early and riding losers."
        )

    if stats["profit_factor"] < 1.0:
        st.error(
            f"🚨 **Profit factor below 1.0** ({stats['profit_factor']:.2f}): "
            "gross losses exceed gross profits. Review position sizing or entry criteria."
        )
    elif stats["profit_factor"] < 1.5:
        st.warning(
            f"⚠️ **Profit factor {stats['profit_factor']:.2f}** — below the 1.5 "
            "benchmark. Aim to either raise win rate or improve avg win/loss ratio."
        )

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # 2. Win/Loss distribution
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Win / Loss Distribution")
    dist_col1, dist_col2 = st.columns([3, 2])

    closed = tlog[tlog["Outcome"].isin(["Win", "Loss"])].copy()
    with dist_col1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=closed[closed["Outcome"] == "Win"]["RealizedPnL"],
            name="Wins", marker_color="#00c97a", opacity=0.75,
            nbinsx=30,
        ))
        fig_hist.add_trace(go.Histogram(
            x=closed[closed["Outcome"] == "Loss"]["RealizedPnL"],
            name="Losses", marker_color="#ff4b4b", opacity=0.75,
            nbinsx=30,
        ))
        fig_hist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig_hist.update_layout(
            barmode="overlay", title="P&L Distribution",
            xaxis_title="Realized P&L ($)", yaxis_title="# Trades",
            height=320, margin=dict(t=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with dist_col2:
        fig_wb = go.Figure(go.Bar(
            x=["Avg Win", "Avg Loss"],
            y=[stats["avg_win"], abs(stats["avg_loss"])],
            marker_color=["#00c97a", "#ff4b4b"],
            text=[f"${stats['avg_win']:,.2f}", f"${abs(stats['avg_loss']):,.2f}"],
            textposition="outside",
        ))
        fig_wb.update_layout(
            title=f"Avg Win vs Avg Loss  (R:R = {stats['rr_ratio']:.2f}x)",
            yaxis_title="$", height=320, margin=dict(t=40),
        )
        st.plotly_chart(fig_wb, use_container_width=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # 3. Rolling win rate
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Rolling Win Rate")
    rw_col1, _ = st.columns([1, 3])
    with rw_col1:
        window = st.slider("Window (trades)", 5, 50, 20, step=5)

    rwr = rolling_win_rate(tlog, window)
    if not rwr.empty:
        fig_rwr = go.Figure()
        fig_rwr.add_hline(y=50, line_dash="dot", line_color="gray",
                          annotation_text="50%", annotation_position="right")
        win_rate_overall = stats["win_rate"]
        fig_rwr.add_hline(y=win_rate_overall, line_dash="dash",
                          line_color="#4C9BE8", opacity=0.7,
                          annotation_text=f"Overall {win_rate_overall:.1f}%",
                          annotation_position="right")
        clr = ["#00c97a" if v >= 50 else "#ff4b4b" for v in rwr["RollingWR"]]
        fig_rwr.add_trace(go.Scatter(
            x=rwr["CloseDate"], y=rwr["RollingWR"],
            mode="lines",
            line=dict(color="#4C9BE8", width=2),
            fill="tozeroy",
            fillcolor="rgba(76,155,232,0.1)",
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Rolling Win Rate: %{y:.1f}%<extra></extra>"
            ),
            name=f"{window}-trade rolling WR",
        ))
        fig_rwr.update_layout(
            yaxis=dict(title="Win Rate (%)", range=[0, 100]),
            xaxis_title="Date",
            height=300, margin=dict(t=20),
        )
        st.plotly_chart(fig_rwr, use_container_width=True)
        st.caption(
            "Periods **below 50%** (red) indicate the strategy was losing more "
            "than half its trades in that stretch. Look for what changed."
        )

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # 4. Hold time analysis
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Hold Time Analysis")
    ht_col1, ht_col2 = st.columns(2)

    with ht_col1:
        fig_box = go.Figure()
        for outcome, color in [("Win", "#00c97a"), ("Loss", "#ff4b4b")]:
            subset = closed[closed["Outcome"] == outcome]
            fig_box.add_trace(go.Box(
                y=subset["HoldDays"], name=outcome,
                marker_color=color, boxmean=True,
                hovertemplate="Hold: %{y:.0f}d<extra></extra>",
            ))
        fig_box.update_layout(
            title="Hold Time: Winners vs Losers",
            yaxis_title="Days held", height=340, margin=dict(t=40),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with ht_col2:
        # Scatter: hold days vs P&L, sized by |P&L|
        scatter_data = closed.copy()
        scatter_data["AbsPnL"] = scatter_data["RealizedPnL"].abs()
        fig_sc = px.scatter(
            scatter_data,
            x="HoldDays", y="RealizedPnL",
            color="Outcome",
            size="AbsPnL",
            size_max=20,
            color_discrete_map={"Win": "#00c97a", "Loss": "#ff4b4b"},
            hover_data=["Symbol", "TradeType", "CloseDate"],
            title="Hold Days vs Realized P&L",
            labels={"HoldDays": "Days Held", "RealizedPnL": "P&L ($)"},
        )
        fig_sc.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig_sc.update_layout(height=340, margin=dict(t=40))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # 5. Day-of-week & month-of-year patterns
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Day-of-Week & Month-of-Year Patterns")
    dow_df, month_df = dow_month_heatmap_data(tlog)

    pat_col1, pat_col2 = st.columns(2)

    with pat_col1:
        if not dow_df.dropna().empty:
            bar_clr = ["#00c97a" if v >= 0 else "#ff4b4b"
                       for v in dow_df["TotalPnL"].fillna(0)]
            fig_dow = go.Figure()
            fig_dow.add_trace(go.Bar(
                x=dow_df["DayOfWeek"], y=dow_df["TotalPnL"].fillna(0),
                name="Total P&L", marker_color=bar_clr,
                yaxis="y",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Total P&L: $%{y:,.2f}<br>"
                    "Win Rate: %{customdata[0]:.1f}%<br>"
                    "Trades: %{customdata[1]}<extra></extra>"
                ),
                customdata=dow_df[["WinRate", "Count"]].fillna(0).values,
            ))
            fig_dow.add_trace(go.Scatter(
                x=dow_df["DayOfWeek"], y=dow_df["WinRate"].fillna(0),
                name="Win Rate %", mode="lines+markers",
                line=dict(color="#F4A460", width=2),
                yaxis="y2",
            ))
            fig_dow.update_layout(
                title="Performance by Day of Week",
                yaxis=dict(title="Total P&L ($)"),
                yaxis2=dict(title="Win Rate (%)", overlaying="y",
                            side="right", range=[0, 100]),
                legend=dict(orientation="h", y=1.1),
                height=340, margin=dict(t=50),
            )
            st.plotly_chart(fig_dow, use_container_width=True)

    with pat_col2:
        if not month_df.empty:
            bar_clr_m = ["#00c97a" if v >= 0 else "#ff4b4b"
                         for v in month_df["TotalPnL"]]
            fig_mo = go.Figure()
            fig_mo.add_trace(go.Bar(
                x=month_df["MonthName"], y=month_df["TotalPnL"],
                name="Total P&L", marker_color=bar_clr_m,
                yaxis="y",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Total P&L: $%{y:,.2f}<br>"
                    "Win Rate: %{customdata[0]:.1f}%<br>"
                    "Trades: %{customdata[1]}<extra></extra>"
                ),
                customdata=month_df[["WinRate", "Count"]].values,
            ))
            fig_mo.add_trace(go.Scatter(
                x=month_df["MonthName"], y=month_df["WinRate"],
                name="Win Rate %", mode="lines+markers",
                line=dict(color="#F4A460", width=2),
                yaxis="y2",
            ))
            fig_mo.update_layout(
                title="Performance by Month",
                yaxis=dict(title="Total P&L ($)"),
                yaxis2=dict(title="Win Rate (%)", overlaying="y",
                            side="right", range=[0, 100]),
                legend=dict(orientation="h", y=1.1),
                height=340, margin=dict(t=50),
            )
            st.plotly_chart(fig_mo, use_container_width=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # 6. Symbol performance ranking
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Symbol Performance Ranking")

    ranking = symbol_ranking(tlog, opt_pnl_df)

    rank_col1, rank_col2 = st.columns([1, 2])
    with rank_col1:
        sort_by = st.selectbox("Sort by", ["TotalPnL", "WinRate", "ProfitFactor",
                                           "Trades", "AvgPnL"])
    ranking_sorted = ranking.sort_values(sort_by, ascending=False)

    # Top 10 chart
    top10 = ranking_sorted.head(10)
    bottom10 = ranking_sorted.tail(10)
    with rank_col2:
        fig_rank = go.Figure()
        clr_top = ["#00c97a" if v >= 0 else "#ff4b4b" for v in top10["TotalPnL"]]
        fig_rank.add_trace(go.Bar(
            x=top10["Symbol"], y=top10["TotalPnL"],
            name="Top 10", marker_color=clr_top,
        ))
        fig_rank.update_layout(
            title="Top 10 Symbols by Total P&L",
            yaxis_title="Total P&L ($)", height=280, margin=dict(t=40),
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    # Full ranking table
    disp = ranking_sorted.copy()
    disp["WinRate"]       = disp["WinRate"].map(lambda v: f"{v:.1f}%")
    disp["TotalPnL"]      = disp["TotalPnL"].map(lambda v: f"${v:,.2f}")
    disp["AvgPnL"]        = disp["AvgPnL"].map(lambda v: f"${v:,.2f}")
    disp["AvgWin"]        = disp["AvgWin"].map(lambda v: f"${v:,.2f}")
    disp["AvgLoss"]       = disp["AvgLoss"].map(lambda v: f"${v:,.2f}")
    disp["BestTrade"]     = disp["BestTrade"].map(lambda v: f"${v:,.2f}")
    disp["WorstTrade"]    = disp["WorstTrade"].map(lambda v: f"${v:,.2f}")
    disp["ProfitFactor"]  = disp["ProfitFactor"].map(
        lambda v: f"{v:.2f}" if v != float("inf") else "∞")
    disp["AvgHoldDays"]   = disp["AvgHoldDays"].map(lambda v: f"{v:.1f}d")
    disp = disp.rename(columns={
        "TotalPnL": "Total P&L", "WinRate": "Win Rate",
        "AvgPnL": "Avg P&L", "AvgWin": "Avg Win", "AvgLoss": "Avg Loss",
        "BestTrade": "Best", "WorstTrade": "Worst",
        "ProfitFactor": "Prof. Factor", "AvgHoldDays": "Avg Hold",
    })
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.download_button(
        label="⬇ Download Symbol Ranking CSV",
        data=ranking.to_csv(index=False).encode("utf-8"),
        file_name="symbol_ranking.csv",
        mime="text/csv",
    )

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # 7. Position sizing vs outcome
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Position Sizing vs Outcome")
    st.caption("Do you size up on trades that lose? Clusters of large deployed capital "
               "in the loss zone signal over-sizing on bad trades.")

    sizing = closed[closed["AmountDeployed"] > 0].copy()
    if not sizing.empty:
        fig_sz = px.scatter(
            sizing,
            x="AmountDeployed", y="RealizedPnL",
            color="TradeType",
            color_discrete_map={"Stock": "#4C9BE8", "Option": "#F4A460"},
            size="AmountDeployed", size_max=22,
            hover_data=["Symbol", "CloseDate", "HoldDays", "Outcome"],
            labels={"AmountDeployed": "Amount Deployed ($)",
                    "RealizedPnL": "Realized P&L ($)"},
            title="Capital Deployed vs Realized P&L",
        )
        fig_sz.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig_sz.add_vline(x=sizing["AmountDeployed"].median(),
                         line_dash="dot", line_color="gray",
                         annotation_text="median size",
                         annotation_position="top right")

        # Regression line
        from numpy.polynomial import polynomial as P
        x_s = sizing["AmountDeployed"].values
        y_s = sizing["RealizedPnL"].values
        if len(x_s) > 2:
            coef = P.polyfit(x_s, y_s, 1)
            x_line = [x_s.min(), x_s.max()]
            y_line = [P.polyval(v, coef) for v in x_line]
            fig_sz.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color="white", width=1.5, dash="longdash"),
                name="Trend",
            ))

        fig_sz.update_layout(height=400, margin=dict(t=40))
        st.plotly_chart(fig_sz, use_container_width=True)

        # Quartile breakdown
        sizing["SizeQuartile"] = pd.qcut(
            sizing["AmountDeployed"], q=4,
            labels=["Q1 (small)", "Q2", "Q3", "Q4 (large)"]
        )
        qgrp = (
            sizing.groupby("SizeQuartile", observed=True)
            .agg(
                AvgPnL=("RealizedPnL", "mean"),
                TotalPnL=("RealizedPnL", "sum"),
                WinRate=("Outcome", lambda x: (x == "Win").mean() * 100),
                Count=("RealizedPnL", "count"),
            )
            .reset_index()
        )
        q1, q2, q3, q4 = st.columns(4)
        for col, (_, row) in zip([q1, q2, q3, q4], qgrp.iterrows()):
            col.metric(
                str(row["SizeQuartile"]),
                f"WR {row['WinRate']:.0f}%",
                delta=f"Avg {fmt(row['AvgPnL'])} | {int(row['Count'])} trades",
            )

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # 8. Options deep-dive
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Options Deep-Dive")

    dd = options_deep_dive(opt_pnl_df)
    if not dd:
        st.caption("No options data.")
    else:
        od1, od2, od3, od4 = st.columns(4)
        od1.metric("Expired Worthless",    dd["n_expired"])
        od2.metric("Lost to Expirations",  fmt(abs(dd["lost_to_expiry"])),
                   delta=f"{dd['lost_to_expiry']:+,.2f}", delta_color="inverse")
        od3.metric("Avg DTE — Winners",    f"{dd['avg_dte_winners']:.0f}d")
        od4.metric("Avg DTE — Losers",     f"{dd['avg_dte_losers']:.0f}d")

        od5, od6 = st.columns(2)
        od5.metric("Premium Capture Rate (winners)", f"{dd['premium_capture_pct']:.1f}%")
        od6.metric("Still-Open Premium at Risk",      fmt(dd["still_open_risk"]))

        if dd["avg_dte_losers"] < dd["avg_dte_winners"] * 0.5:
            st.warning(
                "⚠️ Losing options tend to have **shorter DTE** than winners. "
                "You may be buying contracts too close to expiry, leaving little "
                "time for the trade to work."
            )

        # DTE bucket table + chart
        dte_stats = dd["dte_stats"]
        if not dte_stats.empty:
            dte_col1, dte_col2 = st.columns([2, 3])
            with dte_col1:
                st.markdown("**P&L by DTE Bucket (at open)**")
                dte_disp = dte_stats.copy()
                dte_disp["TotalPnL"] = dte_disp["TotalPnL"].map(lambda v: f"${v:,.2f}")
                dte_disp["AvgPnL"]   = dte_disp["AvgPnL"].map(lambda v: f"${v:,.2f}")
                dte_disp["WinRate"]  = dte_disp["WinRate"].map(lambda v: f"{v:.1f}%")
                st.dataframe(dte_disp.rename(columns={
                    "DTEBucket": "DTE at Open", "WinRate": "Win Rate",
                    "TotalPnL": "Total P&L", "AvgPnL": "Avg P&L",
                }), use_container_width=True, hide_index=True)

            with dte_col2:
                fig_dte = go.Figure()
                bar_clr = ["#00c97a" if v >= 0 else "#ff4b4b"
                           for v in dd["dte_stats"]["TotalPnL"]]
                fig_dte.add_trace(go.Bar(
                    x=dte_stats["DTEBucket"].astype(str),
                    y=dte_stats["TotalPnL"],
                    name="Total P&L",
                    marker_color=bar_clr, yaxis="y",
                ))
                fig_dte.add_trace(go.Scatter(
                    x=dte_stats["DTEBucket"].astype(str),
                    y=dte_stats["WinRate"],
                    name="Win Rate %", mode="lines+markers",
                    line=dict(color="#F4A460", width=2), yaxis="y2",
                ))
                fig_dte.update_layout(
                    title="Options P&L & Win Rate by DTE at Open",
                    yaxis=dict(title="Total P&L ($)"),
                    yaxis2=dict(title="Win Rate (%)", overlaying="y",
                                side="right", range=[0, 100]),
                    legend=dict(orientation="h", y=1.1),
                    height=340, margin=dict(t=50),
                )
                st.plotly_chart(fig_dte, use_container_width=True)

        # Scatter: DTE vs P&L
        opt_scatter = dd["opt_with_dte"]
        if not opt_scatter.empty:
            opt_scatter = opt_scatter[
                opt_scatter["Outcome"].isin(["Closed", "Expired Worthless", "Exercised"])
            ].copy()
            color_map = {
                "Closed": "#4C9BE8",
                "Expired Worthless": "#ff4b4b",
                "Exercised": "#F4A460",
            }
            fig_dte_sc = px.scatter(
                opt_scatter,
                x="DTE", y="RealizedPnL",
                color="Outcome",
                color_discrete_map=color_map,
                hover_data=["Underlying", "OptionStrike", "OptionExpiry", "OpenCost"],
                labels={"DTE": "DTE at Open (days)", "RealizedPnL": "P&L ($)"},
                title="DTE at Open vs Realized P&L",
            )
            fig_dte_sc.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
            fig_dte_sc.update_layout(height=360, margin=dict(t=40))
            st.plotly_chart(fig_dte_sc, use_container_width=True)
