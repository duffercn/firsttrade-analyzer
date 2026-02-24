# FirstTrade Analysis

An interactive Streamlit dashboard for analysing your FirstTrade brokerage trading history.

## Features

| Page | What it shows |
|---|---|
| **Account Overview** | Net P&L metrics, monthly stacked bar chart, equity curve + drawdown, SPY benchmark comparison, GitHub-style P&L calendar (day / week / month / year views) |
| **Symbol Analysis** | Per-ticker key metrics, cumulative P&L chart, trade history, dividend history, full options activity breakdown |
| **Daily View** | All trades, options P&L, dividends and margin interest for any selected date |
| **Symbol + Day** | Intersection of a specific ticker and date with cash-flow waterfall chart |
| **Options Analysis** | Portfolio-wide options metrics, P&L by underlying, outcome pie chart, filterable contracts table |
| **Performance Analysis** | Win rate, profit factor, R:R, rolling win-rate chart, hold-time analysis, day-of-week / month patterns, symbol ranking, position sizing scatter, options deep-dive |
| **Open Positions** | Current stock holdings (shares + avg cost), optional live price fetch via Yahoo Finance, portfolio pie chart, open options table |
| **Tax Summary** | Short-term (<365 d) vs long-term (≥365 d) capital-gains split with downloadable trade lists |

All major tables have a **Download CSV** button.

## Setup

```bash
pip install streamlit pandas plotly numpy yfinance watchdog
```

## Usage

1. Export your trading history from FirstTrade as a CSV.
2. Place the CSV in this directory (or upload it via the sidebar widget).
3. Edit `DEFAULT_PATH` in `app.py` if your filename differs from `FT_CSV_export.csv`.
4. Run:

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## CSV format

The app expects the standard FirstTrade CSV export with these columns:

`Symbol, Quantity, Price, Action, Description, TradeDate, SettledDate, Interest, Amount, Commission, Fee, CUSIP, RecordType`

> **Your CSV file is listed in `.gitignore` and will never be committed.**

## Notes

- P&L uses the **average cost method** for stocks and **FIFO contract matching** for options.
- Options are parsed from the `Description` field (CALL/PUT regex).
- The SPY benchmark comparison fetches live data from Yahoo Finance on demand.
- Live prices on the Open Positions page are also fetched on demand (checkbox).
