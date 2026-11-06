import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="15-Year Cross-Asset Lead-Lag Backtester", layout="wide")
st.title("15-Year Cross-Market Predictive Backtester (TLT, GLD, SPY, QQQ, VXX)")

TICKERS = ["TLT", "GLD", "SPY", "QQQ", "VXX", "^VIX"]

@st.cache_data(ttl=3600)
def load_15yr_data(years=15):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    
    raw = yf.download(TICKERS, start=start_date, end=end_date, auto_adjust=True, progress=False)
    closes = raw["Close"].copy()
    opens = raw["Open"].copy()
    
    # Stitch VXX prior to Jan 2018 using scaled ^VIX daily % changes (0.48 beta proxy for 30-day VIX futures)
    vxx_rets = closes["VXX"].pct_change()
    vix_rets = closes["^VIX"].pct_change() * 0.48
    stitched_vxx_rets = vxx_rets.combine_first(vix_rets)
    
    # Reconstruct continuous Close & Open returns
    assets = ["TLT", "GLD", "SPY", "QQQ", "VXX"]
    ret_cc = closes[["TLT", "GLD", "SPY", "QQQ"]].pct_change()
    ret_cc["VXX"] = stitched_vxx_rets
    
    # Intraday return (Day 2 Open to Day 2 Close) for realistic execution after Day 1 Close signal
    ret_oc = (closes[["TLT", "GLD", "SPY", "QQQ"]] - opens[["TLT", "GLD", "SPY", "QQQ"]]) / opens[["TLT", "GLD", "SPY", "QQQ"]]
    vxx_oc = (closes["VXX"] - opens["VXX"]) / opens["VXX"]
    vix_oc = ((closes["^VIX"] - opens["^VIX"]) / opens["^VIX"]) * 0.48
    ret_oc["VXX"] = vxx_oc.combine_first(vix_oc)
    
    return closes[assets], ret_cc.dropna(), ret_oc.reindex(ret_cc.dropna().index)

closes, ret_cc, ret_oc = load_15yr_data(15)

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("Lead-Lag Backtest Settings")
signal_asset = st.sidebar.selectbox("Day 1 Signal Asset (X)", ["TLT", "GLD", "SPY", "QQQ", "VXX"], index=0)
target_asset = st.sidebar.selectbox("Day 2 Target Asset (Y)", ["TLT", "GLD", "SPY", "QQQ", "VXX"], index=1)

condition_type = st.sidebar.selectbox("Day 1 Trigger Mode", ["Return > Threshold (%)", "Return < Threshold (%)", "Z-Score > Threshold", "Z-Score < Threshold"])
threshold_val = st.sidebar.number_input("Threshold Value (e.g. 1.0 for 1% or 1.5 for Z)", value=1.0, step=0.25)
trade_direction = st.sidebar.radio("Day 2 Trade Direction on Target", ["Long", "Short"], index=0)
exec_mode = st.sidebar.radio("Execution Timing", ["Day 1 Close to Day 2 Close", "Day 2 Open to Day 2 Close (No Lookahead)"], index=0)

initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000, step=10000)
commission = st.sidebar.number_input("Commission per Trade ($)", value=1.0, step=0.5)

# --- 1. LEAD-LAG CORRELATION MATRIX ---
st.subheader("1. 15-Year Cross-Correlation Matrix: Day 1 (Rows) → Day 2 (Columns)")
assets = ["TLT", "GLD", "SPY", "QQQ", "VXX"]
lag_corr = pd.DataFrame(index=assets, columns=assets, dtype=float)

for sig in assets:
    for tgt in assets:
        # Correlate sig at (t) with tgt at (t+1)
        lag_corr.loc[sig, tgt] = ret_cc[sig].corr(ret_cc[tgt].shift(-1))

st.dataframe(lag_corr.style.format("{:.4f}").background_gradient(cmap="RdYlGn", vmin=-0.08, vmax=0.08))

# --- 2. CONDITIONAL BACKTEST ENGINE ---
st.subheader(f"2. Backtest Results: {signal_asset} (Day 1) Predicting {target_asset} (Day 2)")

df_bt = pd.DataFrame(index=ret_cc.index)
df_bt["Sig_Ret"] = ret_cc[signal_asset] * 100.0  # in %
rolling_mean = ret_cc[signal_asset].rolling(20).mean()
rolling_std = ret_cc[signal_asset].rolling(20).std()
df_bt["Sig_Z"] = (ret_cc[signal_asset] - rolling_mean) / rolling_std

if exec_mode == "Day 1 Close to Day 2 Close":
    df_bt["Tgt_Next_Ret"] = ret_cc[target_asset].shift(-1)
else:
    df_bt["Tgt_Next_Ret"] = ret_oc[target_asset].shift(-1)

df_bt = df_bt.dropna()

# Evaluate trigger on Day 1
if condition_type == "Return > Threshold (%)":
    df_bt["Signal"] = df_bt["Sig_Ret"] > threshold_val
elif condition_type == "Return < Threshold (%)":
    df_bt["Signal"] = df_bt["Sig_Ret"] < threshold_val
elif condition_type == "Z-Score > Threshold":
    df_bt["Signal"] = df_bt["Sig_Z"] > threshold_val
else:
    df_bt["Signal"] = df_bt["Sig_Z"] < threshold_val

direction_mult = 1.0 if trade_direction == "Long" else -1.0
df_bt["Strat_Ret"] = np.where(df_bt["Signal"], df_bt["Tgt_Next_Ret"] * direction_mult, 0.0)

# Equity curve with compounding and $ commissions
equity = [initial_capital]
trades = []
for idx, row in df_bt.iterrows():
    if row["Signal"]:
        gross_pnl = equity[-1] * row["Strat_Ret"]
        net_pnl = gross_pnl - (2 * commission)  # Entry + Exit
        equity.append(equity[-1] + net_pnl)
        trades.append(net_pnl / equity[-2])
    else:
        equity.append(equity[-1])

df_bt["Equity"] = equity[1:]
trades_s = pd.Series(trades)

# Metrics Summary
total_trades = len(trades_s)
win_rate = (trades_s > 0).mean() * 100 if total_trades > 0 else 0.0
avg_trade = trades_s.mean() * 100 if total_trades > 0 else 0.0
gross_wins = trades_s[trades_s > 0].sum()
gross_losses = abs(trades_s[trades_s < 0].sum())
profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else np.nan
max_dd = ((df_bt["Equity"] / df_bt["Equity"].cummax()) - 1.0).min() * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Trades", f"{total_trades}")
c2.metric("Win Rate", f"{win_rate:.1f}%")
c3.metric("Avg Net Trade", f"{avg_trade:.2f}%")
c4.metric("Profit Factor", f"{profit_factor:.2f}")
c5.metric("Max Drawdown", f"{max_dd:.2f}%")

st.line_chart(df_bt[["Equity"]])
