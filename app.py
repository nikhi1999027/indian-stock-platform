import streamlit as st
import yfinance as yf
import pandas as pd
import feedparser
import urllib.parse
from textblob import TextBlob
from datetime import datetime
import os
import requests

# Streamlit Page Configuration
st.set_page_config(
    page_title="Institutional Global Stock & Options Platform",
    page_icon="⚡",
    layout="wide"
)

# Persistent storage file paths
EQUITY_PORTFOLIO_FILE = "active_trades.csv"
OPTIONS_PORTFOLIO_FILE = "options_active_trades.csv"
US_EQUITY_PORTFOLIO_FILE = "us_active_trades.csv"

# ==============================================================================
# 1. MATHEMATICALLY ACCURATE TECHNICAL INDICATOR ENGINE
# ==============================================================================
def calculate_sma(series, window=20):
    """Calculates Simple Moving Average."""
    return series.rolling(window=window).mean()

def calculate_ema(series, span=20):
    """Calculates Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, window=14):
    """
    Calculates Relative Strength Index (RSI) using Wilder's Smoothing.
    Protected against division-by-zero errors.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    # Prevent division by zero
    safe_loss = avg_loss.replace(0.0, 1e-6)
    rs = avg_gain / safe_loss
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculates MACD Line, Signal Line, and MACD Histogram."""
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Calculates 20-period, 2-std deviation Bollinger Bands."""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

def sanitize_dataframe(df):
    """Ensures MultiIndex columns from yfinance batch downloads are cleanly flattened."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=['Close', 'Volume'])

# ==============================================================================
# 2. PUBLIC DATA PROVIDERS & NEWS SENTIMENT ENGINES
# ==============================================================================
def fetch_indian_tickers():
    """Fetches liquid NSE tickers with reliable fallback."""
    url = "https://raw.githubusercontent.com/kprohith/nse-stock-analysis/master/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        valid = [str(s).strip().upper() for s in df['Symbol'].dropna().unique().tolist()]
        if len(valid) >= 50:
            return valid[:100]
    except Exception:
        pass
    return [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BHARTIARTL", 
        "SBIN", "TATAMOTORS", "LT", "ITC", "AXISBANK", "MARUTI", "BAJFINANCE",
        "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "POWERGRID", "NTPC", "M&M"
    ]

def fetch_us_tickers():
    """Returns top liquid S&P 500 and Nasdaq-100 leaders."""
    return [
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
        "AMD", "COST", "NFLX", "PLTR", "JPM", "V", "CRM", "INTC", "QCOM",
        "AMAT", "MU", "WMT", "UNH", "CAT", "PANW", "DELL", "CRWD", "GS"
    ]

def fetch_fii_dii_live():
    """Fetches public institutional activity flows with browser headers."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    url = "https://www.moneycontrol.com/stocks/fii_dii_activity.php"
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            tables = pd.read_html(response.text)
            for tbl in tables:
                col_str = " ".join([str(c) for c in tbl.columns])
                if "FII" in col_str or "DII" in col_str:
                    return tbl.dropna(how="all").head(5)
    except Exception:
        pass
    return pd.DataFrame({
        "Participant": ["FII / FPI (Foreign)", "DII (Domestic)"],
        "Trading Bias": ["Institutional Cash Flow", "Domestic Mutual Funds"],
        "Strategic Stance": ["Index Momentum Driver", "Market Dip Absorption"]
    })

def fetch_live_news_sentiment(query, market="IN"):
    """Scrapes public Google News RSS feeds for real-time sentiment polarity."""
    if market == "US":
        encoded = urllib.parse.quote(f"{query} stock Wall Street SEC business news")
        rss_url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    else:
        encoded = urllib.parse.quote(f"{query} share price stock market India")
        rss_url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
        
    feed = feedparser.parse(rss_url)
    scores = []
    headlines = []
    for entry in feed.entries[:4]:
        headline = entry.title
        polarity = TextBlob(headline).sentiment.polarity
        scores.append(polarity)
        headlines.append(headline)

    avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    label = "POSITIVE" if avg_score > 0.05 else ("NEGATIVE" if avg_score < -0.05 else "NEUTRAL")
    return {"label": label, "score": avg_score, "headlines": headlines}

# ==============================================================================
# 3. PERSISTENT STORAGE MANAGEMENT
# ==============================================================================
def load_csv_data(filepath, columns):
    if os.path.exists(filepath):
        try:
            loaded_df = pd.read_csv(filepath)
            for col in columns:
                if col not in loaded_df.columns:
                    loaded_df[col] = None
            return loaded_df[columns]
        except Exception:
            return pd.DataFrame(columns=columns)
    return pd.DataFrame(columns=columns)

def save_csv_data(df, filepath):
    df.to_csv(filepath, index=False)

if "eq_portfolio" not in st.session_state:
    st.session_state.eq_portfolio = load_csv_data(
        EQUITY_PORTFOLIO_FILE, 
        ["Symbol", "Entry_Price", "Target_Price", "Stop_Loss", "Entry_Date"]
    )

if "opt_portfolio" not in st.session_state:
    st.session_state.opt_portfolio = load_csv_data(
        OPTIONS_PORTFOLIO_FILE, 
        ["Contract", "Underlying", "Option_Type", "Entry_Premium", "Target_Premium", "Stop_Loss_Premium", "Underlying_SL", "Entry_Date"]
    )

if "us_eq_portfolio" not in st.session_state:
    st.session_state.us_eq_portfolio = load_csv_data(
        US_EQUITY_PORTFOLIO_FILE,
        ["Symbol", "Entry_Price", "Target_Price", "Stop_Loss", "Entry_Date"]
    )

INDEX_DICT = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK"}
FNO_WATCHLIST = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "TATAMOTORS",
    "SBIN", "BHARTIARTL", "AXISBANK", "LT", "KOTAKBANK", "MARUTI", "BAJFINANCE"
]

def round_to_strike(price, step=50):
    return int(round(price / step) * step)

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("⚡ Global Quant Suite")
app_mode = st.sidebar.radio(
    "Select Platform Section:",
    [
        "1. Full-Market Equity Scanner (Indian Market)",
        "2. Active Equity Portfolio & Exit Alerts (Indian Market)",
        "3. High-Probability Options Trading Desk (Indian Market)",
        "4. Active Options Exit Alert Engine (Indian Market)",
        "5. Institutional FII/DII & Market Sentiment",
        "6. US Equities Swing Trading Desk (US Market)",
        "7. Active US Swing Portfolio & Exit Monitor (US Market)"
    ],
    key="nav_platform_mode_selector"
)

# ##############################################################################
# SECTION 1: FULL-MARKET EQUITY SCANNER (INDIAN MARKET)
# ##############################################################################
if app_mode == "1. Full-Market Equity Scanner (Indian Market)":
    st.title("📈 Indian Equities Multi-Indicator Swing Scanner")
    st.caption("Screens liquid NSE equities in real-time using strict 5-factor mathematical confluence.")

    t_scan, t_single, t_backtest, t_sizer = st.tabs([
        "🚀 Market Scanner", "🔍 Stock Deep Dive", "📊 Backtester", "🧮 Position Sizer"
    ])

    with t_scan:
        st.subheader("Automated NSE Market Scanner")
        if st.button("Run Full-Market Scanner", key="btn_exec_in_equity_scan"):
            with st.spinner("Screening liquid NSE universe..."):
                tickers = fetch_indian_tickers()
                yf_tickers = [f"{s}.NS" for s in tickers]
                data = yf.download(yf_tickers, period="60d", group_by='ticker', progress=False)

                candidates = []
                for s in tickers:
                    ticker_key = f"{s}.NS"
                    try:
                        if ticker_key not in data or data[ticker_key].empty:
                            continue
                        df = sanitize_dataframe(data[ticker_key].copy())
                        if len(df) < 30:
                            continue

                        df['EMA_20'] = calculate_ema(df['Close'], span=20)
                        df['EMA_50'] = calculate_ema(df['Close'], span=50)
                        df['SMA_Vol'] = calculate_sma(df['Volume'], window=20)
                        df['RSI'] = calculate_rsi(df['Close'], window=14)
                        df['MACD'], df['Signal'], df['Hist'] = calculate_macd(df['Close'])

                        lat = df.iloc[-1]
                        price, open_p = float(lat['Close']), float(lat['Open'])
                        rsi, ema20, ema50 = float(lat['RSI']), float(lat['EMA_20']), float(lat['EMA_50'])
                        macd, sig = float(lat['MACD']), float(lat['Signal'])
                        vol = float(lat['Volume'])
                        vol_sma = float(lat['SMA_Vol']) if float(lat['SMA_Vol']) > 0 else 1.0
                        vol_ratio = round(vol / vol_sma, 2)

                        # Strict Multi-Factor Confluence
                        if (price > open_p) and (vol_ratio >= 1.25) and (price > ema20 > ema50) and (macd > sig) and (45.0 <= rsi <= 65.0):
                            candidates.append({
                                "Symbol": s,
                                "Price (₹)": round(price, 2),
                                "Green Vol Spike": f"{vol_ratio}x Avg",
                                "Wilder RSI": round(rsi, 1),
                                "MACD Status": "Bullish Crossover",
                                "Target (+5%) (₹)": round(price * 1.05, 2),
                                "Stop Loss (-2%) (₹)": round(price * 0.98, 2)
                            })
                    except Exception:
                        continue

                st.session_state.active_equity_scan_results = candidates

        if "active_equity_scan_results" in st.session_state and st.session_state.active_equity_scan_results:
            cands = st.session_state.active_equity_scan_results
            scan_df = pd.DataFrame(cands)
            st.dataframe(scan_df, use_container_width=True)

            csv = scan_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Scanner Results CSV",
                data=csv,
                file_name=f"NSE_Scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="btn_dl_in_scan_csv"
            )

            st.markdown("---")
            st.markdown("#### **⚡ Add Candidate Directly to Active Equity Exit Engine**")
            c_add1, c_add2 = st.columns([3, 1])
            selected_sym = c_add1.selectbox("Select stock to track:", [c['Symbol'] for c in cands], key="sb_in_equity_cand_sel")
            if c_add2.button("Track This Stock", key="btn_track_in_equity_cand"):
                matched = [c for c in cands if c['Symbol'] == selected_sym][0]
                new_entry = pd.DataFrame([{
                    "Symbol": matched['Symbol'],
                    "Entry_Price": matched['Price (₹)'],
                    "Target_Price": matched['Target (+5%) (₹)'],
                    "Stop_Loss": matched['Stop Loss (-2%) (₹)'],
                    "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }])
                st.session_state.eq_portfolio = pd.concat([st.session_state.eq_portfolio, new_entry], ignore_index=True)
                save_csv_data(st.session_state.eq_portfolio, EQUITY_PORTFOLIO_FILE)
                st.success(f"✅ Added {matched['Symbol']} to Section 2 (Active Equity Portfolio)!")

        elif "active_equity_scan_results" in st.session_state and not st.session_state.active_equity_scan_results:
            st.info("No stocks passed the strict institutional confluence filters today. Capital preserved.")

    with t_single:
        st.subheader("Single Stock Multi-Factor Analysis")
        sym_q = st.text_input("Enter NSE Stock Symbol:", value="RELIANCE", key="ti_single_stock_in").upper()
        if st.button("Analyze Stock", key="btn_single_stock_eval_in"):
            ticker_obj = yf.Ticker(f"{sym_q}.NS")
            hist = ticker_obj.history(period="60d")
            if not hist.empty:
                hist['EMA_20'] = calculate_ema(hist['Close'], span=20)
                hist['EMA_50'] = calculate_ema(hist['Close'], span=50)
                hist['RSI'] = calculate_rsi(hist['Close'], window=14)
                last_bar = hist.iloc[-1]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"₹{round(float(last_bar['Close']), 2)}")
                c2.metric("Wilder's RSI (14)", round(float(last_bar['RSI']), 1))
                c3.metric("20-Day EMA", f"₹{round(float(last_bar['EMA_20']), 2)}")
                st.line_chart(hist[['Close', 'EMA_20', 'EMA_50']].dropna())
            else:
                st.error("No historical market data found for this symbol.")

    with t_backtest:
        st.subheader("1-Year Strategy Performance")
        bt_target = st.text_input("NSE Symbol for Backtest:", value="TCS", key="ti_backtest_sym_in").upper()
        if st.button("Run 1-Year Backtest", key="btn_exec_backtest_in"):
            df_bt = yf.download(f"{bt_target}.NS", period="1y", interval="1d", progress=False)
            df_bt = sanitize_dataframe(df_bt)
            df_bt['EMA_20'] = calculate_ema(df_bt['Close'], span=20)
            df_bt['RSI'] = calculate_rsi(df_bt['Close'], window=14)
            df_bt.dropna(inplace=True)

            capital, position, entry_val, trade_pnls = 100000.0, 0, 0.0, []
            for dt, r in df_bt.iterrows():
                p, r_val, e_val = float(r['Close']), float(r['RSI']), float(r['EMA_20'])
                if position > 0:
                    if p >= (entry_val * 1.05) or p <= (entry_val * 0.98) or r_val > 70.0:
                        trade_pnls.append((position * p) - (position * entry_val))
                        capital += (position * p)
                        position = 0
                elif position == 0 and r_val < 45.0 and p > e_val:
                    position, entry_val, capital = capital / p, p, 0.0

            final_total = capital + (position * float(df_bt['Close'].iloc[-1]))
            roi_pct = round(((final_total - 100000.0) / 100000.0) * 100, 2)
            wins = len([t for t in trade_pnls if t > 0])
            win_pct = round((wins / len(trade_pnls) * 100), 1) if trade_pnls else 0.0

            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Final Capital (₹1L Base)", f"₹{round(final_total, 2)}")
            bc2.metric("Total ROI", f"{roi_pct}%")
            bc3.metric("Win Rate", f"{win_pct}%")

    with t_sizer:
        st.subheader("Position Size Calculator")
        c_cap = st.number_input("Total Trading Capital (₹):", value=100000.0, step=10000.0, key="sz_total_cap_in")
        c_risk = st.number_input("Max Risk (%):", value=2.0, step=0.5, key="sz_risk_percent_in")
        c_entry = st.number_input("Entry Price (₹):", value=2500.0, step=10.0, key="sz_entry_val_in")
        c_sl = st.number_input("Stop Loss Price (₹):", value=2450.0, step=10.0, key="sz_sl_val_in")
        c_tgt = st.number_input("Target Price (₹):", value=2625.0, step=10.0, key="sz_tgt_val_in")

        if st.button("Calculate Position", key="btn_calc_position_in"):
            risk_per_unit = c_entry - c_sl
            if risk_per_unit <= 0:
                st.error("Entry price must be higher than Stop Loss.")
            else:
                qty = int(min((c_cap * (c_risk / 100.0)) // risk_per_unit, c_cap // c_entry))
                st.success(f"**Recommended Quantity:** {qty} Shares")
                st.write(f"**Capital Committed:** ₹{round(qty * c_entry, 2)}")
                st.write(f"**Max Loss:** :red[-₹{round(qty * risk_per_unit, 2)}]")
                st.write(f"**Max Gain:** :green[+₹{round(qty * (c_tgt - c_entry), 2)}]")

# ##############################################################################
# SECTION 2: ACTIVE EQUITY PORTFOLIO & EXIT ALERTS (INDIAN MARKET)
# ##############################################################################
elif app_mode == "2. Active Equity Portfolio & Exit Alerts (Indian Market)":
    st.title("🛡️ Active Equity Portfolio & Dynamic Exit Monitor (NSE)")
    st.caption("Tracks open Indian equity trades against targets (+5%), stop losses (-2%), and live MACD trend reversals.")

    port_df = st.session_state.eq_portfolio

    with st.expander("➕ Manually Add an Equity Position"):
        mc1, mc2, mc3, mc4 = st.columns(4)
        m_sym = mc1.text_input("NSE Symbol (e.g. RELIANCE, TCS):", key="ti_manual_eq_sym_in").upper()
        m_entry = mc2.number_input("Entry Price (₹):", value=2500.0, key="ti_manual_eq_entry_in")
        m_tgt = mc3.number_input("Target Price (₹):", value=2625.0, key="ti_manual_eq_tgt_in")
        m_sl = mc4.number_input("Stop Loss (₹):", value=2450.0, key="ti_manual_eq_sl_in")
        
        if st.button("Save Position to Monitor", key="btn_save_manual_eq_in"):
            if m_sym:
                new_row = pd.DataFrame([{
                    "Symbol": m_sym, "Entry_Price": m_entry, "Target_Price": m_tgt, "Stop_Loss": m_sl,
                    "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }])
                st.session_state.eq_portfolio = pd.concat([st.session_state.eq_portfolio, new_row], ignore_index=True)
                save_csv_data(st.session_state.eq_portfolio, EQUITY_PORTFOLIO_FILE)
                st.success(f"Tracking {m_sym}!")
                st.rerun()

    if port_df.empty:
        st.info("No active Indian equity trades in portfolio.")
    else:
        st.subheader("Live Exit Matrix")
        live_rows = []
        for index, row in port_df.iterrows():
            sym = row['Symbol']
            entry, target, sl = float(row['Entry_Price']), float(row['Target_Price']), float(row['Stop_Loss'])
            try:
                t = yf.Ticker(f"{sym}.NS")
                hist = t.history(period="15d")
                if not hist.empty:
                    cur_p = float(hist['Close'].iloc[-1])
                    hist['MACD'], hist['Signal'], _ = calculate_macd(hist['Close'])
                    cur_macd, cur_sig = float(hist['MACD'].iloc[-1]), float(hist['Signal'].iloc[-1])
                    pnl_pct = round(((cur_p - entry) / entry) * 100, 2)

                    if cur_p >= target:
                        action = "🎯 TAKE PROFIT (Target Reached)"
                    elif cur_p <= sl:
                        action = "🛑 STOP LOSS (Exit Immediately)"
                    elif cur_macd < cur_sig:
                        action = "⚠️ SELL (Trend Reversal Detected)"
                    else:
                        action = "✅ HOLD (Trend Intact)"

                    live_rows.append({
                        "Symbol": sym, "Entry (₹)": entry, "Live Price (₹)": round(cur_p, 2),
                        "P&L (%)": f"{pnl_pct}%", "Target (₹)": target, "Stop Loss (₹)": sl, "Exit Signal": action
                    })
            except Exception:
                continue

        if live_rows:
            st.dataframe(pd.DataFrame(live_rows), use_container_width=True)

        st.markdown("---")
        del_sym = st.selectbox("Select Trade to Remove:", port_df['Symbol'].tolist(), key="sb_delete_equity_trade_in")
        if st.button("Remove Trade", key="btn_remove_equity_trade_in"):
            st.session_state.eq_portfolio = port_df[port_df['Symbol'] != del_sym].reset_index(drop=True)
            save_csv_data(st.session_state.eq_portfolio, EQUITY_PORTFOLIO_FILE)
            st.success(f"Removed {del_sym} from tracking.")
            st.rerun()

# ##############################################################################
# SECTION 3: HIGH-PROBABILITY OPTIONS TRADING DESK (INDIAN MARKET)
# ##############################################################################
elif app_mode == "3. High-Probability Options Trading Desk (Indian Market)":
    st.title("🎯 High-Probability Options Trading Desk (NSE)")
    st.caption("Quantitative Confluence Strategy: EMA 20/50 + Bollinger Bands + MACD Histogram + Wilder's RSI")

    opt_choice = st.radio("Choose Trading Segment:", ["Benchmark Indices (Nifty / Bank Nifty)", "Top F&O Momentum Stocks"], horizontal=True, key="rb_options_segment_in")

    if opt_choice == "Benchmark Indices (Nifty / Bank Nifty)":
        st.subheader("Benchmark Index Directional Engine")
        idx_name = st.selectbox("Select Index:", list(INDEX_DICT.keys()), key="sb_index_choice_in")
        idx_ticker = INDEX_DICT[idx_name]

        if st.button("Evaluate High-Probability Setup", key="btn_eval_index_options_in"):
            with st.spinner(f"Analyzing multi-factor confluence for {idx_name}..."):
                df_idx = yf.download(idx_ticker, period="60d", interval="1d", progress=False)
                df_idx = sanitize_dataframe(df_idx)

                if df_idx.empty or len(df_idx) < 30:
                    st.error("Insufficient market data available for index.")
                else:
                    df_idx['EMA_20'] = calculate_ema(df_idx['Close'], span=20)
                    df_idx['EMA_50'] = calculate_ema(df_idx['Close'], span=50)
                    df_idx['RSI'] = calculate_rsi(df_idx['Close'], window=14)
                    df_idx['MACD'], df_idx['Signal'], df_idx['Hist'] = calculate_macd(df_idx['Close'])

                    lat = df_idx.iloc[-1]
                    price, ema20, ema50 = float(lat['Close']), float(lat['EMA_20']), float(lat['EMA_50'])
                    rsi, hist = float(lat['RSI']), float(lat['Hist'])

                    news_sent = fetch_live_news_sentiment(idx_name, market="IN")
                    step = 50 if "NIFTY 50" in idx_name else 100
                    atm_strike = round_to_strike(price, step=step)

                    call_cond = (price > ema20 > ema50) and (rsi >= 55.0) and (hist > 0) and (news_sent['score'] >= 0)
                    put_cond = (price < ema20 < ema50) and (rsi <= 45.0) and (hist < 0) and (news_sent['score'] <= 0)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Spot Level", f"{round(price, 1)}")
                    col2.metric("Wilder's RSI (14)", round(rsi, 1))
                    col3.metric("MACD Histogram", round(hist, 2))
                    col4.metric("News Sentiment", news_sent['label'])

                    st.markdown("---")
                    if call_cond:
                        contract_name = f"{idx_name} {atm_strike} CE"
                        st.success(f"### 🟢 HIGH PROBABILITY CALL SETUP: BUY {contract_name}")
                        st.write(f"**Strike:** ATM {atm_strike} Call (CE) | **Underlying SL:** Spot below EMA 20 (₹{round(ema20, 1)})")
                        st.session_state.active_index_signal = {
                            "Contract": contract_name, "Underlying": "NIFTY" if "NIFTY 50" in idx_name else "BANKNIFTY",
                            "Option_Type": "CE", "Underlying_SL": round(ema20, 1)
                        }
                    elif put_cond:
                        contract_name = f"{idx_name} {atm_strike} PE"
                        st.error(f"### 🔴 HIGH PROBABILITY PUT SETUP: BUY {contract_name}")
                        st.write(f"**Strike:** ATM {atm_strike} Put (PE) | **Underlying SL:** Spot above EMA 20 (₹{round(ema20, 1)})")
                        st.session_state.active_index_signal = {
                            "Contract": contract_name, "Underlying": "NIFTY" if "NIFTY 50" in idx_name else "BANKNIFTY",
                            "Option_Type": "PE", "Underlying_SL": round(ema20, 1)
                        }
                    else:
                        st.session_state.active_index_signal = None
                        st.warning("### ⚠️ NO TRADE SETUP — STAND ASIDE")

        if "active_index_signal" in st.session_state and st.session_state.active_index_signal:
            sig = st.session_state.active_index_signal
            prem_in = st.number_input("Bought Option Premium (₹):", value=120.0, step=5.0, key="ni_index_prem_in_opt")
            if st.button("Add to Options Exit Engine", key="btn_add_idx_to_exit_opt"):
                new_row = pd.DataFrame([{
                    "Contract": sig["Contract"], "Underlying": sig["Underlying"], "Option_Type": sig["Option_Type"],
                    "Entry_Premium": prem_in, "Target_Premium": round(prem_in * 1.30, 2), "Stop_Loss_Premium": round(prem_in * 0.80, 2),
                    "Underlying_SL": sig["Underlying_SL"], "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }])
                st.session_state.opt_portfolio = pd.concat([st.session_state.opt_portfolio, new_row], ignore_index=True)
                save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
                st.success(f"✅ Added {sig['Contract']} to Section 4!")

    else:
        st.subheader("F&O Stock High-Win-Rate Momentum Scanner")
        if st.button("Scan F&O Universe for Confluence Setups", key="btn_scan_fno_momentum_in"):
            with st.spinner("Filtering F&O universe..."):
                fno_tickers = [f"{s}.NS" for s in FNO_WATCHLIST]
                data_fno = yf.download(fno_tickers, period="60d", group_by='ticker', progress=False)

                fno_cands = []
                for s in FNO_WATCHLIST:
                    t_key = f"{s}.NS"
                    try:
                        if t_key not in data_fno or data_fno[t_key].empty:
                            continue
                        df = sanitize_dataframe(data_fno[t_key].copy())
                        if len(df) < 30:
                            continue

                        df['EMA_20'] = calculate_ema(df['Close'], span=20)
                        df['EMA_50'] = calculate_ema(df['Close'], span=50)
                        df['SMA_Vol'] = calculate_sma(df['Volume'], window=20)
                        df['RSI'] = calculate_rsi(df['Close'], window=14)
                        df['MACD'], df['Signal'], df['Hist'] = calculate_macd(df['Close'])

                        lat = df.iloc[-1]
                        price, open_p, rsi, ema20, ema50, hist = float(lat['Close']), float(lat['Open']), float(lat['RSI']), float(lat['EMA_20']), float(lat['EMA_50']), float(lat['Hist'])
                        vol = float(lat['Volume'])
                        vol_sma = float(lat['SMA_Vol']) if float(lat['SMA_Vol']) > 0 else 1.0
                        vol_ratio = round(vol / vol_sma, 2)
                        step = 100 if price > 2000 else (50 if price > 500 else 10)
                        strike = round_to_strike(price, step=step)

                        if (price > open_p) and (price > ema20 > ema50) and (vol_ratio >= 1.3) and (55.0 <= rsi <= 70.0) and (hist > 0):
                            fno_cands.append({
                                "Stock": s, "Spot Price (₹)": round(price, 2), "Recommended Contract": f"{s} {strike} CE",
                                "Option_Type": "CE", "Volume Surge": f"{vol_ratio}x Avg", "RSI": round(rsi, 1),
                                "Setup": "Long Breakout", "Underlying SL (₹)": round(ema20, 2)
                            })
                        elif (price < open_p) and (price < ema20 < ema50) and (vol_ratio >= 1.3) and (30.0 <= rsi <= 45.0) and (hist < 0):
                            fno_cands.append({
                                "Stock": s, "Spot Price (₹)": round(price, 2), "Recommended Contract": f"{s} {strike} PE",
                                "Option_Type": "PE", "Volume Surge": f"{vol_ratio}x Avg", "RSI": round(rsi, 1),
                                "Setup": "Short Breakdown", "Underlying SL (₹)": round(ema20, 2)
                            })
                    except Exception:
                        continue

                st.session_state.active_fno_scan_results = fno_cands

        if "active_fno_scan_results" in st.session_state and st.session_state.active_fno_scan_results:
            results = st.session_state.active_fno_scan_results
            st.dataframe(pd.DataFrame(results), use_container_width=True)

            st.markdown("---")
            cf1, cf2, cf3 = st.columns([2, 1, 1])
            sel_fno_contract = cf1.selectbox("Select Option Contract:", [r["Recommended Contract"] for r in results], key="sb_fno_contract_sel_in")
            fno_prem = cf2.number_input("Option Premium (₹):", value=50.0, step=2.0, key="ni_fno_premium_in_desk")

            if cf3.button("Add to Exit Engine", key="btn_add_fno_to_exit_desk"):
                matched_fno = [r for r in results if r["Recommended Contract"] == sel_fno_contract][0]
                new_row = pd.DataFrame([{
                    "Contract": matched_fno["Recommended Contract"], "Underlying": matched_fno["Stock"], "Option_Type": matched_fno["Option_Type"],
                    "Entry_Premium": fno_prem, "Target_Premium": round(fno_prem * 1.30, 2), "Stop_Loss_Premium": round(fno_prem * 0.80, 2),
                    "Underlying_SL": matched_fno["Underlying SL (₹)"], "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }])
                st.session_state.opt_portfolio = pd.concat([st.session_state.opt_portfolio, new_row], ignore_index=True)
                save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
                st.success(f"✅ Added {matched_fno['Recommended Contract']} to Section 4!")

# ##############################################################################
# SECTION 4: ACTIVE OPTIONS EXIT ALERT ENGINE (INDIAN MARKET)
# ##############################################################################
elif app_mode == "4. Active Options Exit Alert Engine (Indian Market)":
    st.title("🛡️ Real-Time Options Exit Alert Engine (NSE)")
    st.caption("Monitors active option positions, tracking target profits (+30%), stop-loss (-20%), and underlying EMA reversals.")

    port_df = st.session_state.opt_portfolio

    with st.expander("➕ Manually Log an Active Option Trade"):
        oc1, oc2, oc3 = st.columns(3)
        c_underlying = oc1.selectbox("Underlying Asset:", ["NIFTY", "BANKNIFTY"] + FNO_WATCHLIST, key="sb_log_und_in")
        c_type = oc2.selectbox("Option Type:", ["CE", "PE"], key="sb_log_type_in")
        c_strike = oc3.number_input("Strike Price:", value=25000, step=50, key="ni_log_strike_in")

        oc4, oc5 = st.columns(2)
        c_entry_prem = oc4.number_input("Your Entry Premium (₹):", value=120.0, step=5.0, key="ni_log_prem_in")
        c_underlying_sl = oc5.number_input("Underlying Stop-Loss Level (₹):", value=24800.0, step=10.0, key="ni_log_und_sl_in")

        if st.button("Start Tracking Exit Signals", key="btn_start_tracking_options_in"):
            contract_code = f"{c_underlying} {c_strike} {c_type}"
            new_trade = pd.DataFrame([{
                "Contract": contract_code, "Underlying": c_underlying, "Option_Type": c_type,
                "Entry_Premium": c_entry_prem, "Target_Premium": round(c_entry_prem * 1.30, 2),
                "Stop_Loss_Premium": round(c_entry_prem * 0.80, 2), "Underlying_SL": c_underlying_sl,
                "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
            }])
            st.session_state.opt_portfolio = pd.concat([st.session_state.opt_portfolio, new_trade], ignore_index=True)
            save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
            st.success(f"Tracking {contract_code}!")
            st.rerun()

    if port_df.empty:
        st.info("No active option trades currently logged.")
    else:
        st.subheader("Live Exit Status Matrix")
        live_reports = []
        for idx, row in port_df.iterrows():
            und, contract, op_type = row['Underlying'], row['Contract'], row['Option_Type']
            entry_prem, tgt_prem, sl_prem, und_sl = float(row['Entry_Premium']), float(row['Target_Premium']), float(row['Stop_Loss_Premium']), float(row['Underlying_SL'])
            ticker_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
            query_ticker = ticker_map.get(und, f"{und}.NS")

            try:
                t = yf.Ticker(query_ticker)
                h = t.history(period="10d")
                if not h.empty:
                    cur_spot = float(h['Close'].iloc[-1])
                    h['EMA_20'] = calculate_ema(h['Close'], span=20)
                    ema20 = float(h['EMA_20'].iloc[-1])

                    if op_type == "CE" and cur_spot <= und_sl:
                        status = "🛑 EXIT NOW: Underlying SL Breached"
                    elif op_type == "PE" and cur_spot >= und_sl:
                        status = "🛑 EXIT NOW: Underlying SL Breached"
                    elif op_type == "CE" and cur_spot < ema20:
                        status = "⚠️ CAUTION: Spot Fell Below 20-EMA"
                    elif op_type == "PE" and cur_spot > ema20:
                        status = "⚠️ CAUTION: Spot Rose Above 20-EMA"
                    else:
                        status = "✅ HOLD: Trend Intact"

                    live_reports.append({
                        "Contract": contract, "Entry Premium (₹)": entry_prem, "Target (+30%) (₹)": tgt_prem,
                        "SL (-20%) (₹)": sl_prem, "Live Spot (₹)": round(cur_spot, 1), "Underlying SL (₹)": und_sl, "Exit Signal": status
                    })
            except Exception:
                continue

        if live_reports:
            st.dataframe(pd.DataFrame(live_reports), use_container_width=True)

        st.markdown("---")
        del_contract = st.selectbox("Select Trade to Remove:", port_df['Contract'].tolist(), key="sb_del_opt_contract_in")
        if st.button("Remove Trade", key="btn_remove_opt_trade_in"):
            st.session_state.opt_portfolio = port_df[port_df['Contract'] != del_contract].reset_index(drop=True)
            save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
            st.success(f"Removed {del_contract} from tracking.")
            st.rerun()

# ##############################################################################
# SECTION 5: INSTITUTIONAL FII/DII & MARKET SENTIMENT
# ##############################################################################
elif app_mode == "5. Institutional FII/DII & Market Sentiment":
    st.title("🏛️ Institutional Activity & Market Sentiment")
    st.caption("Foreign & Domestic Institutional Investors drive large-scale liquidity trends in the Indian Market.")

    st.subheader("Daily Institutional Cash Flow Activity (NSE/BSE)")
    fii_data = fetch_fii_dii_live()
    st.table(fii_data)

    st.markdown("---")
    st.subheader("Live Google News Financial Sentiment Analyzer")
    query_sym = st.text_input("Enter Stock, Sector, or Index Name:", value="Nifty 50", key="ti_nlp_query_in")
    
    if st.button("Analyze News Polarity", key="btn_analyze_nlp_in"):
        with st.spinner("Parsing live RSS headlines..."):
            res = fetch_live_news_sentiment(query_sym, market="IN")
            sc1, sc2 = st.columns(2)
            sc1.metric("Market Headline Bias", res['label'])
            sc2.metric("Polarity Score (-1.0 to +1.0)", res['score'])
            st.write("#### Recent Headlining News:")
            for h in res['headlines']:
                st.write(f"- {h}")

# ##############################################################################
# SECTION 6: US EQUITIES SWING TRADING DESK
# ##############################################################################
elif app_mode == "6. US Equities Swing Trading Desk (US Market)":
    st.title("🗽 US Equities Multi-Indicator Swing Desk")
    st.caption("Screens high-liquidity S&P 500 and Nasdaq-100 leaders using strict institutional momentum rules (Public Data).")

    us_tab_scan, us_tab_single = st.tabs(["🚀 US Market Scanner", "🔍 US Stock Deep Dive & Sentiment"])

    with us_tab_scan:
        st.subheader("Automated S&P 500 / Nasdaq Swing Scanner")
        st.write("Filters for: Green Bar (`Close > Open`), Volume Expansion ($\ge 1.25\\times$), Trend (`Price > EMA 20 > EMA 50`), Positive MACD, and RSI ($45\\text{--}65$).")

        if st.button("Run US Market Scanner", key="btn_exec_us_scan_desk"):
            with st.spinner("Downloading live public market data for US leaders..."):
                us_symbols = fetch_us_tickers()
                data_us = yf.download(us_symbols, period="60d", group_by='ticker', progress=False)

                us_candidates = []
                for s in us_symbols:
                    try:
                        if s not in data_us or data_us[s].empty:
                            continue
                        df = sanitize_dataframe(data_us[s].copy())
                        if len(df) < 30:
                            continue

                        df['EMA_20'] = calculate_ema(df['Close'], span=20)
                        df['EMA_50'] = calculate_ema(df['Close'], span=50)
                        df['SMA_Vol'] = calculate_sma(df['Volume'], window=20)
                        df['RSI'] = calculate_rsi(df['Close'], window=14)
                        df['MACD'], df['Signal'], df['Hist'] = calculate_macd(df['Close'])

                        lat = df.iloc[-1]
                        price, open_p = float(lat['Close']), float(lat['Open'])
                        rsi = float(lat['RSI'])
                        ema20, ema50 = float(lat['EMA_20']), float(lat['EMA_50'])
                        macd, sig = float(lat['MACD']), float(lat['Signal'])
                        vol = float(lat['Volume'])
                        vol_sma = float(lat['SMA_Vol']) if float(lat['SMA_Vol']) > 0 else 1.0
                        vol_ratio = round(vol / vol_sma, 2)

                        # Strict US Swing Setup Rules:
                        if (price > open_p) and (vol_ratio >= 1.25) and (price > ema20 > ema50) and (macd > sig) and (45.0 <= rsi <= 65.0):
                            target_us = round(price * 1.06, 2)       # +6% Target for US Swings
                            stop_loss_us = round(price * 0.975, 2)   # -2.5% Stop Loss
                            us_candidates.append({
                                "Symbol": s,
                                "Price ($)": round(price, 2),
                                "Volume Surge": f"{vol_ratio}x Avg",
                                "Wilder RSI": round(rsi, 1),
                                "MACD Status": "Bullish Expansion",
                                "Target (+6%) ($)": target_us,
                                "Stop Loss (-2.5%) ($)": stop_loss_us
                            })
                    except Exception:
                        continue

                st.session_state.active_us_scan_results = us_candidates

        if "active_us_scan_results" in st.session_state and st.session_state.active_us_scan_results:
            us_cands = st.session_state.active_us_scan_results
            us_scan_df = pd.DataFrame(us_cands)
            st.dataframe(us_scan_df, use_container_width=True)

            csv_us = us_scan_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download US Scanner Results CSV",
                data=csv_us,
                file_name=f"US_Stock_Scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="btn_download_us_csv_desk"
            )

            st.markdown("---")
            st.markdown("#### **⚡ Add Candidate to Active US Swing Exit Monitor**")
            us_add1, us_add2 = st.columns([3, 1])
            sel_us_stock = us_add1.selectbox("Select US stock to track:", [c['Symbol'] for c in us_cands], key="sb_us_cand_select_desk")
            
            if us_add2.button("Track This US Stock", key="btn_track_us_stock_desk"):
                matched_us = [c for c in us_cands if c['Symbol'] == sel_us_stock][0]
                new_us_entry = pd.DataFrame([{
                    "Symbol": matched_us['Symbol'],
                    "Entry_Price": matched_us['Price ($)'],
                    "Target_Price": matched_us['Target (+6%) ($)'],
                    "Stop_Loss": matched_us['Stop Loss (-2.5%) ($)'],
                    "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }])
                st.session_state.us_eq_portfolio = pd.concat([st.session_state.us_eq_portfolio, new_us_entry], ignore_index=True)
                save_csv_data(st.session_state.us_eq_portfolio, US_EQUITY_PORTFOLIO_FILE)
                st.success(f"✅ Added {matched_us['Symbol']} to Section 7 (Active US Swing Portfolio & Exit Monitor)!")

        elif "active_us_scan_results" in st.session_state and not st.session_state.active_us_scan_results:
            st.info("No US stocks met the strict institutional criteria today. Capital is safe.")

    with us_tab_single:
        st.subheader("US Stock Deep Dive & Wall Street Sentiment")
        us_sym = st.text_input("Enter US Ticker Symbol (e.g. NVDA, AAPL, MSFT, TSLA):", value="NVDA", key="ti_us_single_sym_desk").upper()
        
        if st.button("Analyze US Stock", key="btn_us_single_eval_desk"):
            t_us = yf.Ticker(us_sym)
            hist_us = t_us.history(period="60d")
            if not hist_us.empty:
                hist_us['EMA_20'] = calculate_ema(hist_us['Close'], span=20)
                hist_us['EMA_50'] = calculate_ema(hist_us['Close'], span=50)
                hist_us['RSI'] = calculate_rsi(hist_us['Close'], window=14)
                hist_us['MACD'], hist_us['Signal'], hist_us['Hist'] = calculate_macd(hist_us['Close'])

                last_u = hist_us.iloc[-1]
                u_price = float(last_u['Close'])
                u_rsi = float(last_u['RSI'])
                u_ema20 = float(last_u['EMA_20'])
                u_ema50 = float(last_u['EMA_50'])

                u_sent = fetch_live_news_sentiment(us_sym, market="US")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Price ($)", f"${round(u_price, 2)}")
                c2.metric("Wilder's RSI (14)", round(u_rsi, 1))
                c3.metric("20-Day EMA ($)", f"${round(u_ema20, 2)}")
                c4.metric("US News Sentiment", u_sent['label'])

                st.line_chart(hist_us[['Close', 'EMA_20', 'EMA_50']].dropna())

                st.write("#### Wall Street Headlines Polarity:")
                st.write(f"**Score:** {u_sent['score']} (-1.0 to +1.0)")
                for hl in u_sent['headlines']:
                    st.write(f"- {hl}")
            else:
                st.error("No data found for this US ticker symbol.")

# ##############################################################################
# SECTION 7: ACTIVE US SWING PORTFOLIO & EXIT MONITOR
# ##############################################################################
elif app_mode == "7. Active US Swing Portfolio & Exit Monitor (US Market)":
    st.title("🛡️ Active US Swing Portfolio & Exit Monitor (Wall Street)")
    st.caption("Monitors active US equity swing trades against profit targets (+6%), stop losses (-2.5%), and 20-EMA trend breakdowns.")

    us_port_df = st.session_state.us_eq_portfolio

    with st.expander("➕ Manually Add a US Stock Position"):
        uc1, uc2, uc3, uc4 = st.columns(4)
        u_sym = uc1.text_input("US Ticker (e.g. NVDA, AAPL):", key="ti_manual_us_sym_desk").upper()
        u_entry = uc2.number_input("Entry Price ($):", value=120.0, step=5.0, key="ti_manual_us_entry_desk")
        u_tgt = uc3.number_input("Target Price ($):", value=127.2, step=5.0, key="ti_manual_us_tgt_desk")
        u_sl = uc4.number_input("Stop Loss ($):", value=117.0, step=5.0, key="ti_manual_us_sl_desk")

        if st.button("Save US Trade to Monitor", key="btn_save_manual_us_desk"):
            if u_sym:
                new_us_row = pd.DataFrame([{
                    "Symbol": u_sym, "Entry_Price": u_entry, "Target_Price": u_tgt, "Stop_Loss": u_sl,
                    "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }])
                st.session_state.us_eq_portfolio = pd.concat([st.session_state.us_eq_portfolio, new_us_row], ignore_index=True)
                save_csv_data(st.session_state.us_eq_portfolio, US_EQUITY_PORTFOLIO_FILE)
                st.success(f"Tracking {u_sym}!")
                st.rerun()

    if us_port_df.empty:
        st.info("No active US swing positions in portfolio. Run the US scanner in Section 6 to populate.")
    else:
        st.subheader("Live US Exit Status Matrix")
        live_us_rows = []
        for index, row in us_port_df.iterrows():
            sym = row['Symbol']
            entry, target, sl = float(row['Entry_Price']), float(row['Target_Price']), float(row['Stop_Loss'])

            try:
                t = yf.Ticker(sym)
                hist = t.history(period="15d")
                if not hist.empty:
                    cur_p = float(hist['Close'].iloc[-1])
                    hist['EMA_20'] = calculate_ema(hist['Close'], span=20)
                    ema20 = float(hist['EMA_20'].iloc[-1])
                    pnl_pct = round(((cur_p - entry) / entry) * 100, 2)

                    if cur_p >= target:
                        action = "🎯 TAKE PROFIT (Target Reached)"
                    elif cur_p <= sl:
                        action = "🛑 STOP LOSS (Exit Immediately)"
                    elif cur_p < ema20:
                        action = "⚠️ SELL (Fell Below 20-EMA)"
                    else:
                        action = "✅ HOLD (Trend Intact)"

                    live_us_rows.append({
                        "Symbol": sym, "Entry ($)": entry, "Live Price ($)": round(cur_p, 2),
                        "P&L (%)": f"{pnl_pct}%", "Target ($)": target, "Stop Loss ($)": sl, "Exit Signal": action
                    })
            except Exception:
                continue

        if live_us_rows:
            st.dataframe(pd.DataFrame(live_us_rows), use_container_width=True)

        st.markdown("---")
        del_us_sym = st.selectbox("Select US Trade to Remove:", us_port_df['Symbol'].tolist(), key="sb_delete_us_trade_desk")
        if st.button("Remove US Trade", key="btn_remove_us_trade_desk"):
            st.session_state.us_eq_portfolio = us_port_df[us_port_df['Symbol'] != del_us_sym].reset_index(drop=True)
            save_csv_data(st.session_state.us_eq_portfolio, US_EQUITY_PORTFOLIO_FILE)
            st.success(f"Removed {del_us_sym} from tracking.")
            st.rerun()
