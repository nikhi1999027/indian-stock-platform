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
    return series.rolling(window=window).mean()

def calculate_ema(series, span=20):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, window=14):
    """Wilder's smoothed RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 0.00001)
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

# ==============================================================================
# 2. PUBLIC DATA PROVIDERS & NEWS SENTIMENT ENGINES
# ==============================================================================
def fetch_indian_tickers():
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
    """Returns top liquid S&P 500 / Nasdaq-100 swing trading equities."""
    return [
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
        "AMD", "COST", "NFLX", "PLTR", "JPM", "V", "CRM", "INTC", "QCOM",
        "AMAT", "MU", "WMT", "UNH", "CAT", "PANW", "DELL", "CRWD", "GS"
    ]

def fetch_fii_dii_live():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    url = "https://www.moneycontrol.com/stocks/fii_dii_activity.php"
    try:
        response = requests.get(url, headers=headers, timeout=6)
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
    """
    Public Google News RSS scraper for Indian (IN) or US markets (US).
    Computes natural language sentiment polarity (-1.0 to +1.0).
    """
    if market == "US":
        encoded = urllib.parse.quote(f"{query} stock market Wall Street SEC news")
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
            return pd.read_csv(filepath)
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
    key="nav_platform_mode"
)

# ##############################################################################
# SECTION 1: FULL-MARKET EQUITY SCANNER (INDIAN MARKET)
# ##############################################################################
if app_mode == "1. Full-Market Equity Scanner (Indian Market)":
    st.title("📈 Indian Equities Multi-Indicator Swing Scanner")
    st.caption("Screens liquid NSE equities in real-time using 5-factor mathematical confluence.")

    t_scan, t_single, t_backtest, t_sizer = st.tabs([
        "🚀 Market Scanner", "🔍 Stock Deep Dive", "📊 Backtester", "🧮 Position Sizer"
    ])

    with t_scan:
        st.subheader("Automated NSE Market Scanner")
        if st.button("Run Full-Market Scanner", key="btn_exec_equity_scan"):
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
                        df = data[ticker_key].dropna(subset=['Close', 'Volume'])
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

                        if (price > open_p) and (vol_ratio >= 1.25) and (price > ema20 > ema50) and (macd > sig) and (45.0 <= rsi <= 65.0):
                            candidates.append({
                                "Symbol": s,
                                "Price (₹)": round(price, 2),
                                "Green Vol Spike": f"{vol_ratio}x Avg",
                                "RSI": round(rsi, 1),
                                "MACD Status": "Bullish",
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
            st.download_button("📥 Download Scanner Results CSV", data=csv, file_name=f"NSE_Scan_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", key="btn_dl_scan")

            st.markdown("---")
            c_add1, c_add2 = st.columns([3, 1])
            selected_sym = c_add1.selectbox("Select stock to track:", [c['Symbol'] for c in cands], key="sb_equity_cand")
            if c_add2.button("Track This Stock", key="btn_track_equity_cand"):
                matched = [c for c in cands if c['Symbol'] == selected_sym][0]
                new_entry = pd.DataFrame([{
                    "Symbol": matched['Symbol'], "Entry_Price": matched['Price (₹)'],
                    "Target_Price": matched['Target (+5%) (₹)'], "Stop_Loss": matched['Stop Loss (-2%) (₹)'],
                    "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }])
                st.session_state.eq_portfolio = pd.concat([st.session_state.eq_portfolio, new_entry], ignore_index=True)
                save_csv_data(st.session_state.eq_portfolio, EQUITY_PORTFOLIO_FILE)
                st.success(f"✅ Added {matched['Symbol']} to Section 2 (Active Equity Portfolio)!")

    with t_single:
        st.subheader("Single Stock Multi-Factor Analysis")
        sym_q = st.text_input("Enter NSE Stock Symbol:", value="RELIANCE", key="ti_single_stock").upper()
        if st.button("Analyze Stock", key="btn_single_stock_eval"):
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

    with t_backtest:
        st.subheader("1-Year Strategy Performance")
        bt_target = st.text_input("NSE Symbol for Backtest:", value="TCS", key="ti_backtest_sym").upper()
        if st.button("Run 1-Year Backtest", key="btn_exec_backtest"):
            df_bt = yf.download(f"{bt_target}.NS", period="1y", interval="1d", progress=False)
            if isinstance(df_bt.columns, pd.MultiIndex):
                df_bt.columns = df_bt.columns.get_level_values(0)
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
        c_cap = st.number_input("Total Trading Capital (₹):", value=100000.0, step=10000.0, key="sz_total_cap")
        c_risk = st.number_input("Max Risk (%):", value=2.0, step=0.5, key="sz_risk_percent")
        c_entry = st.number_input("Entry Price (₹):", value=2500.0, step=10.0, key="sz_entry_val")
        c_sl = st.number_input("Stop Loss Price (₹):", value=2450.0, step=10.0, key="sz_sl_val")
        c_tgt = st.number_input("Target Price (₹):", value=2625.0, step=10.0, key="sz_tgt_val")

        if st.button("Calculate Position", key="btn_calc_position"):
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
    st.caption("Tracks open Indian equity trades against targets (+5%), stop losses (-2%), and MACD trend reversals.")

    port_df = st.session_state.eq_portfolio

    with st.expander("➕ Manually Add an Equity Position"):
        mc1, mc2, mc3, mc4 = st.columns(4)
        m_sym = mc1.text_input("NSE Symbol (e.g. RELIANCE, TCS):", key="ti_manual_eq_sym").upper()
        m_entry = mc2.number_input("Entry Price (₹):", value=2500.0, key="ti_manual_eq_entry")
        m_tgt = mc3.number_input("Target Price (₹):", value=2625.0, key="ti_manual_eq_tgt")
        m_sl = mc4.number_input("Stop Loss (₹):", value=2450.0, key="ti_manual_eq_sl")
        
        if st.button("Save Position to Monitor", key="btn_save_manual_eq"):
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
        del_sym = st.selectbox("Select Trade to Remove:", port_df['Symbol'].tolist(), key="sb_delete_equity_trade")
        if st.button("Remove Trade", key="btn_remove_equity_trade"):
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

    opt_choice = st.radio("Choose Trading Segment:", ["Benchmark Indices (Nifty / Bank Nifty)", "Top F&O Momentum Stocks"], horizontal=True, key="rb_options_segment")

    if opt_choice == "Benchmark Indices (Nifty / Bank Nifty)":
        st.subheader("Benchmark Index Directional Engine")
        idx_name = st.selectbox("Select Index:", list(INDEX_DICT.keys()), key="sb_index_choice")
        idx_ticker = INDEX_DICT[idx_name]

        if st.button("Evaluate High-Probability Setup", key="btn_eval_index_options"):
            with st.spinner(f"Analyzing multi-factor confluence for {idx_name}..."):
                df_idx = yf.download(idx_ticker, period="60d", interval="1d", progress=False)
                if isinstance(df_idx.columns, pd.MultiIndex):
                    df_idx.columns = df_idx.columns.get_level_values(0)

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
            prem_in = st.number_input("Bought Option Premium (₹):", value=120.0, step=5.0, key="ni_index_prem_in")
            if st.button("Add to Options Exit Engine", key="btn_add_idx_to_exit"):
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
        if st.button("Scan F&O Universe for Confluence Setups", key="btn_scan_fno_momentum"):
            with st.spinner("Filtering F&O universe..."):
                fno_tickers = [f"{s}.NS" for s in FNO_WATCHLIST]
                data_fno = yf.download(fno_tickers, period="60d", group_by='ticker', progress=False)

                fno_cands = []
                for s in FNO_WATCHLIST:
                    t_key = f"{s}.NS"
                    try:
                        if t_key not in data_fno or data_fno[t_key].empty:
                            continue
                        df = data_fno[t_key].dropna(subset=['Close', 'Volume'])
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

        if "active_
