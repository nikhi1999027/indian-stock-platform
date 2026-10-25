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
st.set_page_config(page_title="Institutional Stock & Options Platform", page_icon="⚡", layout="wide")

OPTIONS_FILE = "options_active_trades.csv"
EQUITY_PORTFOLIO_FILE = "active_trades.csv"

# ==============================================================================
# 1. PURE-PANDAS TECHNICAL INDICATOR ENGINE
# ==============================================================================
def calculate_sma(series, window=20):
    return series.rolling(window=window).mean()

def calculate_ema(series, span=20):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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
# 2. MARKET DATA & INSTITUTIONAL FLOWS
# ==============================================================================
def fetch_nifty_500_tickers():
    url = "https://raw.githubusercontent.com/kprohith/nse-stock-analysis/master/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        return [str(s).strip().upper() for s in df['Symbol'].dropna().unique().tolist()][:100]
    except Exception:
        return [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BHARTIARTL", 
            "SBIN", "TATAMOTORS", "LT", "ITC", "AXISBANK", "MARUTI", "BAJFINANCE",
            "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "POWERGRID", "NTPC", "M&M"
        ]

def fetch_fii_dii_activity():
    url = "https://www.moneycontrol.com/stocks/fii_dii_activity.php"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        tables = pd.read_html(response.text)
        for tbl in tables:
            if 'FII' in str(tbl) or 'DII' in str(tbl):
                return tbl.head(4)
    except Exception:
        pass
    return pd.DataFrame({
        "Category": ["FII / FPI", "DII (Domestic)"],
        "Net Flow Bias": ["Bullish / Inflow", "Strong Buyers"],
        "Market Implication": ["Index Call Support", "Cash Delivery Support"]
    })

def fetch_news_sentiment(query):
    encoded_query = urllib.parse.quote(f"{query} share stock news India")
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    polarity_scores = [TextBlob(entry.title).sentiment.polarity for entry in feed.entries[:4]]
    avg_polarity = sum(polarity_scores) / len(polarity_scores) if polarity_scores else 0.0
    label = "POSITIVE" if avg_polarity > 0.05 else ("NEGATIVE" if avg_polarity < -0.05 else "NEUTRAL")
    headlines = [entry.title for entry in feed.entries[:4]]
    return {"label": label, "score": round(avg_polarity, 2), "headlines": headlines}

# ==============================================================================
# 3. PORTFOLIO PERSISTENCE
# ==============================================================================
def load_options_portfolio():
    if os.path.exists(OPTIONS_FILE):
        try:
            return pd.read_csv(OPTIONS_FILE)
        except Exception:
            return pd.DataFrame(columns=["Contract", "Underlying", "Option_Type", "Entry_Premium", "Target_Premium", "Stop_Loss_Premium", "Underlying_SL", "Entry_Date"])
    return pd.DataFrame(columns=["Contract", "Underlying", "Option_Type", "Entry_Premium", "Target_Premium", "Stop_Loss_Premium", "Underlying_SL", "Entry_Date"])

def save_options_portfolio(df):
    df.to_csv(OPTIONS_FILE, index=False)

def load_equity_portfolio():
    if os.path.exists(EQUITY_PORTFOLIO_FILE):
        try:
            return pd.read_csv(EQUITY_PORTFOLIO_FILE)
        except Exception:
            return pd.DataFrame(columns=["Symbol", "Entry_Price", "Target_Price", "Stop_Loss", "Entry_Date"])
    return pd.DataFrame(columns=["Symbol", "Entry_Price", "Target_Price", "Stop_Loss", "Entry_Date"])

def save_equity_portfolio(df):
    df.to_csv(EQUITY_PORTFOLIO_FILE, index=False)

if "opt_portfolio" not in st.session_state:
    st.session_state.opt_portfolio = load_options_portfolio()

if "eq_portfolio" not in st.session_state:
    st.session_state.eq_portfolio = load_equity_portfolio()

INDEX_DICT = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK"}
FNO_WATCHLIST = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "TATAMOTORS",
    "SBIN", "BHARTIARTL", "AXISBANK", "LT", "KOTAKBANK", "MARUTI", "BAJFINANCE"
]

def round_strike(price, step=50):
    return int(round(price / step) * step)

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("⚡ Institutional Suite")
app_mode = st.sidebar.radio(
    "Select Platform Section:",
    [
        "1. Full-Market Equity Scanner & Investment",
        "2. Active Equity Portfolio & Exit Alerts",
        "3. High-Probability Options Trading Desk",
        "4. Active Options Exit Alert Engine",
        "5. Institutional FII/DII & Market Sentiment"
    ],
    key="app_mode_radio"
)

# ##############################################################################
# SECTION 1: FULL-MARKET EQUITY SCANNER & INVESTMENT
# ##############################################################################
if app_mode == "1. Full-Market Equity Scanner & Investment":
    st.title("📈 Full-Market Equity Scanner & Swing Investment")
    st.caption("Scans the liquid NSE universe automatically using multi-indicator confirmation (Volume, EMA, MACD, RSI).")

    tab_scan, tab_single, tab_backtest, tab_sizer = st.tabs([
        "🚀 Full-Market Scanner",
        "🔍 Single Stock Deep Dive",
        "📊 Strategy Backtester",
        "🧮 Position Size Calculator"
    ])

    with tab_scan:
        st.subheader("Automated Market Scanner")
        st.write("Click below to screen top liquid stocks across the NSE market for high-probability bullish swing setups.")

        if st.button("Run Full-Market Scanner", key="run_mkt_scan_btn"):
            with st.spinner("Downloading and screening market data for liquid NSE stocks..."):
                symbols = fetch_nifty_500_tickers()
                yf_symbols = [f"{s}.NS" for s in symbols]
                data = yf.download(yf_symbols, period="60d", group_by='ticker', progress=False)

                candidates = []
                for s in symbols:
                    t_ns = f"{s}.NS"
                    if t_ns in data and not data[t_ns].empty:
                        d = data[t_ns].dropna(subset=['Close', 'Volume'])
                        if len(d) >= 30:
                            d['EMA_20'] = calculate_ema(d['Close'], span=20)
                            d['EMA_50'] = calculate_ema(d['Close'], span=50)
                            d['SMA_Vol'] = calculate_sma(d['Volume'], window=20)
                            d['RSI'] = calculate_rsi(d['Close'], window=14)
                            d['MACD'], d['Signal'], d['Hist'] = calculate_macd(d['Close'])

                            lat = d.iloc[-1]
                            price = float(lat['Close'])
                            open_p = float(lat['Open'])
                            rsi = float(lat['RSI'])
                            ema20 = float(lat['EMA_20'])
                            ema50 = float(lat['EMA_50'])
                            macd = float(lat['MACD'])
                            sig = float(lat['Signal'])
                            vol = float(lat['Volume'])
                            vol_sma = float(lat['SMA_Vol']) if float(lat['SMA_Vol']) > 0 else 1.0
                            vol_ratio = round(vol / vol_sma, 2)

                            if (price > open_p) and (vol_ratio >= 1.2) and (price > ema20 > ema50) and (macd > sig) and (45 <= rsi <= 65):
                                target = round(price * 1.05, 2)
                                stop_loss = round(price * 0.98, 2)
                                candidates.append({
                                    "Symbol": s,
                                    "Price (₹)": round(price, 2),
                                    "Green Vol Spike": f"{vol_ratio}x Avg",
                                    "RSI": round(rsi, 1),
                                    "MACD": "Bullish Crossover",
                                    "Target (+5%) (₹)": target,
                                    "Stop Loss (-2%) (₹)": stop_loss
                                })

                st.session_state.last_equity_scan = candidates

        if "last_equity_scan" in st.session_state and st.session_state.last_equity_scan:
            candidates = st.session_state.last_equity_scan
            scan_df = pd.DataFrame(candidates)
            st.dataframe(scan_df, use_container_width=True)

            # Download CSV
            csv = scan_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scanner Results CSV",
                data=csv,
                file_name=f"Market_Scan_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

            # Automated Selection to Exit Engine
            st.markdown("---")
            st.markdown("#### **⚡ Add Candidate Directly to Active Equity Exit Engine**")
            c_add1, c_add2 = st.columns([3, 1])
            selected_stock = c_add1.selectbox("Choose stock to track for exit signals:", [c['Symbol'] for c in candidates], key="sel_cand_stock")
            if c_add2.button("Add to Exit Engine", key="track_cand_btn"):
                matched = [c for c in candidates if c['Symbol'] == selected_stock][0]
                new_trade = pd.DataFrame([{
                    "Symbol": matched['Symbol'],
                    "Entry_Price": matched['Price (₹)'],
                    "Target_Price": matched['Target (+5%) (₹)'],
                    "Stop_Loss": matched['Stop Loss (-2%) (₹)'],
                    "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }])
                st.session_state.eq_portfolio = pd.concat([st.session_state.eq_portfolio, new_trade], ignore_index=True)
                save_equity_portfolio(st.session_state.eq_portfolio)
                st.success(f"✅ Added {matched['Symbol']} to Section 2 (Active Equity Portfolio & Exit Alerts)!")

        elif "last_equity_scan" in st.session_state and not st.session_state.last_equity_scan:
            st.info("No stocks matched the strict confluence filter today. Market is consolidative; capital is preserved.")

    with tab_single:
        st.subheader("Single Stock Multi-Factor Analysis")
        sym_in = st.text_input("Enter NSE Stock Symbol:", value="RELIANCE", key="sym_single_in").upper()
        if st.button("Analyze Stock", key="btn_single_in"):
            t = yf.Ticker(f"{sym_in}.NS")
            df = t.history(period="60d")
            if not df.empty:
                df['EMA_20'] = calculate_ema(df['Close'], span=20)
                df['RSI'] = calculate_rsi(df['Close'], window=14)
                lat = df.iloc[-1]
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"₹{round(float(lat['Close']), 2)}")
                c2.metric("RSI (14)", round(float(lat['RSI']), 1))
                c3.metric("20-Day EMA", f"₹{round(float(lat['EMA_20']), 2)}")
                st.line_chart(df[['Close', 'EMA_20']].dropna())

    with tab_backtest:
        st.subheader("1-Year Strategy Historical Performance")
        bt_sym = st.text_input("NSE Symbol for Backtest:", value="TCS", key="bt_sym_key").upper()
        if st.button("Run Backtest", key="bt_btn_key"):
            df = yf.download(f"{bt_sym}.NS", period="1y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df['EMA_20'] = calculate_ema(df['Close'], span=20)
            df['RSI'] = calculate_rsi(df['Close'], window=14)
            df.dropna(inplace=True)

            cash, pos, entry_p, trades = 100000.0, 0, 0.0, []
            for date, row in df.iterrows():
                p, rsi, ema = row['Close'], row['RSI'], row['EMA_20']
                if pos > 0:
                    if p >= entry_p * 1.05 or p <= entry_p * 0.98 or rsi > 70:
                        trades.append((pos * p) - (pos * entry_p))
                        cash += pos * p
                        pos = 0
                elif pos == 0 and rsi < 45 and p > ema:
                    pos = cash / p
                    entry_p = p
                    cash = 0.0

            final_v = cash + (pos * df['Close'].iloc[-1])
            ret_p = round(((final_v - 100000.0) / 100000.0) * 100, 2)
            win_r = round((len([t for t in trades if t > 0]) / len(trades) * 100), 1) if trades else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Final Value", f"₹{round(final_v, 2)}")
            c2.metric("Total Return", f"{ret_p}%")
            c3.metric("Win Rate", f"{win_r}%")

    with tab_sizer:
        st.subheader("Position Size Calculator")
        cap = st.number_input("Total Capital (₹):", value=100000.0, key="sz_cap")
        risk_pct = st.number_input("Max Risk (%):", value=2.0, key="sz_risk")
        entry_p = st.number_input("Entry Price (₹):", value=2500.0, key="sz_entry")
        sl_p = st.number_input("Stop Loss Price (₹):", value=2450.0, key="sz_sl")
        tgt_p = st.number_input("Target Price (₹):", value=2625.0, key="sz_tgt")

        if st.button("Calculate Position", key="sz_btn"):
            risk_per_sh = entry_p - sl_p
            if risk_per_sh <= 0:
                st.error("Entry price must be higher than Stop Loss.")
            else:
                max_risk_val = cap * (risk_pct / 100.0)
                shares = int(min(max_risk_val // risk_per_sh, cap // entry_p))
                st.success(f"**Recommended Buy Quantity:** {shares} Shares")
                st.write(f"**Capital Required:** ₹{round(shares * entry_p, 2)}")
                st.write(f"**Max Loss:** ₹{round(shares * risk_per_sh, 2)}")
                st.write(f"**Max Gain:** ₹{round(shares * (tgt_p - entry_p), 2)}")

# ##############################################################################
# SECTION 2: ACTIVE EQUITY PORTFOLIO & EXIT ALERTS
# ##############################################################################
elif app_mode == "2. Active Equity Portfolio & Exit Alerts":
    st.title("🛡️ Active Equity Portfolio & Dynamic Exit Monitor")
    st.caption("Tracks your open equity swing trades and evaluates real-time prices against Targets, Stops, and Trend Reversals.")

    port_df = st.session_state.eq_portfolio

    with st.expander("➕ Manually Add an Equity Position"):
        c1, c2, c3, c4 = st.columns(4)
        m_sym = c1.text_input("NSE Symbol (e.g. RELIANCE, TCS):", key="eq_m_sym").upper()
        m_entry = c2.number_input("Entry Price (₹):", value=2500.0, key="eq_m_entry")
        m_target = c3.number_input("Target Price (₹):", value=2625.0, key="eq_m_tgt")
        m_sl = c4.number_input("Stop Loss (₹):", value=2450.0, key="eq_m_sl")
        if st.button("Save Equity Position", key="eq_m_save_btn"):
            if m_sym:
                new_row = pd.DataFrame([{
                    "Symbol": m_sym, "Entry_Price": m_entry,
                    "Target_Price": m_target, "Stop_Loss": m_sl,
                    "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }])
                st.session_state.eq_portfolio = pd.concat([st.session_state.eq_portfolio, new_row], ignore_index=True)
                save_equity_portfolio(st.session_state.eq_portfolio)
                st.success(f"Tracking {m_sym}!")
                st.rerun()

    if port_df.empty:
        st.info("No active equity trades in portfolio. Run the scanner in Section 1 to find setups.")
    else:
        st.subheader("Live Exit Matrix")
        status_reports = []
        for index, row in port_df.iterrows():
            sym = row['Symbol']
            entry = float(row['Entry_Price'])
            target = float(row['Target_Price'])
            sl = float(row['Stop_Loss'])

            try:
                t = yf.Ticker(f"{sym}.NS")
                h = t.history(period="15d")
                if not h.empty:
                    cur_p = float(h['Close'].iloc[-1])
                    h['MACD'], h['Signal'], _ = calculate_macd(h['Close'])
                    cur_macd = float(h['MACD'].iloc[-1])
                    cur_sig = float(h['Signal'].iloc[-1])
                    pnl = round(((cur_p - entry) / entry) * 100, 2)

                    if cur_p >= target:
                        action = "🎯 TAKE PROFIT (Target Hit)"
                    elif cur_p <= sl:
                        action = "🛑 STOP LOSS (Exit Immediately)"
                    elif cur_macd < cur_sig:
                        action = "⚠️ SELL (Trend Reversal)"
                    else:
                        action = "✅ HOLD (Trend Intact)"

                    status_reports.append({
                        "Symbol": sym,
                        "Entry (₹)": entry,
                        "Live Price (₹)": round(cur_p, 2),
                        "P&L (%)": f"{pnl}%",
                        "Target (₹)": target,
                        "Stop Loss (₹)": sl,
                        "Exit Signal": action
                    })
            except Exception:
                continue

        if status_reports:
            st.dataframe(pd.DataFrame(status_reports), use_container_width=True)

        st.markdown("---")
        del_sym = st.selectbox("Select Trade to Remove After Exit:", port_df['Symbol'].tolist(), key="del_eq_sel")
        if st.button("Remove Trade", key="del_eq_btn"):
            st.session_state.eq_portfolio = port_df[port_df['Symbol'] != del_sym].reset_index(drop=True)
            save_equity_portfolio(st.session_state.eq_portfolio)
            st.success(f"Removed {del_sym} from tracking.")
            st.rerun()

# ##############################################################################
# SECTION 3: HIGH-PROBABILITY OPTIONS TRADING DESK
# ##############################################################################
elif app_mode == "3. High-Probability Options Trading Desk":
    st.title("🎯 High-Probability Options Trading Desk")
    st.caption("Confluence Strategy: EMA 20/50 + Bollinger Squeeze/Breakout + MACD Histogram + Green Volume")

    opt_choice = st.radio("Choose Trading Segment:", ["Benchmark Indices (Nifty / Bank Nifty)", "Top F&O Momentum Stocks"], horizontal=True, key="opt_choice_radio")

    if opt_choice == "Benchmark Indices (Nifty / Bank Nifty)":
        st.subheader("Benchmark Index Directional Engine")
        idx_name = st.selectbox("Select Index:", list(INDEX_DICT.keys()), key="idx_select")
        idx_ticker = INDEX_DICT[idx_name]

        if st.button("Evaluate High-Probability Setup", key="eval_idx_btn"):
            with st.spinner(f"Analyzing multi-factor confluence for {idx_name}..."):
                df = yf.download(idx_ticker, period="60d", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                if df.empty or len(df) < 30:
                    st.error("Insufficient market data for index.")
                else:
                    df['EMA_20'] = calculate_ema(df['Close'], span=20)
                    df['EMA_50'] = calculate_ema(df['Close'], span=50)
                    df['RSI'] = calculate_rsi(df['Close'], window=14)
                    df['MACD'], df['Signal'], df['Hist'] = calculate_macd(df['Close'])
                    df['BB_Up'], df['BB_Mid'], df['BB_Low'] = calculate_bollinger_bands(df['Close'])

                    lat = df.iloc[-1]
                    price = float(lat['Close'])
                    ema20, ema50 = float(lat['EMA_20']), float(lat['EMA_50'])
                    rsi = float(lat['RSI'])
                    macd, sig, hist = float(lat['MACD']), float(lat['Signal']), float(lat['Hist'])

                    sentiment = fetch_news_sentiment(idx_name)

                    step = 50 if "NIFTY 50" in idx_name else 100
                    atm_strike = round_strike(price, step=step)

                    call_condition = (price > ema20) and (ema20 > ema50) and (rsi >= 55) and (hist > 0) and (sentiment['score'] >= 0)
                    put_condition = (price < ema20) and (ema20 < ema50) and (rsi <= 45) and (hist < 0) and (sentiment['score'] <= 0)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Spot Level", f"{round(price, 1)}")
                    col2.metric("RSI (14)", round(rsi, 1))
                    col3.metric("MACD Hist", round(hist, 2))
                    col4.metric("Market Sentiment", sentiment['label'])

                    st.markdown("---")
                    if call_condition:
                        contract_name = f"{idx_name} {atm_strike} CE"
                        st.success(f"### 🟢 HIGH PROBABILITY CALL SETUP: BUY {contract_name}")
                        st.write(f"**Strike:** ATM {atm_strike} Call (CE)")
                        st.write(f"**Underlying Stop-Loss:** Exit if spot drops below EMA 20 (₹{round(ema20, 1)})")
                        
                        st.session_state.current_idx_signal = {
                            "Contract": contract_name,
                            "Underlying": "NIFTY" if "NIFTY 50" in idx_name else "BANKNIFTY",
                            "Option_Type": "CE",
                            "Underlying_SL": round(ema20, 1)
                        }
                    elif put_condition:
                        contract_name = f"{idx_name} {atm_strike} PE"
                        st.error(f"### 🔴 HIGH PROBABILITY PUT SETUP: BUY {contract_name}")
                        st.write(f"**Strike:** ATM {atm_strike} Put (PE)")
                        st.write(f"**Underlying Stop-Loss:** Exit if spot climbs above EMA 20 (₹{round(ema20, 1)})")
                        
                        st.session_state.current_idx_signal = {
                            "Contract": contract_name,
                            "Underlying": "NIFTY" if "NIFTY 50" in idx_name else "BANKNIFTY",
                            "Option_Type": "PE",
                            "Underlying_SL": round(ema20, 1)
                        }
                    else:
                        st.session_state.current_idx_signal = None
                        st.warning("### ⚠️ NO TRADE SETUP — STAND ASIDE")
                        st.info("Institutional rule: Indicators are divergent or consolidative. Preserving capital.")

        if "current_idx_signal" in st.session_state and st.session_state.current_idx_signal:
            sig = st.session_state.current_idx_signal
            st.markdown("#### **⚡ Add Index Setup Directly to Options Exit Alert Engine**")
            idx_prem = st.number_input("Estimated / Bought Option Premium (₹):", value=120.0, step=5.0, key="idx_add_prem")
            if st.button("Add to Options Exit Engine", key="add_idx_to_exit_btn"):
                new_trade = pd.DataFrame([{
                    "Contract": sig["Contract"],
                    "Underlying": sig["Underlying"],
                    "Option_Type": sig["Option_Type"],
                    "Entry_Premium": idx_prem,
                    "Target_Premium": round(idx_prem * 1.30, 2),
                    "Stop_Loss_Premium": round(idx_prem * 0.80, 2),
                    "Underlying_SL": sig["Underlying_SL"],
                    "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }])
                st.session_state.opt_portfolio = pd.concat([st.session_state.opt_portfolio, new_trade], ignore_index=True)
                save_options_portfolio(st.session_state.opt_portfolio)
                st.success(f"✅ Added {sig['Contract']} to Section 4 (Active Options Exit Alert Engine)!")

    else:
        st.subheader("F&O Stock High-Win-Rate Momentum Scanner")
        if st.button("Scan F&O Universe for Confluence Setups", key="fno_scan_btn"):
            with st.spinner("Screening F&O universe with strict 5-factor confluence..."):
                fno_symbols = [f"{s}.NS" for s in FNO_WATCHLIST]
                data = yf.download(fno_symbols, period="60d", group_by='ticker', progress=False)

                fno_results = []
                for s in FNO_WATCHLIST:
                    t_ns = f"{s}.NS"
                    if t_ns in data and not data[t_ns].empty:
                        d = data[t_ns].dropna(subset=['Close', 'Volume'])
                        if len(d) >= 30:
                            d['EMA_20'] = calculate_ema(d['Close'], span=20)
                            d['EMA_50'] = calculate_ema(d['Close'], span=50)
                            d['SMA_Vol'] = calculate_sma(d['Volume'], window=20)
                            d['RSI'] = calculate_rsi(d['Close'], window=14)
                            d['MACD'], d['Signal'], d['Hist'] = calculate_macd(d['Close'])

                            lat = d.iloc[-1]
                            price, open_p = float(lat['Close']), float(lat['Open'])
                            rsi = float(lat['RSI'])
                            ema20, ema50 = float(lat['EMA_20']), float(lat['EMA_50'])
                            hist = float(lat['Hist'])
                            vol = float(lat['Volume'])
                            vol_sma = float(lat['SMA_Vol']) if float(lat['SMA_Vol']) > 0 else 1.0
                            vol_ratio = round(vol / vol_sma, 2)

                            step = 100 if price > 2000 else (50 if price > 500 else 10)
                            strike = round_strike(price, step=step)

                            if (price > open_p) and (price > ema20 > ema50) and (vol_ratio >= 1.3) and (55 <= rsi <= 70) and (hist > 0):
                                fno_results.append({
                                    "Stock": s,
                                    "Spot Price (₹)": round(price, 2),
                                    "Recommended Contract": f"{s} {strike} CE",
                                    "Option_Type": "CE",
                                    "Volume Surge": f"{vol_ratio}x Avg",
                                    "RSI": round(rsi, 1),
                                    "Setup": "Institutional Long Breakout",
                                    "Underlying SL (₹)": round(ema20, 2)
                                })
                            elif (price < open_p) and (price < ema20 < ema50) and (vol_ratio >= 1.3) and (30 <= rsi <= 45) and (hist < 0):
                                fno_results.append({
                                    "Stock": s,
                                    "Spot Price (₹)": round(price, 2),
                                    "Recommended Contract": f"{s} {strike} PE",
                                    "Option_Type": "PE",
                                    "Volume Surge": f"{vol_ratio}x Avg",
                                    "RSI": round(rsi, 1),
                                    "Setup": "Institutional Short Breakdown",
                                    "Underlying SL (₹)": round(ema20, 2)
                                })

                st.session_state.last_fno_scan = fno_results

        if "last_fno_scan" in st.session_state and st.session_state.last_fno_scan:
            fno_results = st.session_state.last_fno_scan
            st.dataframe(pd.DataFrame(fno_results), use_container_width=True)

            st.markdown("---")
            st.markdown("#### **⚡ Add F&O Stock Setup Directly to Options Exit Alert Engine**")
            col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
            sel_contract = col_f1.selectbox("Select Option Contract to Track:", [r["Recommended Contract"] for r in fno_results], key="sel_fno_c")
            fno_prem = col_f2.number_input("Option Premium (₹):", value=50.0, step=2.0, key="fno_add_prem")

            if col_f3.button("Add to Options Engine", key="add_fno_btn"):
                matched_fno = [r for r in fno_results if r["Recommended Contract"] == sel_contract][0]
                new_trade = pd.DataFrame([{
                    "Contract": matched_fno["Recommended Contract"],
                    "Underlying": matched_fno["Stock"],
                    "Option_Type": matched_fno["Option_Type"],
                    "Entry_Premium": fno_prem,
                    "Target_Premium": round(fno_prem * 1.30, 2),
                    "Stop_Loss_Premium": round(fno_prem * 0.80, 2),
                    "Underlying_SL": matched_fno["Underlying SL (₹)"],
                    "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }])
                st.session_state.opt_portfolio = pd.concat([st.session_state.opt_portfolio, new_trade], ignore_index=True)
                save_options_portfolio(st.session_state.opt_portfolio)
                st.success(f"✅ Added {matched_fno['Recommended Contract']} to Section 4 (Active Options Exit Alert Engine)!")

        elif "last_fno_scan" in st.session_state and not st.session_state.last_fno_scan:
            st.info("No stocks met the strict institutional criteria right now. Market is in consolidation.")

# ##############################################################################
# SECTION 4: ACTIVE OPTIONS EXIT ALERT ENGINE
# ##############################################################################
elif app_mode == "4. Active Options Exit Alert Engine":
    st.title("🛡️ Real-Time Options Exit Alert Engine")
    st.caption("Monitors active option positions, tracking target profits (+30%/+60%), stop-loss (-20%), and underlying EMA reversals.")

    port_df = st.session_state.opt_portfolio

    with st.expander("➕ Manually Log an Active Option Trade"):
        c1, c2, c3 = st.columns(3)
        c_underlying = c1.selectbox("Underlying Asset:", ["NIFTY", "BANKNIFTY"] + FNO_WATCHLIST, key="log_und")
        c_type = c2.selectbox("Option Type:", ["CE", "PE"], key="log_type")
        c_strike = c3.number_input("Strike Price:", value=25000, step=50, key="log_strike")

        c4, c5 = st.columns(2)
        c_entry_prem = c4.number_input("Your Entry Premium (₹):", value=120.0, step=5.0, key="log_prem")
        c_underlying_sl = c5.number_input("Underlying Stop-Loss Level (₹):", value=24800.0, step=10.0, key="log_und_sl")

        if st.button("Start Tracking Exit Signals", key="log_save_btn"):
            contract_code = f"{c_underlying} {c_strike} {c_type}"
            target_prem = round(c_entry_prem * 1.30, 2)
            sl_prem = round(c_entry_prem * 0.80, 2)

            new_trade = pd.DataFrame([{
                "Contract": contract_code,
                "Underlying": c_underlying,
                "Option_Type": c_type,
                "Entry_Premium": c_entry_prem,
                "Target_Premium": target_prem,
                "Stop_Loss_Premium": sl_prem,
                "Underlying_SL": c_underlying_sl,
                "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
            }])
            st.session_state.opt_portfolio = pd.concat([st.session_state.opt_portfolio, new_trade], ignore_index=True)
            save_options_portfolio(st.session_state.opt_portfolio)
            st.success(f"Tracking {contract_code}!")
            st.rerun()

    if port_df.empty:
        st.info("No active option trades currently logged.")
    else:
        st.subheader("Live Exit Status Matrix")
        live_reports = []

        for idx, row in port_df.iterrows():
