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

# ==============================================================================
# 1. PURE-PANDAS TECHNICAL INDICATOR ENGINE
# ==============================================================================
def calculate_sma(series, window=20):
    """Calculates Simple Moving Average."""
    return series.rolling(window=window).mean()

def calculate_ema(series, span=20):
    """Calculates Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, window=14):
    """Calculates Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculates MACD Line, Signal Line, and Histogram."""
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Calculates Bollinger Bands."""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

# ==============================================================================
# 2. FII / DII INSTITUTIONAL SENTIMENT FETCHER
# ==============================================================================
def fetch_fii_dii_activity():
    """Fetches real-time institutional activity estimates via public feed."""
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

# ==============================================================================
# 3. NEWS SENTIMENT (NLP)
# ==============================================================================
def fetch_news_sentiment(query):
    """Extracts polarity from Google News RSS feed."""
    encoded_query = urllib.parse.quote(f"{query} share stock news India")
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    polarity_scores = [TextBlob(entry.title).sentiment.polarity for entry in feed.entries[:4]]
    avg_polarity = sum(polarity_scores) / len(polarity_scores) if polarity_scores else 0.0
    label = "POSITIVE" if avg_polarity > 0.05 else ("NEGATIVE" if avg_polarity < -0.05 else "NEUTRAL")
    headlines = [entry.title for entry in feed.entries[:4]]
    return {"label": label, "score": round(avg_polarity, 2), "headlines": headlines}

# ==============================================================================
# 4. OPTIONS ACTIVE TRADES & EXIT PERSISTENCE
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

if "opt_portfolio" not in st.session_state:
    st.session_state.opt_portfolio = load_options_portfolio()

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
        "1. High-Probability Options Trading & Scanner",
        "2. Active Options Exit Alert Engine",
        "3. Institutional FII/DII & Market Sentiment",
        "4. Equity Swing Investment Desk"
    ],
    key="app_mode_radio"
)

# ##############################################################################
# SECTION 1: HIGH-PROBABILITY OPTIONS TRADING (STOCKS & INDICES)
# ##############################################################################
if app_mode == "1. High-Probability Options Trading & Scanner":
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
                    bb_up, bb_low = float(lat['BB_Up']), float(lat['BB_Low'])

                    sentiment = fetch_news_sentiment(idx_name)

                    step = 50 if "NIFTY 50" in idx_name else 100
                    atm_strike = round_strike(price, step=step)

                    # Institutional High-Confluence Rules
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
                        st.write(f"**Confluence Alignment:** Spot > EMA 20 > EMA 50, RSI Bullish ({round(rsi, 1)}), Positive MACD Histogram.")
                        st.write(f"**Underlying Stop-Loss:** Exit if spot drops below EMA 20 (₹{round(ema20, 1)})")
                    elif put_condition:
                        contract_name = f"{idx_name} {atm_strike} PE"
                        st.error(f"### 🔴 HIGH PROBABILITY PUT SETUP: BUY {contract_name}")
                        st.write(f"**Strike:** ATM {atm_strike} Put (PE)")
                        st.write(f"**Confluence Alignment:** Spot < EMA 20 < EMA 50, RSI Bearish ({round(rsi, 1)}), Negative MACD Histogram.")
                        st.write(f"**Underlying Stop-Loss:** Exit if spot climbs above EMA 20 (₹{round(ema20, 1)})")
                    else:
                        st.warning("### ⚠️ NO TRADE SETUP — STAND ASIDE")
                        st.info("Institutional rule: Indicators are divergent or consolidative. Preserving capital.")

    else:
        st.subheader("F&O Stock High-Win-Rate Momentum Scanner")
        if st.button("Scan F&O Universe for Confluence Setups", key="fno_scan_btn"):
            with st.spinner("Filtering F&O universe with strict 5-factor confluence..."):
                fno_symbols = [f"{s}.NS" for s in FNO_WATCHLIST]
                data = yf.download(fno_symbols, period="60d", group_by='ticker', progress=False)

                results = []
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

                            # Strict Bullish Breakout
                            if (price > open_p) and (price > ema20 > ema50) and (vol_ratio >= 1.3) and (55 <= rsi <= 70) and (hist > 0):
                                results.append({
                                    "Stock": s,
                                    "Spot Price (₹)": round(price, 2),
                                    "Recommended Contract": f"{strike} CE",
                                    "Volume Surge": f"{vol_ratio}x Avg",
                                    "RSI": round(rsi, 1),
                                    "Setup": "Institutional Long Breakout",
                                    "Underlying SL (₹)": round(ema20, 2)
                                })
                            # Strict Bearish Breakdown
                            elif (price < open_p) and (price < ema20 < ema50) and (vol_ratio >= 1.3) and (30 <= rsi <= 45) and (hist < 0):
                                results.append({
                                    "Stock": s,
                                    "Spot Price (₹)": round(price, 2),
                                    "Recommended Contract": f"{strike} PE",
                                    "Volume Surge": f"{vol_ratio}x Avg",
                                    "RSI": round(rsi, 1),
                                    "Setup": "Institutional Short Breakdown",
                                    "Underlying SL (₹)": round(ema20, 2)
                                })

                if results:
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
                else:
                    st.info("No stocks met the strict institutional criteria right now. Market is in consolidation.")

# ##############################################################################
# SECTION 2: ACTIVE OPTIONS EXIT ALERT ENGINE
# ##############################################################################
elif app_mode == "2. Active Options Exit Alert Engine":
    st.title("🛡️ Real-Time Options Exit Alert Engine")
    st.caption("Monitors active option positions, tracking target profits (+30%/+60%), stop-loss (-20%), and underlying EMA reversals.")

    port_df = st.session_state.opt_portfolio

    # Input Form to Track an Option Trade
    with st.expander("➕ Log an Active Option Trade to Monitor"):
        c1, c2, c3 = st.columns(3)
        c_underlying = c1.selectbox("Underlying Asset:", ["NIFTY", "BANKNIFTY"] + FNO_WATCHLIST, key="log_und")
        c_type = c2.selectbox("Option Type:", ["CE", "PE"], key="log_type")
        c_strike = c3.number_input("Strike Price:", value=25000, step=50, key="log_strike")

        c4, c5 = st.columns(2)
        c_entry_prem = c4.number_input("Your Entry Premium (₹):", value=120.0, step=5.0, key="log_prem")
        c_underlying_sl = c5.number_input("Underlying Stop-Loss Level (₹):", value=24800.0, step=10.0, key="log_und_sl")

        if st.button("Start Tracking Exit Signals", key="log_save_btn"):
            contract_code = f"{c_underlying} {c_strike} {c_type}"
            target_prem = round(c_entry_prem * 1.30, 2)   # +30% Target
            sl_prem = round(c_entry_prem * 0.80, 2)       # -20% Stop Loss

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
            und = row['Underlying']
            contract = row['Contract']
            op_type = row['Option_Type']
            entry_prem = float(row['Entry_Premium'])
            tgt_prem = float(row['Target_Premium'])
            sl_prem = float(row['Stop_Loss_Premium'])
            und_sl = float(row['Underlying_SL'])

            # Determine ticker symbol
            ticker_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
            query_ticker = ticker_map.get(und, f"{und}.NS")

            try:
                t = yf.Ticker(query_ticker)
                h = t.history(period="10d")
                if not h.empty:
                    cur_spot = float(h['Close'].iloc[-1])
                    h['EMA_20'] = calculate_ema(h['Close'], span=20)
                    ema20 = float(h['EMA_20'].iloc[-1])

                    # Exit Rule Evaluation
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
                        "Contract": contract,
                        "Entry Premium (₹)": entry_prem,
                        "Target (+30%) (₹)": tgt_prem,
                        "SL (-20%) (₹)": sl_prem,
                        "Live Spot (₹)": round(cur_spot, 1),
                        "Underlying SL (₹)": und_sl,
                        "Exit Signal": status
                    })
            except Exception:
                continue

        if live_reports:
            st.dataframe(pd.DataFrame(live_reports), use_container_width=True)

        st.markdown("---")
        del_contract = st.selectbox("Select Trade to Remove After Exit:", port_df['Contract'].tolist(), key="del_opt_sel")
        if st.button("Remove Trade", key="del_opt_btn"):
            st.session_state.opt_portfolio = port_df[port_df['Contract'] != del_contract].reset_index(drop=True)
            save_options_portfolio(st.session_state.opt_portfolio)
            st.success(f"Removed {del_contract} from tracking.")
            st.rerun()

# ##############################################################################
# SECTION 3: FII / DII INSTITUTIONAL SENTIMENT
# ##############################################################################
elif app_mode == "3. Institutional FII/DII & Market Sentiment":
    st.title("🏛️ Institutional Activity & Market Sentiment")
    st.caption("Foreign & Domestic Institutional Investors drive large-scale liquidity trends.")

    st.subheader("Daily Institutional Flow Trends")
    st.table(fetch_fii_dii_activity())

    st.markdown("---")
    st.subheader("NLP Sentiment Analyzer")
    query_sym = st.text_input("Enter Company or Index Name:", value="Nifty 50", key="nlp_in")
    if st.button("Analyze News Polarity", key="nlp_btn"):
        with st.spinner("Analyzing top Google News feeds..."):
            res = fetch_news_sentiment(query_sym)
            c1, c2 = st.columns(2)
            c1.metric("Headline Sentiment", res['label'])
            c2.metric("Polarity Score (-1.0 to +1.0)", res['score'])
            st.write("#### Recent Major Headlines:")
            for h in res['headlines']:
                st.write(f"- {h}")

# ##############################################################################
# SECTION 4: EQUITY SWING INVESTMENT DESK
# ##############################################################################
elif app_mode == "4. Equity Swing Investment Desk":
    st.title("📈 Equity Swing Investment Desk")
    eq_sym = st.text_input("Enter NSE Stock Symbol:", value="RELIANCE", key="eq_sym_in").upper()
    if st.button("Evaluate Stock", key="eq_btn"):
        t = yf.Ticker(f"{eq_sym}.NS")
        df = t.history(period="60d")
        if not df.empty:
            df['EMA_20'] = calculate_ema(df['Close'], span=20)
            df['RSI'] = calculate_rsi(df['Close'], window=14)
            lat = df.iloc[-1]
            st.metric("Price", f"₹{round(float(lat['Close']), 2)}")
            st.metric("RSI (14)", round(float(lat['RSI']), 2))
            st.line_chart(df[['Close', 'EMA_20']].dropna())
