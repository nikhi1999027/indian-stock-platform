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
                    if p >= (
