import streamlit as st
import yfinance as yf
import pandas as pd
import feedparser
import urllib.parse
from textblob import TextBlob
from datetime import datetime
import os

# Streamlit Page Configuration
st.set_page_config(page_title="Indian Stock Platform & Exit Tracker", page_icon="📈", layout="wide")

PORTFOLIO_FILE = "active_trades.csv"

# ==============================================================================
# PURE-PANDAS TECHNICAL INDICATOR CALCULATORS
# ==============================================================================
def calculate_sma(series, window=20):
    """Calculates Simple Moving Average."""
    return series.rolling(window=window).mean()

def calculate_rsi(series, window=14):
    """Calculates Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculates MACD Line, Signal Line, and Histogram using pure pandas EMAs."""
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# ==============================================================================
# PORTFOLIO PERSISTENT MEMORY FUNCTIONS
# ==============================================================================
def load_portfolio():
    """Loads saved active trades from local CSV storage."""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            return pd.read_csv(PORTFOLIO_FILE)
        except Exception:
            return pd.DataFrame(columns=["Symbol", "Entry_Price", "Target_Price", "Stop_Loss", "Entry_Date"])
    return pd.DataFrame(columns=["Symbol", "Entry_Price", "Target_Price", "Stop_Loss", "Entry_Date"])

def save_portfolio(df):
    """Saves active trades to local CSV storage."""
    df.to_csv(PORTFOLIO_FILE, index=False)

if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_portfolio()

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def fetch_news_sentiment(company_name):
    encoded_query = urllib.parse.quote(f"{company_name} share stock news India")
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    polarity_scores = [TextBlob(entry.title).sentiment.polarity for entry in feed.entries[:3]]
    avg_polarity = sum(polarity_scores) / len(polarity_scores) if polarity_scores else 0.0
    label = "POSITIVE" if avg_polarity > 0.05 else ("NEGATIVE" if avg_polarity < -0.05 else "NEUTRAL")
    headlines = [entry.title for entry in feed.entries[:3]]
    return {"label": label, "score": round(avg_polarity, 2), "headlines": headlines}

def fetch_nifty_tickers():
    url = "https://raw.githubusercontent.com/kprohith/nse-stock-analysis/master/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        return [str(s).strip().upper() for s in df['Symbol'].dropna().unique().tolist()][:100]
    except Exception:
        return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BHARTIARTL", "SBIN", "TATAMOTORS", "LT", "ITC"]

# Main Application Title
st.title("📈 Advanced Indian Stock Scanner & Dynamic Exit System")

# 5 Core Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Live Stock Analysis", 
    "2. Bullish Multi-Indicator Scanner", 
    "3. Strategy Backtest", 
    "4. Position Sizer",
    "5. Active Portfolio & Exit Signals"
])

# ==============================================================================
# TAB 1: LIVE STOCK ANALYSIS
# ==============================================================================
with tab1:
    st.subheader("Single Stock Technical & Trend Analysis")
    symbol = st.text_input("Enter NSE Symbol:", value="RELIANCE", key="tab1_symbol_input").upper()
    if st.button("Analyze Stock", key="tab1_analyze_button"):
        with st.spinner("Fetching data..."):
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                df = ticker.history(period="60d")
                if df.empty:
                    st.error("No data found for symbol.")
                else:
                    info = ticker.info if hasattr(ticker, 'info') else {}
                    company = info.get('shortName', symbol)
                    
                    # Calculations
                    df['SMA_20'] = calculate_sma(df['Close'], window=20)
                    df['RSI_14'] = calculate_rsi(df['Close'], window=14)
                    df['MACD'], df['Signal'], df['Hist'] = calculate_macd(df['Close'])
                    
                    latest = df.iloc[-1]
                    price = float(latest['Close'])
                    rsi = float(latest['RSI_14'])
                    sma = float(latest['SMA_20'])
                    macd = float(latest['MACD'])
                    sig = float(latest['Signal'])
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Current Price", f"₹{round(price, 2)}")
                    c2.metric("RSI (14)", round(rsi, 2))
                    c3.metric("20-Day SMA", f"₹{round(sma, 2)}")
                    c4.metric("MACD vs Signal", f"{round(macd, 2)} / {round(sig, 2)}")
                    
                    # Trend Evaluation
                    is_bullish = (price > sma) and (macd > sig) and (rsi > 40 and rsi < 70)
                    if is_bullish:
                        st.success("🟢 Overall Technical Trend: STRONG BULLISH SETUP")
                    else:
                        st.warning("🟡 Overall Technical Trend: NEUTRAL / CAUTION")

                    # Price & Moving Average Chart
                    chart_data = df[['Close', 'SMA_20']].dropna()
                    st.line_chart(chart_data)
            except Exception as e:
                st.error(f"Error: {e}")

# ==============================================================================
# TAB 2: BULLISH MULTI-INDICATOR SCANNER (Strict Green Momentum)
# ==============================================================================
with tab2:
    st.subheader("Bullish Volume Spike & Trend Momentum Scanner")
    st.caption("Filters strictly for upward green candles, volume expansion (>1.2x), and MACD crossover confirmation.")

    if st.button("Run Bullish Scanner", key="tab2_scan_button"):
        with st.spinner("Scanning universe for high-probability setups..."):
            sample_watchlist = fetch_nifty_tickers()
            yf_symbols = [f"{s}.NS" for s in sample_watchlist]
            data = yf.download(yf_symbols, period="60d", group_by='ticker', progress=False)
            
            candidates = []
            for s in sample_watchlist:
                t_ns = f"{s}.NS"
                if t_ns in data and not data[t_ns].empty:
                    d = data[t_ns].dropna(subset=['Close', 'Volume'])
                    if len(d) >= 30:
                        d['SMA_20'] = calculate_sma(d['Close'], window=20)
                        d['RSI_14'] = calculate_rsi(d['Close'], window=14)
                        d['Vol_SMA_20'] = calculate_sma(d['Volume'], window=20)
                        d['MACD'], d['Signal'], d['Hist'] = calculate_macd(d['Close'])
                        
                        lat = d.iloc[-1]
                        price = float(lat['Close'])
                        open_p = float(lat['Open'])
                        rsi = float(lat['RSI_14'])
                        sma = float(lat['SMA_20'])
                        macd = float(lat['MACD'])
                        sig = float(lat['Signal'])
                        vol_r = round(float(lat['Volume']) / float(lat['Vol_SMA_20']), 2) if float(lat['Vol_SMA_20']) > 0 else 1.0

                        # STRICT CRITERIA:
                        # 1. Price is GREEN (Close > Open)
                        # 2. Volume Spike (> 1.2x average volume)
                        # 3. Trend Alignment (Price > 20-Day SMA)
                        # 4. Momentum (MACD > Signal Line)
                        # 5. RSI not overbought (40 to 65)
                        if (price > open_p) and (vol_r >= 1.2) and (price > sma) and (macd > sig) and (40 <= rsi <= 65):
                            target = round(price * 1.05, 2)
                            stop_loss = round(price * 0.98, 2)
                            candidates.append({
                                "Symbol": s,
                                "Price (₹)": round(price, 2),
                                "Green Vol Spike": f"{vol_r}x Avg",
                                "RSI": round(rsi, 2),
                                "MACD Status": "Bullish Crossover",
                                "Target (+5%)": target,
                                "Stop Loss (-2%)": stop_loss
                            })

            if candidates:
                scan_df = pd.DataFrame(candidates)
                st.dataframe(scan_df, use_container_width=True)
                
                # Selection dropdown to easily add to active portfolio
                st.markdown("#### **Save Selected Stock to Active Portfolio Memory**")
                selected_sym = st.selectbox("Select stock to add to active portfolio:", [c['Symbol'] for c in candidates], key="add_sym_select")
                
                if st.button("Add to Active Portfolio Tracker", key="add_to_port_btn"):
                    matched = [c for c in candidates if c['Symbol'] == selected_sym][0]
                    new_row = pd.DataFrame([{
                        "Symbol": matched['Symbol'],
                        "Entry_Price": matched['Price (₹)'],
                        "Target_Price": matched['Target (+5%)'],
                        "Stop_Loss": matched['Stop Loss (-2%)'],
                        "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }])
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                    save_portfolio(st.session_state.portfolio)
                    st.success(f"Added {matched['Symbol']} to your active portfolio! Open Tab 5 to track exit signals.")
            else:
                st.info("No stocks met the strict multi-indicator bullish criteria today. Cash remains protected.")

# ==============================================================================
# TAB 3: BACKTESTER
# ==============================================================================
with tab3:
    st.subheader("1-Year Strategy Backtest")
    bt_sym = st.text_input("NSE Symbol for Backtest:", value="TCS", key="tab3_symbol_input").upper()
    if st.button("Run Backtest", key="tab3_backtest_button"):
        with st.spinner("Calculating performance..."):
            df = yf.download(f"{bt_sym}.NS", period="1y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df['SMA_20'] = calculate_sma(df['Close'], window=20)
            df['RSI_14'] = calculate_rsi(df['Close'], window=14)
            df.dropna(inplace=True)
            
            cash, pos, entry_p, trades = 100000.0, 0, 0.0, []
            for date, row in df.iterrows():
                p, rsi, sma = row['Close'], row['RSI_14'], row['SMA_20']
                if pos > 0:
                    if p >= entry_p * 1.05 or p <= entry_p * 0.98 or rsi > 70:
                        trades.append((pos * p) - (pos * entry_p))
                        cash += pos * p
                        pos = 0
                elif pos == 0 and rsi < 45 and p > sma:
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

# ==============================================================================
# TAB 4: POSITION SIZER
# ==============================================================================
with tab4:
    st.subheader("Position Size Calculator")
    cap = st.number_input("Total Capital (₹):", value=100000.0, key="tab4_capital")
    risk_pct = st.number_input("Max Risk (%):", value=2.0, key="tab4_risk")
    entry_p = st.number_input("Entry Price (₹):", value=2500.0, key="tab4_entry")
    sl_p = st.number_input("Stop Loss Price (₹):", value=2450.0, key="tab4_sl")
    tgt_p = st.number_input("Target Price (₹):", value=2625.0, key="tab4_target")
    
    if st.button("Calculate Position", key="tab4_calc_button"):
        risk_per_sh = entry_p - sl_p
        if risk_per_sh <= 0:
            st.error("Entry price must be higher than Stop Loss.")
        else:
            max_risk_val = cap * (risk_pct / 100.0)
            shares = int(min(max_risk_val // risk_per_sh, cap // entry_p))
            st.success(f"**Buy Quantity:** {shares} Shares")
            st.write(f"**Total Capital Needed:** ₹{round(shares * entry_p, 2)}")
            st.write(f"**Max Loss:** ₹{round(shares * risk_per_sh, 2)}")
            st.write(f"**Max Gain:** ₹{round(shares * (tgt_p - entry_p), 2)}")

# ==============================================================================
# TAB 5: ACTIVE PORTFOLIO & REAL-TIME EXIT SIGNALS
# ==============================================================================
with tab5:
    st.subheader("Active Position Monitor & Exit Trigger Engine")
    st.caption("Tracks your active trades, compares live prices with targets/stops, and evaluates MACD/trend reversals to alert you when to sell.")

    port_df = st.session_state.portfolio

    if port_df.empty:
        st.info("No active trades in portfolio. Add stocks from Tab 2 after scanning.")
    else:
        st.write("### Active Monitored Positions")
        
        status_reports = []
        for index, row in port_df.iterrows():
            sym = row['Symbol']
            entry = float(row['Entry_Price'])
            target = float(row['Target_Price'])
            sl = float(row['Stop_Loss'])
            
            # Fetch live current market data
            try:
                t = yf.Ticker(f"{sym}.NS")
                h = t.history(period="15d")
                if not h.empty:
                    current_price = float(h['Close'].iloc[-1])
                    h['MACD'], h['Signal'], _ = calculate_macd(h['Close'])
                    current_macd = float(h['MACD'].iloc[-1])
                    current_sig = float(h['Signal'].iloc[-1])
                    
                    pnl_pct = round(((current_price - entry) / entry) * 100, 2)
                    
                    # SELL SIGNAL RULES:
                    # 1. Target Reached (+5% or higher)
                    # 2. Stop Loss Breached (-2% or lower)
                    # 3. MACD Bearish Reversal (MACD crosses below Signal)
                    if current_price >= target:
                        action = "🎯 TAKE PROFIT (Target Hit)"
                        action_color = "green"
                    elif current_price <= sl:
                        action = "🛑 STOP LOSS (Exit Immediately)"
                        action_color = "red"
                    elif current_macd < current_sig:
                        action = "⚠️ SELL (Trend Reversal Detected)"
                        action_color = "orange"
                    else:
                        action = "✅ HOLD (Trend Still Bullish)"
                        action_color = "blue"

                    status_reports.append({
                        "Symbol": sym,
                        "Entry Price (₹)": entry,
                        "Live Price (₹)": round(current_price, 2),
                        "P&L (%)": f"{pnl_pct}%",
                        "Target (₹)": target,
                        "Stop Loss (₹)": sl,
                        "Exit Recommendation": action
                    })
            except Exception:
                continue

        if status_reports:
            st.dataframe(pd.DataFrame(status_reports), use_container_width=True)

        # Clear/Delete Trade Option
        st.markdown("---")
        del_sym = st.selectbox("Select trade to remove after closing:", port_df['Symbol'].tolist(), key="del_sym_select")
        if st.button("Remove Trade from Portfolio", key="del_trade_btn"):
            st.session_state.portfolio = port_df[port_df['Symbol'] != del_sym].reset_index(drop=True)
            save_portfolio(st.session_state.portfolio)
            st.success(f"Removed {del_sym} from active portfolio.")
            st.rerun()
