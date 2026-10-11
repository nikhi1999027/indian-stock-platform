import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import feedparser
import urllib.parse
from textblob import TextBlob
from datetime import datetime

# Streamlit Page Configuration
st.set_page_config(page_title="Indian Stock Platform", page_icon="📈", layout="wide")

# Helper Function: Google News Sentiment
def fetch_news_sentiment(company_name):
    encoded_query = urllib.parse.quote(f"{company_name} share stock news India")
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    polarity_scores = [TextBlob(entry.title).sentiment.polarity for entry in feed.entries[:3]]
    avg_polarity = sum(polarity_scores) / len(polarity_scores) if polarity_scores else 0.0
    label = "POSITIVE" if avg_polarity > 0.05 else ("NEGATIVE" if avg_polarity < -0.05 else "NEUTRAL")
    headlines = [entry.title for entry in feed.entries[:3]]
    return {"label": label, "score": round(avg_polarity, 2), "headlines": headlines}

# Main Application Title
st.title("📈 Private Indian Stock Analysis Platform")

# 4 Core Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs(["1. Live Stock Analysis", "2. Market Scanner", "3. Strategy Backtest", "4. Position Sizer"])

# TAB 1: LIVE ANALYSIS
with tab1:
    st.subheader("Single Stock Multi-Factor Analysis")
    symbol = st.text_input("Enter NSE Symbol:", value="RELIANCE", key="t1_s").upper()
    if st.button("Analyze Stock", key="t1_b"):
        with st.spinner("Fetching data..."):
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                df = ticker.history(period="60d")
                if df.empty:
                    st.error("No data found for symbol.")
                else:
                    info = ticker.info if hasattr(ticker, 'info') else {}
                    company = info.get('shortName', symbol)
                    pe_ratio = info.get('trailingPE', 'N/A')
                    
                    df['SMA_20'] = ta.sma(df['Close'], length=20)
                    df['RSI_14'] = ta.rsi(df['Close'], length=14)
                    
                    latest = df.iloc[-1]
                    price, rsi, sma = float(latest['Close']), float(latest['RSI_14']), float(latest['SMA_20'])
                    sentiment = fetch_news_sentiment(company)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Current Price", f"₹{round(price, 2)}")
                    c2.metric("RSI (14)", round(rsi, 2))
                    c3.metric("20-Day SMA", f"₹{round(sma, 2)}")
                    c4.metric("P/E Ratio", pe_ratio)
                    
                    st.success(f"Target Price (+5%): ₹{round(price * 1.05, 2)} | Stop Loss (-2%): ₹{round(price * 0.98, 2)}")
                    st.write(f"**News Sentiment:** {sentiment['label']} ({sentiment['score']})")
                    for h in sentiment['headlines']:
                        st.write(f"- {h}")
            except Exception as e:
                st.error(f"Error: {e}")

# TAB 2: MARKET SCANNER
with tab2:
    st.subheader("Nifty Volume Spike Scanner")
    if st.button("Scan Market", key="t2_b"):
        with st.spinner("Scanning top NSE stocks..."):
            sample_watchlist = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BHARTIARTL", "SBIN", "TATAMOTORS", "LT", "ITC"]
            yf_symbols = [f"{s}.NS" for s in sample_watchlist]
            data = yf.download(yf_symbols, period="60d", group_by='ticker', progress=False)
            
            candidates = []
            for s in sample_watchlist:
                t_ns = f"{s}.NS"
                if t_ns in data and not data[t_ns].empty:
                    d = data[t_ns].dropna(subset=['Close', 'Volume'])
                    if len(d) >= 20:
                        d['SMA_20'] = ta.sma(d['Close'], length=20)
                        d['RSI_14'] = ta.rsi(d['Close'], length=14)
                        d['Vol_SMA_20'] = ta.sma(d['Volume'], length=20)
                        lat = d.iloc[-1]
                        vol_r = round(float(lat['Volume']) / float(lat['Vol_SMA_20']), 2) if float(lat['Vol_SMA_20']) > 0 else 1.0
                        if vol_r >= 1.0:
                            candidates.append({
                                "Symbol": s, "Price (₹)": round(float(lat['Close']), 2),
                                "RSI": round(float(lat['RSI_14']), 2), "Volume Spike": f"{vol_r}x Avg",
                                "Target (₹)": round(float(lat['Close']) * 1.05, 2),
                                "Stop Loss (₹)": round(float(lat['Close']) * 0.98, 2)
                            })
            if candidates:
                scan_df = pd.DataFrame(candidates)
                st.dataframe(scan_df, use_container_width=True)
            else:
                st.info("No volume spike opportunities found.")

# TAB 3: BACKTESTER
with tab3:
    st.subheader("1-Year Strategy Backtest")
    bt_sym = st.text_input("NSE Symbol for Backtest:", value="TCS", key="t3_s").upper()
    if st.button("Run Backtest", key="t3_b"):
        with st.spinner("Calculating performance..."):
            df = yf.download(f"{bt_sym}.NS", period="1y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['RSI_14'] = ta.rsi(df['Close'], length=14)
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

# TAB 4: POSITION SIZER
with tab4:
    st.subheader("Position Size Calculator")
    cap = st.number_input("Total Capital (₹):", value=100000.0)
    risk_pct = st.number_input("Max Risk (%):", value=2.0)
    entry_p = st.number_input("Entry Price (₹):", value=2500.0)
    sl_p = st.number_input("Stop Loss Price (₹):", value=2450.0)
    tgt_p = st.number_input("Target Price (₹):", value=2625.0)
    
    if st.button("Calculate Position", key="t4_b"):
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
