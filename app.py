"""
app.py — Global Quantitative Stock & Options Trading Platform  (v2.0)
==============================================================================
Every new or modified block is tagged  # [CHG-xx]  — see CHANGELOG.md for the
full list (what changed, why, and which section it touches).
Untagged code is carried over from v1 unchanged in behaviour.
==============================================================================
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import urllib.parse
import requests
import os
import io
import math
import base64
import uuid
from datetime import datetime, timedelta, date, timezone
from zoneinfo import ZoneInfo  # [CHG-09] market-hours awareness

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
JOURNAL_FILE = "trade_journal.csv"  # [CHG-32] closed-trade journal

# [CHG-25][CHG-23][CHG-20] Extended schemas (old CSVs load fine; missing columns are added)
EQ_COLS = ["Row_ID", "Symbol", "Entry_Price", "Target_Price", "Stop_Loss", "Entry_Date",
           "Qty", "Initial_Stop", "ATR_Entry", "Setup"]
OPT_COLS = ["Row_ID", "Contract", "Underlying", "Option_Type", "Strike", "Expiry", "Lots", "Lot_Size",
            "Entry_Premium", "Target_Premium", "Stop_Loss_Premium", "Underlying_SL", "Last_Premium", "Entry_Date"]
JOURNAL_COLS = ["Market", "Instrument", "Direction", "Entry", "Exit", "Qty", "PnL", "PnL_Pct",
                "R_Multiple", "Entry_Date", "Exit_Date", "Exit_Reason", "Setup"]

# [CHG-09][CHG-11][CHG-22] Per-market constants
MARKETS = {
    "IN": {"tz": "Asia/Kolkata", "open": (9, 15), "close": (15, 30), "bench": "^NSEI", "bench_name": "NIFTY 50",
           "vix": "^INDIAVIX", "suffix": ".NS", "ccy": "₹", "min_value": 1e8, "rf": 0.06, "cost_pct": 0.15},
    "US": {"tz": "America/New_York", "open": (9, 30), "close": (16, 0), "bench": "^GSPC", "bench_name": "S&P 500",
           "vix": "^VIX", "suffix": "", "ccy": "$", "min_value": 2.5e7, "rf": 0.04, "cost_pct": 0.05},
}

# [CHG-14][CHG-16] Index metadata — lot sizes per NSE revision effective Jan-2026 series. VERIFY periodically.
INDEX_DICT = {
    "NIFTY 50": {"yf": "^NSEI", "nse": "NIFTY", "step": 50, "lot": 65, "weekly": True},
    "BANK NIFTY": {"yf": "^NSEBANK", "nse": "BANKNIFTY", "step": 100, "lot": 30, "weekly": False},
}
UNDERLYING_YF = {v["nse"]: v["yf"] for v in INDEX_DICT.values()}

# [CHG-14] TATAMOTORS was renamed TMPV (Oct-2025 demerger). Expanded liquid F&O list.
FNO_WATCHLIST = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "TMPV", "SBIN", "BHARTIARTL", "AXISBANK", "LT",
    "KOTAKBANK", "MARUTI", "BAJFINANCE", "ITC", "HINDUNILVR", "SUNPHARMA", "TITAN", "M&M", "NTPC",
    "HCLTECH", "TATASTEEL", "JSWSTEEL", "HINDALCO", "ADANIENT", "BEL", "TRENT", "ONGC", "COALINDIA",
    "DRREDDY", "CIPLA", "EICHERMOT", "ULTRACEMCO", "ASIANPAINT", "BAJAJFINSV", "WIPRO", "TECHM"
]
SYMBOL_RENAMES = {"TATAMOTORS": "TMPV"}  # [CHG-14]


def round_to_strike(price, step=50):
    return int(round(price / step) * step)


def stock_strike_step(price):
    """[CHG-15] Heuristic NSE stock-option strike interval. Always confirm on the live chain."""
    if price < 250:
        return 5
    if price < 500:
        return 10
    if price < 1000:
        return 20
    if price < 2500:
        return 50
    if price < 5000:
        return 100
    return 250


def show_df(df, **kw):
    """[CHG-30] Streamlit >=1.4x uses width='stretch'; older builds use use_container_width."""
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)


def _f(x, default=np.nan):
    """Safe float conversion."""
    try:
        v = float(x)
        return default if (v != v) else v
    except Exception:
        return default


# ==============================================================================
# 0. PERSISTENCE  [CHG-24]
#    Streamlit Community Cloud has an EPHEMERAL filesystem: local CSVs are wiped
#    on every reboot/redeploy. If st.secrets["github"] is configured, portfolios
#    are also mirrored to a GitHub repo (use a SEPARATE private data repo, or a
#    branch the app does not deploy from — pushing to the app's own branch
#    triggers a redeploy on every save).
#    secrets.toml:
#      [github]
#      token = "ghp_..."            # fine-grained PAT, contents:read/write
#      repo = "youruser/trading-data"
#      branch = "main"
#      path = "data"
# ==============================================================================
def _gh_cfg():
    try:
        cfg = st.secrets["github"]
        if cfg.get("token") and cfg.get("repo"):
            return cfg
    except Exception:
        pass
    return None


def _gh_url(cfg, fname):
    return f"https://api.github.com/repos/{cfg['repo']}/contents/{cfg.get('path', 'data')}/{fname}"


def _gh_headers(cfg):
    return {"Authorization": f"Bearer {cfg['token']}", "Accept": "application/vnd.github+json"}


def gh_read(fname):
    cfg = _gh_cfg()
    if not cfg:
        return None
    try:
        r = requests.get(_gh_url(cfg, fname), headers=_gh_headers(cfg),
                         params={"ref": cfg.get("branch", "main")}, timeout=8)
        if r.status_code == 200:
            return base64.b64decode(r.json()["content"]).decode("utf-8")
    except Exception:
        pass
    return None


def gh_write(fname, text):
    cfg = _gh_cfg()
    if not cfg:
        return None
    try:
        url = _gh_url(cfg, fname)
        branch = cfg.get("branch", "main")
        sha = None
        r = requests.get(url, headers=_gh_headers(cfg), params={"ref": branch}, timeout=8)
        if r.status_code == 200:
            sha = r.json().get("sha")
        body = {"message": f"update {fname} {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
                "content": base64.b64encode(text.encode("utf-8")).decode("ascii"), "branch": branch}
        if sha:
            body["sha"] = sha
        w = requests.put(url, headers=_gh_headers(cfg), json=body, timeout=10)
        return w.status_code in (200, 201)
    except Exception:
        return False


def _ensure_ids(df):
    """[CHG-25] Unique row id so duplicate symbols can be removed individually."""
    if "Row_ID" not in df.columns:
        df["Row_ID"] = None
    ids = df["Row_ID"].astype(str)
    mask = df["Row_ID"].isna() | ids.isin(["", "nan", "None"])
    if mask.any():
        df.loc[mask, "Row_ID"] = [uuid.uuid4().hex[:8] for _ in range(int(mask.sum()))]
    df["Row_ID"] = df["Row_ID"].astype(str)
    dup = df["Row_ID"].duplicated(keep="first")  # [REV-1] repair any collided ids
    if dup.any():
        df.loc[dup, "Row_ID"] = [uuid.uuid4().hex[:8] for _ in range(int(dup.sum()))]
    return df


def load_csv_data(filepath, columns):
    loaded_df = None
    text = gh_read(filepath)  # [CHG-24] remote copy wins if configured
    try:
        if text:
            loaded_df = pd.read_csv(io.StringIO(text), dtype={"Row_ID": str})  # [REV-1]
        elif os.path.exists(filepath):
            loaded_df = pd.read_csv(filepath, dtype={"Row_ID": str})  # [REV-1]
    except Exception:
        loaded_df = None
    if loaded_df is None:
        loaded_df = pd.DataFrame(columns=columns)
    for col in columns:
        if col not in loaded_df.columns:
            loaded_df[col] = None
    loaded_df = loaded_df[columns].astype(object)
    if "Row_ID" in columns:
        loaded_df = _ensure_ids(loaded_df)
    return loaded_df


def save_csv_data(df, filepath):
    df.to_csv(filepath, index=False)
    ok = gh_write(filepath, df.to_csv(index=False))  # [CHG-24]
    if ok is False:
        st.toast("⚠️ Saved locally, but GitHub backup failed — check secrets.", icon="⚠️")


if "eq_portfolio" not in st.session_state:
    st.session_state.eq_portfolio = load_csv_data(EQUITY_PORTFOLIO_FILE, EQ_COLS)
if "opt_portfolio" not in st.session_state:
    st.session_state.opt_portfolio = load_csv_data(OPTIONS_PORTFOLIO_FILE, OPT_COLS)
if "us_eq_portfolio" not in st.session_state:
    st.session_state.us_eq_portfolio = load_csv_data(US_EQUITY_PORTFOLIO_FILE, EQ_COLS)
if "journal" not in st.session_state:  # [CHG-32]
    st.session_state.journal = load_csv_data(JOURNAL_FILE, JOURNAL_COLS)

PORTFOLIO_KEYS = {"IN": ("eq_portfolio", EQUITY_PORTFOLIO_FILE), "US": ("us_eq_portfolio", US_EQUITY_PORTFOLIO_FILE)}


def log_to_journal(market, instrument, direction, entry, exit_p, qty, entry_date, reason, setup="", init_risk=None):
    """[CHG-32] Append a closed trade to the journal."""
    sign = 1 if direction == "LONG" else -1
    entry, exit_p, qty = _f(entry, 0.0), _f(exit_p, 0.0), _f(qty, 0.0)
    pnl = (exit_p - entry) * qty * sign
    pnl_pct = ((exit_p / entry) - 1) * 100 * sign if entry else 0.0
    r_mult = ((exit_p - entry) * sign / init_risk) if init_risk and init_risk > 0 else None
    row = pd.DataFrame([{
        "Market": market, "Instrument": instrument, "Direction": direction, "Entry": round(entry, 2),
        "Exit": round(exit_p, 2), "Qty": qty, "PnL": round(pnl, 2), "PnL_Pct": round(pnl_pct, 2),
        "R_Multiple": round(r_mult, 2) if r_mult is not None else None, "Entry_Date": entry_date,
        "Exit_Date": datetime.now().strftime("%Y-%m-%d %H:%M"), "Exit_Reason": reason, "Setup": setup
    }])
    st.session_state.journal = pd.concat([st.session_state.journal, row], ignore_index=True)
    save_csv_data(st.session_state.journal, JOURNAL_FILE)


# ==============================================================================
# 1. MARKET DATA LAYER  [CHG-02][CHG-03]
#    - >= 2y of daily bars so EMA50 / SMA200 / 52-week ranges are mathematically valid
#      (v1 used 60d, i.e. ~40 bars — EMA50 never warmed up; monitors used 10–15 bars for EMA20/MACD26).
#    - Cached to cut Yahoo rate-limit failures on Streamlit Cloud.
# ==============================================================================
def _extract_ticker_frame(data, ticker):
    if data is None or len(data) == 0:
        return None
    df = None
    if isinstance(data.columns, pd.MultiIndex):
        for lvl in range(data.columns.nlevels):
            if ticker in data.columns.get_level_values(lvl):
                df = data.xs(ticker, axis=1, level=lvl).copy()
                break
        if df is None:
            return None
    else:
        df = data.copy()
    df.columns = [str(c).strip().title() for c in df.columns]
    need = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in need):
        return None
    df = df[need].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close", "High", "Low"])
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["Volume"] = df["Volume"].fillna(0.0)
    return df if not df.empty else None


@st.cache_data(ttl=600, show_spinner=False)
def download_many(tickers, period="2y", interval="1d"):
    """Batch download (chunks of 100). Returns {ticker: OHLCV DataFrame}."""
    out = {}
    tickers = list(tickers)
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        try:
            data = yf.download(chunk, period=period, interval=interval, group_by="ticker",
                               auto_adjust=True, progress=False, threads=True)
        except Exception:
            continue
        for t in chunk:
            try:
                df = _extract_ticker_frame(data, t)
                if df is not None:
                    out[t] = df
            except Exception:
                continue
    return out


def get_history(ticker, period="2y"):
    return download_many((ticker,), period).get(ticker, pd.DataFrame())


def session_progress(market):
    """[CHG-09] Fraction of today's session elapsed (1.0 when closed)."""
    m = MARKETS[market]
    now = datetime.now(ZoneInfo(m["tz"]))
    if now.weekday() >= 5:
        return 1.0, now
    o = now.replace(hour=m["open"][0], minute=m["open"][1], second=0, microsecond=0)
    c = now.replace(hour=m["close"][0], minute=m["close"][1], second=0, microsecond=0)
    if now <= o or now >= c:
        return 1.0, now
    return max(0.02, (now - o).total_seconds() / (c - o).total_seconds()), now


def projected_rel_volume(df, market):
    """[CHG-09] v1 compared a half-day's volume to full-day averages (understates surges intraday).
    Projects today's volume to a full session. Cumulative intraday volume is front-loaded, so
    a concave curve (progress**0.85) is used — an approximation; label as 'projected'."""
    if df is None or len(df) < 22:
        return np.nan, False
    vol_avg = float(df["Volume"].iloc[-21:-1].mean())
    if vol_avg <= 0:
        return np.nan, False
    vol = float(df["Volume"].iloc[-1])
    prog, now = session_progress(market)
    is_today = df.index[-1].date() == now.date()
    if is_today and prog < 1.0:
        if prog < 0.08:  # [REV-4] first ~30 min: projection too noisy, volume factor skipped
            return np.nan, True
        return vol / (prog ** 0.85) / vol_avg, True
    return vol / vol_avg, False


# ==============================================================================
# 2. TECHNICAL INDICATOR ENGINE  (v1 formulas retained; new indicators tagged)
# ==============================================================================
def calculate_sma(series, window=20):
    """Calculates Simple Moving Average."""
    return series.rolling(window=window).mean()


def calculate_ema(series, span=20):
    """Calculates Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series, window=14):
    """Wilder's RSI, protected against division-by-zero."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    safe_loss = avg_loss.replace(0.0, 1e-6)
    rs = avg_gain / safe_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_macd(series, fast=12, slow=26, signal=9):
    """MACD Line, Signal Line, Histogram."""
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return sma + std * num_std, sma, sma - std * num_std


def _wilder(series, n):
    return series.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def calculate_atr(df, n=14):
    """[CHG-04] Wilder ATR — basis for volatility-adjusted stops/targets."""
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return _wilder(tr, n)


def calculate_adx(df, n=14):
    """[CHG-04][CHG-18] ADX / +DI / -DI — trend-strength filter (avoids buying options in chop)."""
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = calculate_atr(df, n).replace(0, np.nan)
    plus_di = 100 * _wilder(plus_dm, n) / atr
    minus_di = 100 * _wilder(minus_dm, n) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx, n), plus_di, minus_di


def add_indicators(df_in):
    """[CHG-04] One engine for every desk, scanner, monitor and backtest (no drift between them)."""
    df = df_in.copy()
    c = df["Close"]
    df["EMA_20"] = calculate_ema(c, 20)
    df["EMA_50"] = calculate_ema(c, 50)
    df["SMA_50"] = calculate_sma(c, 50)
    df["SMA_150"] = calculate_sma(c, 150)
    df["SMA_200"] = calculate_sma(c, 200)
    df["SMA200_Slope"] = df["SMA_200"] - df["SMA_200"].shift(21)
    df["RSI"] = calculate_rsi(c, 14)
    df["MACD"], df["Signal"], df["Hist"] = calculate_macd(c)
    df["Hist_Prev"] = df["Hist"].shift(1)
    df["ATR"] = calculate_atr(df, 14)
    df["ATR_Pct"] = df["ATR"] / c * 100
    df["ADX"], df["PDI"], df["MDI"] = calculate_adx(df, 14)
    vol_avg = df["Volume"].rolling(20).mean().shift(1)  # prior 20 sessions (excludes today)
    df["RelVol"] = df["Volume"] / vol_avg.replace(0, np.nan)
    df["High_20"] = df["High"].rolling(20).max().shift(1)  # prior 20-day high (breakout pivot)
    df["Low_20"] = df["Low"].rolling(20).min().shift(1)
    df["Low_5"] = df["Low"].rolling(5).min()
    df["High_5"] = df["High"].rolling(5).max()
    df["High_252"] = df["High"].rolling(252, min_periods=200).max()
    df["Low_252"] = df["Low"].rolling(252, min_periods=200).min()
    up, mid, lo = calculate_bollinger_bands(c)
    df["BBW"] = (up - lo) / mid
    df["BBW_Rank"] = df["BBW"].rolling(126, min_periods=60).rank(pct=True)  # squeeze detector
    logret = np.log(c / c.shift(1))
    df["HV20"] = logret.rolling(20).std() * math.sqrt(252) * 100
    df["HV_Rank"] = df["HV20"].rolling(252, min_periods=120).rank(pct=True)
    df["TradedValue20"] = (c * df["Volume"]).rolling(20).mean()
    # [CHG-28] Weekly (multi-timeframe) confirmation — completed weeks only, no look-ahead.
    wk = c.resample("W-FRI").last().dropna()
    if len(wk) >= 30:
        w_ema = calculate_ema(wk, 10)
        _, _, w_hist = calculate_macd(wk)
        w_up = ((wk > w_ema) & (w_hist > 0)).astype(float)
        w_dn = ((wk < w_ema) & (w_hist < 0)).astype(float)
        df["Weekly_Up"] = w_up.reindex(df.index, method="ffill").fillna(0.0)
        df["Weekly_Down"] = w_dn.reindex(df.index, method="ffill").fillna(0.0)
    else:
        df["Weekly_Up"] = 0.0
        df["Weekly_Down"] = 0.0
    return df


def sanitize_dataframe(df):
    """Retained for compatibility. Prefer _extract_ticker_frame()."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close", "Volume"])


# ==============================================================================
# 3. MARKET REGIME & RELATIVE STRENGTH  [CHG-05][CHG-06]
#    Most swing-trading edge comes from trading WITH the broad market. Leaders
#    (O'Neil / Minervini style) are bought only when the index is in an uptrend,
#    and only the strongest relative performers are selected.
# ==============================================================================
def regime_series(bench_df):
    """RISK-ON when index > SMA50 & SMA200 and SMA50 rising; RISK-OFF when <=1 of those hold."""
    c = bench_df["Close"]
    s50, s200 = calculate_sma(c, 50), calculate_sma(c, 200)
    pts = (c > s200).astype(int) + (c > s50).astype(int) + (s50 > s50.shift(10)).astype(int)
    lab = pd.Series(np.where(pts >= 3, "RISK-ON", np.where(pts == 2, "NEUTRAL", "RISK-OFF")), index=c.index)
    lab[s200.isna()] = "NEUTRAL"
    return lab


@st.cache_data(ttl=900, show_spinner=False)
def get_market_regime(market):
    m = MARKETS[market]
    data = download_many((m["bench"], m["vix"]), "2y")
    bench = data.get(m["bench"])
    out = {"label": "NEUTRAL", "bench_close": None, "vix": None, "vix_pct": None,
           "bench_ret_63": None, "details": "Benchmark data unavailable"}
    if bench is None or len(bench) < 210:
        return out
    lab = regime_series(bench)
    c = bench["Close"]
    s50, s200 = calculate_sma(c, 50).iloc[-1], calculate_sma(c, 200).iloc[-1]
    out.update({
        "label": lab.iloc[-1], "bench_close": c,
        "bench_ret_63": float(c.iloc[-1] / c.iloc[-64] - 1) * 100,
        "details": f"{m['bench_name']} {c.iloc[-1]:,.1f} | SMA50 {s50:,.1f} | SMA200 {s200:,.1f}",
        "regime_hist": lab,
    })
    vix = data.get(m["vix"])
    if vix is not None and len(vix) > 100:
        v = vix["Close"]
        out["vix"] = float(v.iloc[-1])
        out["vix_pct"] = float((v.iloc[-252:] <= v.iloc[-1]).mean())
    return out


def rs_raw_score(close):
    """IBD-style weighted 12-month performance (40% latest quarter)."""
    n = len(close)
    def r(k):
        k = min(k, n - 1)
        return float(close.iloc[-1] / close.iloc[-1 - k] - 1) if k > 0 else 0.0
    return 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)


def rs_proxy_rating(excess_63d_pct):
    """Single-stock RS proxy (deep dive / backtest) when no universe ranking is available."""
    if excess_63d_pct is None or excess_63d_pct != excess_63d_pct:
        return 50
    if excess_63d_pct > 10:
        return 85
    if excess_63d_pct > 0:
        return 65
    if excess_63d_pct > -10:
        return 45
    return 25


# ==============================================================================
# 4. CONFLUENCE SCORING ENGINE  [CHG-07][CHG-08][CHG-34][CHG-35]
#    v1 used all-or-nothing gates (one miss = invisible). v2 scores 0–100 so
#    candidates can be ranked and near-misses seen, with explicit reasons.
# ==============================================================================
def score_setup(r, direction="long", rs_rating=None, regime="NEUTRAL", relvol=None, sentiment=None, earnings_days=None):
    """[CHG-29] Symmetric scoring: direction="long" (cash longs / CE) or "short" (PE)."""
    L = direction == "long"
    pts, mx, why, flags = 0.0, 0.0, [], []

    def add(cond, p, label):
        nonlocal pts, mx
        mx += p
        if bool(cond):
            pts += p
            why.append(label)

    c, o = _f(r["Close"]), _f(r["Open"])
    e20, e50, s200 = _f(r["EMA_20"]), _f(r["EMA_50"]), _f(r["SMA_200"])
    atr = _f(r["ATR"], 0.0) or max(c * 0.02, 1e-6)
    rsi, adx = _f(r["RSI"], 50), _f(r["ADX"], 0)

    # Trend structure (25)
    if L:
        add(c > e20 > e50, 10, "Price > EMA20 > EMA50")
        add(c > s200 and _f(r["SMA200_Slope"], 0) > 0, 10, "Above rising 200-SMA")
        add(_f(r["Weekly_Up"], 0) > 0, 5, "Weekly trend up")
    else:
        add(c < e20 < e50, 10, "Price < EMA20 < EMA50")
        add(c < s200 and _f(r["SMA200_Slope"], 0) < 0, 10, "Below falling 200-SMA")
        add(_f(r["Weekly_Down"], 0) > 0, 5, "Weekly trend down")

    # Momentum (20)
    hist, macd, sig = _f(r["Hist"], 0), _f(r["MACD"], 0), _f(r["Signal"], 0)
    hist_prev = _f(r.get("Hist_Prev", np.nan), hist)
    if L:
        add(macd > sig and hist >= hist_prev, 8, "MACD bullish & expanding")
        rsi_pts = 7 if 50 <= rsi <= 70 else (3 if (45 <= rsi < 50 or 70 < rsi <= 75) else 0)
        add(_f(r["PDI"], 0) > _f(r["MDI"], 0) and adx >= 20, 5, f"ADX {adx:.0f} trending (+DI>-DI)")
    else:
        add(macd < sig and hist <= hist_prev, 8, "MACD bearish & expanding")
        rsi_pts = 7 if 30 <= rsi <= 50 else (3 if (25 <= rsi < 30 or 50 < rsi <= 55) else 0)
        add(_f(r["MDI"], 0) > _f(r["PDI"], 0) and adx >= 20, 5, f"ADX {adx:.0f} trending (-DI>+DI)")
    mx += 7
    pts += rsi_pts
    if rsi_pts:
        why.append(f"RSI {rsi:.0f} in momentum zone")

    # Volume on the right-coloured bar (15) — skipped for instruments without volume (indices)
    rv = relvol if relvol is not None else _f(r["RelVol"])
    if rv == rv and rv is not None and np.isfinite(rv):
        bar_ok = (c > o) if L else (c < o)
        vpts = (15 if rv >= 1.5 else 10 if rv >= 1.25 else 4 if rv >= 1.0 else 0) if bar_ok else 0
        mx += 15
        pts += vpts
        if vpts:
            why.append(f"{rv:.2f}x volume on {'green' if L else 'red'} bar")

    # Relative strength (20)
    if rs_rating is not None:
        if L:
            rpts = 20 if rs_rating >= 80 else 14 if rs_rating >= 70 else 6 if rs_rating >= 50 else 0
        else:
            rpts = 20 if rs_rating <= 20 else 14 if rs_rating <= 30 else 6 if rs_rating <= 50 else 0
        mx += 20
        pts += rpts
        if rpts:
            why.append(f"RS rating {rs_rating:.0f}")

    # Setup quality (10)
    setup = "Trend continuation"
    h20, l20, bbw_rank = _f(r["High_20"]), _f(r["Low_20"]), _f(r["BBW_Rank"], 1.0)
    if L:
        breakout = c > h20
        pullback = _f(r["Low"]) <= e20 + 0.5 * atr and c > e20 and c > o
    else:
        breakout = c < l20
        pullback = _f(r["High"]) >= e20 - 0.5 * atr and c < e20 and c < o
    squeeze = bbw_rank <= 0.20
    if breakout:
        setup, spts = ("20-day breakout" if L else "20-day breakdown"), 10
    elif pullback:
        setup, spts = "Pullback to EMA20", 8
    elif squeeze:
        setup, spts = "Volatility squeeze", 6
    else:
        spts = 0
    mx += 10
    pts += spts
    if spts:
        why.append(setup)

    # Location in 52-week range — Stage-2 / Stage-4 (10)
    h252, l252 = _f(r["High_252"]), _f(r["Low_252"])
    if h252 == h252 and l252 == l252:
        if L:
            add(c >= 0.75 * h252 and c >= 1.25 * l252, 10, "Stage-2: near 52w high, well off lows")
        else:
            add(c <= 1.25 * l252 and c <= 0.75 * h252, 10, "Stage-4: near 52w low, well off highs")

    score = 100.0 * pts / mx if mx else 0.0

    # Penalties / vetoes
    ext = ((c - e20) if L else (e20 - c)) / atr
    if ext > 2.5:
        score -= 10
        flags.append(f"Extended {ext:.1f} ATR from EMA20 — wait for pullback")
    if (L and rsi > 75) or ((not L) and rsi < 25):
        score -= 5
        flags.append("RSI stretched")
    if L:
        pen = {"RISK-OFF": 15, "NEUTRAL": 5}.get(regime, 0)
    else:
        pen = {"RISK-ON": 15, "NEUTRAL": 5}.get(regime, 0)
    if pen:
        score -= pen
        flags.append(f"Market regime {regime} (-{pen})")
    if sentiment is not None:
        if (L and sentiment < -0.25) or ((not L) and sentiment > 0.25):
            score -= 10
            flags.append("News sentiment against trade")
    if earnings_days is not None and 0 <= earnings_days <= 7:
        score -= 10
        flags.append(f"Earnings in {earnings_days}d — gap risk")

    score = max(0.0, min(100.0, score))
    grade = "A+" if score >= 85 else "A" if score >= 75 else "B" if score >= 65 else "C"
    return {"score": round(score, 1), "grade": grade, "setup": setup, "why": why, "flags": flags}


def compute_levels(r, direction="long", atr_mult=2.0, entry=None, t1_r=2.0, t2_r=3.0):
    """[CHG-10] Volatility-based stop: swing low/high, clamped to 1x..atr_mult x ATR.
    v1 used fixed +5% / -2%, which is too tight for volatile names and too loose for quiet ones."""
    L = direction == "long"
    entry = _f(entry if entry is not None else r["Close"])
    atr = _f(r["ATR"], 0.0) or entry * 0.02
    if L:
        swing = _f(r["Low_5"], entry - atr) - 0.25 * atr
        stop = min(max(swing, entry - atr_mult * atr), entry - 1.0 * atr)
        risk = entry - stop
        t1, t2 = entry + t1_r * risk, entry + t2_r * risk
    else:
        swing = _f(r["High_5"], entry + atr) + 0.25 * atr
        stop = max(min(swing, entry + atr_mult * atr), entry + 1.0 * atr)
        risk = stop - entry
        t1, t2 = entry - t1_r * risk, entry - t2_r * risk
    return {"entry": round(entry, 2), "stop": round(stop, 2), "risk": round(risk, 2),
            "t1": round(t1, 2), "t2": round(t2, 2), "risk_pct": round(risk / entry * 100, 2)}


def position_size(capital, risk_pct, entry, stop):
    risk_ps = abs(entry - stop)
    if risk_ps <= 0 or entry <= 0:
        return 0
    return int(max(0, min((capital * risk_pct / 100.0) // risk_ps, capital // entry)))


# ==============================================================================
# 5. NEWS SENTIMENT  [CHG-26][CHG-27]
#    TextBlob is a general-English lexicon: "shares plunge 8%" scores ~0.0. VADER plus
#    a finance lexicon handles headline language far better. Recency-weighted, de-duplicated,
#    last 7 days only. Used as a VETO/penalty, not as a trade trigger.
# ==============================================================================
FIN_LEXICON = {
    "upgrade": 2.0, "upgrades": 2.0, "upgraded": 2.0, "downgrade": -2.2, "downgrades": -2.2, "downgraded": -2.2,
    "beat": 1.6, "beats": 1.8, "miss": -1.8, "misses": -2.0, "missed": -1.8, "surge": 2.0, "surges": 2.0,
    "soar": 2.2, "soars": 2.2, "rally": 1.6, "rallies": 1.6, "jumps": 1.6, "jump": 1.4, "gains": 1.2, "gain": 1.0,
    "climbs": 1.2, "rises": 1.0, "rebound": 1.2, "rebounds": 1.2, "record": 0.8, "outperform": 1.8,
    "bullish": 2.0, "upbeat": 1.6, "buyback": 1.5, "dividend": 0.8, "bags": 1.2, "wins": 1.4, "approval": 1.4,
    "plunge": -2.6, "plunges": -2.6, "slump": -2.2, "slumps": -2.2, "tumble": -2.2, "tumbles": -2.2,
    "crash": -3.0, "crashes": -3.0, "falls": -1.2, "fall": -1.0, "drops": -1.2, "slides": -1.4, "sinks": -1.8,
    "selloff": -2.0, "sell-off": -2.0, "bearish": -2.0, "underperform": -1.8, "weak": -1.4, "loss": -1.6,
    "losses": -1.6, "fraud": -3.2, "probe": -1.6, "raid": -2.0, "penalty": -1.8, "fine": -0.6, "default": -2.6,
    "bankruptcy": -3.2, "insolvency": -3.0, "layoffs": -1.6, "resigns": -1.2, "pledge": -1.0, "ban": -1.8,
    "lawsuit": -1.6, "recall": -1.6, "warning": -1.4, "cuts": -1.0, "slashes": -1.6, "halt": -1.6,
}
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    _VADER.lexicon.update(FIN_LEXICON)
except Exception:
    _VADER = None


def _headline_score(text):
    if _VADER is not None:
        return _VADER.polarity_scores(text)["compound"]
    words = [w.strip(".,:;!?'\"()").lower() for w in text.split()]
    return max(-1.0, min(1.0, sum(FIN_LEXICON.get(w, 0.0) for w in words) / 4.0))


@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_news_sentiment(query, market="IN", max_items=12, max_age_days=7):
    if market == "US":
        q = urllib.parse.quote(f"{query} stock when:{max_age_days}d")
        rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    else:
        q = urllib.parse.quote(f"{query} share price when:{max_age_days}d")
        rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        resp = requests.get(rss_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        feed = feedparser.parse(resp.content)
    except Exception:
        feed = feedparser.parse(b"")
    seen, items = set(), []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for e in feed.entries:
        title = str(e.get("title", ""))
        base = title.rsplit(" - ", 1)[0].strip()
        key = base.lower()[:80]
        if not base or key in seen:
            continue
        seen.add(key)
        pub = e.get("published_parsed")
        age = (now - datetime(*pub[:6])).total_seconds() / 86400 if pub else 3.0
        if age > max_age_days:
            continue
        sc = _headline_score(base)
        items.append({"headline": base, "score": round(sc, 2), "age_days": round(age, 1), "w": math.exp(-age / 3.0)})
        if len(items) >= max_items:
            break
    wsum = sum(i["w"] for i in items)
    avg = round(sum(i["score"] * i["w"] for i in items) / wsum, 2) if wsum else 0.0
    label = "POSITIVE" if avg > 0.10 else ("NEGATIVE" if avg < -0.10 else "NEUTRAL")
    return {"label": label, "score": avg, "count": len(items),
            "headlines": [i["headline"] for i in items],
            "items": [{k: v for k, v in i.items() if k != "w"} for i in items]}


# ==============================================================================
# 6. PUBLIC DATA PROVIDERS
# ==============================================================================
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
}


@st.cache_data(ttl=1800, show_spinner=False)
def nse_reachable():
    """[REV-3] One quick probe, cached 30 min. If NSE blocks this server, skip all NSE calls."""
    try:
        r = requests.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def nse_get_json(path, params=None, referer="https://www.nseindia.com/option-chain"):
    """[CHG-19][CHG-31] Best-effort NSE JSON fetch. NSE often blocks cloud/non-Indian IPs
    (Streamlit Cloud runs in the US) — every caller degrades gracefully."""
    if not nse_reachable():
        return None
    try:
        s = requests.Session()
        s.headers.update(NSE_HEADERS)
        s.get("https://www.nseindia.com", timeout=6)
        s.headers["Referer"] = referer
        r = s.get("https://www.nseindia.com" + path, params=params, timeout=8)
        if r.status_code == 200 and r.text.strip().startswith(("{", "[")):
            return r.json()
    except Exception:
        pass
    return None


def _read_symbol_csv(url, headers=None):
    try:
        r = requests.get(url, headers=headers or {"User-Agent": "Mozilla/5.0"}, timeout=8)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            if "Symbol" in df.columns:
                return [SYMBOL_RENAMES.get(str(s).strip().upper(), str(s).strip().upper())
                        for s in df["Symbol"].dropna().unique()]
    except Exception:
        pass
    return []


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_indian_tickers(universe="Nifty 200"):
    """[CHG-13] v1 took the FIRST 100 rows of an alphabetical Nifty-500 list (A–C bias).
    v2 uses the official index files (Nifty 100/200/500) with fallbacks."""
    fname = {"Nifty 100": "ind_nifty100list.csv", "Nifty 200": "ind_nifty200list.csv",
             "Nifty 500": "ind_nifty500list.csv"}.get(universe, "ind_nifty200list.csv")
    for url in (f"https://archives.nseindia.com/content/indices/{fname}",
                f"https://www.niftyindices.com/IndexConstituent/{fname}",
                "https://raw.githubusercontent.com/kprohith/nse-stock-analysis/master/ind_nifty500list.csv"):
        syms = _read_symbol_csv(url, {"User-Agent": NSE_HEADERS["User-Agent"]})  # [REV-5]
        if len(syms) >= 50:
            return syms
    return list(dict.fromkeys(FNO_WATCHLIST + ["ULTRACEMCO", "WIPRO", "POWERGRID", "NESTLEIND", "GRASIM"]))


US_CURATED = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "COST", "NFLX", "PLTR", "JPM", "V",
    "CRM", "INTC", "QCOM", "AMAT", "MU", "WMT", "UNH", "CAT", "PANW", "DELL", "CRWD", "GS", "LLY", "ORCL",
    "ADBE", "LRCX", "KLAC", "ANET", "UBER", "NOW", "ISRG", "HD", "MA", "BAC", "XOM", "ABBV", "TXN", "BKNG",
    "MRVL", "ARM", "APP", "COIN", "HOOD", "SHOP", "MS", "GE"
]


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_us_tickers(universe="Curated leaders"):
    """[CHG-13] Optional full S&P 500 universe (public GitHub dataset) with curated fallback."""
    if universe == "S&P 500":
        try:
            r = requests.get("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv", timeout=10)
            df = pd.read_csv(io.StringIO(r.text))  # [REV-5] timeout instead of a possible hang
            syms = [str(s).replace(".", "-").strip().upper() for s in df["Symbol"].dropna()]
            if len(syms) > 400:
                return syms
        except Exception:
            pass
    return US_CURATED


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fii_dii_live():
    """[CHG-01][CHG-31] NSE JSON first, Moneycontrol table second. No fake placeholder data."""
    js = nse_get_json("/api/fiidiiTradeReact", referer="https://www.nseindia.com/reports/fii-dii")
    if isinstance(js, list) and js:
        df = pd.DataFrame(js)
        keep = [c for c in ["category", "date", "buyValue", "sellValue", "netValue"] if c in df.columns]
        if keep:
            return df[keep], "NSE"
    try:
        r = requests.get("https://www.moneycontrol.com/stocks/fii_dii_activity.php",
                         headers={"User-Agent": NSE_HEADERS["User-Agent"]}, timeout=6)
        if r.status_code == 200:
            for tbl in pd.read_html(io.StringIO(r.text)):
                col_str = " ".join([str(c) for c in tbl.columns])
                if "FII" in col_str or "DII" in col_str:
                    return tbl.dropna(how="all").head(5), "Moneycontrol"
    except Exception:
        pass
    return None, None


@st.cache_data(ttl=43200, show_spinner=False)
def get_next_earnings(ticker):
    """[CHG-12] Next earnings date (yfinance calendar). Returns ISO date or None."""
    try:
        cal = yf.Ticker(ticker).calendar
        d = None
        if isinstance(cal, dict):
            d = cal.get("Earnings Date")
            if isinstance(d, (list, tuple)):
                d = d[0] if d else None
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                d = cal.loc["Earnings Date"].iloc[0]
        if d is None:
            return None
        return pd.Timestamp(d).date().isoformat()
    except Exception:
        return None


def days_until(iso):
    if not iso:
        return None
    try:
        return (date.fromisoformat(iso) - date.today()).days
    except Exception:
        return None


# ==============================================================================
# 7. OPTIONS ANALYTICS  [CHG-16][CHG-17][CHG-19]
# ==============================================================================
def _last_weekday(y, m, wd):
    d = date(y + (m == 12), (m % 12) + 1, 1) - timedelta(days=1)
    while d.weekday() != wd:
        d -= timedelta(days=1)
    return d


# [REV-6] Exchange holidays that move an expiry to the previous trading day.
# Source: NSE 2026 expiry calendar. UPDATE EACH YEAR from the NSE holiday circular.
NSE_EXPIRY_HOLIDAYS = {date(2026, 10, 20), date(2026, 11, 10), date(2026, 11, 24)}


def _roll_back_holiday(d):
    while d in NSE_EXPIRY_HOLIDAYS or d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def computed_expiries(underlying, today=None, n=3):
    """NSE (2025-26 framework): NIFTY weekly every Tuesday; BANKNIFTY & stock options monthly,
    last Tuesday. Exchange holidays move expiry to the previous trading day — NOT handled here;
    the live NSE expiry list is preferred whenever reachable."""
    today = today or date.today()
    out = []
    if underlying == "NIFTY":
        d = today + timedelta(days=(1 - today.weekday()) % 7)
        while len(out) < n:
            e = _roll_back_holiday(d)
            if e >= today:
                out.append(e)
            d += timedelta(days=7)
    else:
        y, m = today.year, today.month
        while len(out) < n:
            d = _roll_back_holiday(_last_weekday(y, m, 1))
            if d >= today:
                out.append(d)
            m += 1
            if m > 12:
                y, m = y + 1, 1
    return out


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_greeks(S, K, T, r, sigma, typ="CE"):
    """Black-Scholes price, delta and theta/day (European; fine for NSE index/stock options)."""
    T = max(T, 1 / 365)
    sigma = max(sigma, 0.01)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    if typ == "CE":
        price = S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
        delta = _ncdf(d1)
        theta = (-S * pdf * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * _ncdf(d2)) / 365
    else:
        price = K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)
        delta = _ncdf(d1) - 1
        theta = (-S * pdf * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * _ncdf(-d2)) / 365
    return {"price": max(price, 0.05), "delta": delta, "theta": theta}


@st.cache_data(ttl=180, show_spinner=False)
def fetch_option_chain(symbol, is_index=True):
    """[CHG-19] NSE option chain → PCR, max pain, OI walls, ATM IV, live LTPs. Best effort."""
    info = nse_get_json("/api/option-chain-contract-info", {"symbol": symbol})
    expiries = (info.get("expiryDates") if isinstance(info, dict) else None) or []
    data, exp_used = None, (expiries[0] if expiries else None)
    if exp_used:
        data = nse_get_json("/api/option-chain-v3",
                            {"type": "Indices" if is_index else "Equity", "symbol": symbol, "expiry": exp_used})
    if not data:
        data = nse_get_json("/api/option-chain-indices" if is_index else "/api/option-chain-equities",
                            {"symbol": symbol})
    if not isinstance(data, dict) or "records" not in data:
        return None
    rec = data["records"]
    expiries = expiries or rec.get("expiryDates", [])
    exp_used = exp_used or (expiries[0] if expiries else None)
    rows = []
    for d in rec.get("data", []):
        exp = d.get("expiryDate") or d.get("expiryDates")
        if exp_used and exp and exp != exp_used:
            continue
        ce, pe = d.get("CE") or {}, d.get("PE") or {}
        rows.append({"strike": _f(d.get("strikePrice")),
                     "ce_oi": _f(ce.get("openInterest"), 0), "ce_chg_oi": _f(ce.get("changeinOpenInterest"), 0),
                     "ce_iv": _f(ce.get("impliedVolatility"), 0), "ce_ltp": _f(ce.get("lastPrice"), np.nan),
                     "pe_oi": _f(pe.get("openInterest"), 0), "pe_chg_oi": _f(pe.get("changeinOpenInterest"), 0),
                     "pe_iv": _f(pe.get("impliedVolatility"), 0), "pe_ltp": _f(pe.get("lastPrice"), np.nan)})
    if not rows:
        return None
    ch = pd.DataFrame(rows).dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    spot = _f(rec.get("underlyingValue"))
    tot_ce, tot_pe = ch["ce_oi"].sum(), ch["pe_oi"].sum()
    ks = ch["strike"].values
    pain = [float((ch["ce_oi"] * np.maximum(0, k - ks)).sum() + (ch["pe_oi"] * np.maximum(0, ks - k)).sum()) for k in ks]
    atm_row = ch.iloc[(ch["strike"] - spot).abs().argsort().iloc[0]] if spot == spot else None
    ivs = [v for v in ([atm_row["ce_iv"], atm_row["pe_iv"]] if atm_row is not None else []) if v and v > 0]
    return {"spot": spot, "expiry": exp_used, "expiries": expiries, "chain": ch,
            "pcr": round(tot_pe / tot_ce, 2) if tot_ce else None,
            "max_pain": float(ks[int(np.argmin(pain))]) if len(pain) else None,
            "call_wall": float(ch.loc[ch["ce_oi"].idxmax(), "strike"]),
            "put_wall": float(ch.loc[ch["pe_oi"].idxmax(), "strike"]),
            "atm_iv": round(float(np.mean(ivs)), 2) if ivs else None}


def parse_nse_date(s):
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(s), fmt).date()
        except Exception:
            continue
    return None


# ==============================================================================
# 8. SHARED EQUITY ENGINES (India & US use the same, tested logic)  [CHG-08]
# ==============================================================================
def run_equity_scan(symbols, market, min_score, atr_mult, direction="long", check_earnings=True, sentiment_top=0):
    m = MARKETS[market]
    yf_syms = [f"{s}{m['suffix']}" for s in symbols]
    data = download_many(tuple(yf_syms), "2y")
    regime = get_market_regime(market)
    feats, skipped = {}, 0
    for s, t in zip(symbols, yf_syms):
        df = data.get(t)
        if df is None or len(df) < 210:
            skipped += 1
            continue
        ind = add_indicators(df)
        last = ind.iloc[-1]
        if _f(last["TradedValue20"], 0) < m["min_value"]:  # [CHG-11] liquidity gate
            continue
        feats[s] = (ind, df)
    if not feats:
        return pd.DataFrame(), regime, skipped
    raw = pd.Series({s: rs_raw_score(v[0]["Close"]) for s, v in feats.items()})
    rs_rating = (raw.rank(pct=True) * 98 + 1).round(0)  # [CHG-06] 1–99 percentile within universe
    rows = []
    for s, (ind, df) in feats.items():
        last = ind.iloc[-1]
        rv, projected = projected_rel_volume(df, market)
        res = score_setup(last, direction, float(rs_rating[s]), regime["label"], relvol=rv)
        lv = compute_levels(last, direction, atr_mult)
        rows.append({
            "Symbol": s, "Score": res["score"], "Grade": res["grade"], "Setup": res["setup"],
            f"Price ({m['ccy']})": round(_f(last["Close"]), 2), "RS Rating": int(rs_rating[s]),
            "Rel Vol": f"{rv:.2f}x{' (proj)' if projected else ''}" if rv == rv else "n/a",
            "RSI": round(_f(last["RSI"]), 1), "ADX": round(_f(last["ADX"]), 1), "ATR %": round(_f(last["ATR_Pct"]), 2),
            f"Stop ({m['ccy']})": lv["stop"], "Risk %": lv["risk_pct"],
            f"T1 2R ({m['ccy']})": lv["t1"], f"T2 3R ({m['ccy']})": lv["t2"],
            "ATR": round(_f(last["ATR"]), 2), "Flags": "; ".join(res["flags"]), "Why": "; ".join(res["why"]),
        })
    out = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    # [CHG-12] earnings proximity check on the top names only (keeps the scan fast)
    if check_earnings and not out.empty:
        top = out[out["Score"] >= min_score - 10].head(20).index
        for i in top:
            sym = out.at[i, "Symbol"]
            ed = get_next_earnings(f"{sym}{m['suffix']}")
            dd = days_until(ed)
            if dd is not None and 0 <= dd <= 7:
                out.at[i, "Score"] = max(0.0, out.at[i, "Score"] - 10)
                out.at[i, "Flags"] = (out.at[i, "Flags"] + "; " if out.at[i, "Flags"] else "") + f"Earnings {ed} — gap risk"
        out = out.sort_values("Score", ascending=False).reset_index(drop=True)
    out["Grade"] = out["Score"].apply(lambda x: "A+" if x >= 85 else "A" if x >= 75 else "B" if x >= 65 else "C")
    return out, regime, skipped


def regime_banner(regime, market):
    lab = regime["label"]
    vix_txt = f" | VIX {regime['vix']:.1f} ({regime['vix_pct'] * 100:.0f}th pct, 1y)" if regime.get("vix") else ""
    msg = f"**Market regime: {lab}** — {regime['details']}{vix_txt}"
    if lab == "RISK-ON":
        st.success(msg)
    elif lab == "NEUTRAL":
        st.warning(msg + " — reduce size, be selective.")
    else:
        st.error(msg + " — long breakouts fail more often in downtrends; consider standing aside.")


def cfg():
    return st.session_state.get("_cfg", {"min_score": 65, "atr_mult": 2.0, "risk_pct": 1.0,
                                         "cap_IN": 500000.0, "cap_US": 10000.0, "nse_live": True})


def render_scanner(market):
    m = MARKETS[market]
    c = cfg()
    k = market.lower()
    col1, col2, col3 = st.columns(3)
    if market == "IN":
        universe = col1.selectbox("Universe:", ["Nifty 100", "Nifty 200", "Nifty 500"], index=1, key=f"{k}_univ")
    else:
        universe = col1.selectbox("Universe:", ["Curated leaders", "S&P 500"], index=0, key=f"{k}_univ")
    show_all = col2.toggle("Show near-misses (score ≥ 50)", value=False, key=f"{k}_nearmiss")
    chk_earn = col3.toggle("Check earnings dates", value=True, key=f"{k}_earn")
    st.caption(f"Ranks by 0–100 confluence score (trend, momentum, volume, relative strength, setup, 52-week location) "
               f"with regime, extension and earnings penalties. Minimum score: **{c['min_score']}** (sidebar).")

    if st.button("Run Full-Market Scanner", key=f"btn_exec_{k}_equity_scan"):
        with st.spinner("Downloading 2 years of daily data and scoring the universe..."):
            syms = fetch_indian_tickers(universe) if market == "IN" else fetch_us_tickers(universe)
            res, regime, skipped = run_equity_scan(syms, market, c["min_score"], c["atr_mult"], check_earnings=chk_earn)
            st.session_state[f"scan_{k}"] = {"df": res, "regime": regime, "n": len(syms), "skipped": skipped,
                                            "ts": datetime.now().strftime("%Y-%m-%d %H:%M")}

    sc = st.session_state.get(f"scan_{k}")
    if not sc:
        return
    regime_banner(sc["regime"], market)
    res = sc["df"]
    if res.empty:
        st.info("No data returned (Yahoo may be rate-limiting). Try again in a minute.")
        return
    thr = 50 if show_all else c["min_score"]
    view = res[res["Score"] >= thr]
    st.caption(f"Scanned {sc['n']} symbols at {sc['ts']} ({sc['skipped']} lacked 210+ bars). "
               f"{len(view)} at/above score {thr}. Prices are delayed (Yahoo).")
    if view.empty:
        st.info("No stocks passed the confluence threshold today. Capital preserved.")
        return
    show_df(view)
    st.download_button("📥 Download Scanner Results CSV", data=view.to_csv(index=False).encode("utf-8"),
                       file_name=f"{market}_Scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                       mime="text/csv", key=f"btn_dl_{k}_scan_csv")

    st.markdown("---")
    st.markdown("#### **⚡ Add Candidate to Active Exit Engine** (ATR stop, 2R target, risk-based quantity)")
    a1, a2 = st.columns([3, 1])
    sel = a1.selectbox("Select stock to track:", view["Symbol"].tolist(), key=f"sb_{k}_cand_sel")
    if a2.button("Track This Stock", key=f"btn_track_{k}_cand"):
        row = view[view["Symbol"] == sel].iloc[0]
        entry = float(row[f"Price ({m['ccy']})"])
        stop = float(row[f"Stop ({m['ccy']})"])
        qty = position_size(c[f"cap_{market}"], c["risk_pct"], entry, stop)
        new_entry = pd.DataFrame([{
            "Row_ID": uuid.uuid4().hex[:8], "Symbol": sel, "Entry_Price": entry,
            "Target_Price": float(row[f"T1 2R ({m['ccy']})"]), "Stop_Loss": stop,
            "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M"), "Qty": qty,
            "Initial_Stop": stop, "ATR_Entry": float(row["ATR"]), "Setup": row["Setup"]}])
        key, fpath = PORTFOLIO_KEYS[market]
        st.session_state[key] = pd.concat([st.session_state[key], new_entry], ignore_index=True)
        save_csv_data(st.session_state[key], fpath)
        st.success(f"✅ Added {sel} — qty {qty} (risking {c['risk_pct']}% of capital). See the portfolio monitor.")


def render_deep_dive(market):
    m = MARKETS[market]
    k = market.lower()
    c = cfg()
    default = "RELIANCE" if market == "IN" else "NVDA"
    sym = st.text_input("Enter Symbol:", value=default, key=f"ti_{k}_single").upper().strip()
    sym = SYMBOL_RENAMES.get(sym, sym)
    if not st.button("Analyze Stock", key=f"btn_{k}_single_eval"):
        return
    with st.spinner("Analyzing..."):
        df = get_history(f"{sym}{m['suffix']}", "2y")
        if df.empty or len(df) < 60:
            st.error("No (or too little) historical market data found for this symbol.")
            return
        ind = add_indicators(df)
        last = ind.iloc[-1]
        regime = get_market_regime(market)
        excess = None
        if regime["bench_close"] is not None and len(ind) > 64:
            b = regime["bench_close"].reindex(ind.index, method="ffill")
            excess = (float(ind["Close"].iloc[-1] / ind["Close"].iloc[-64]) - float(b.iloc[-1] / b.iloc[-64])) * 100
        rs = rs_proxy_rating(excess)
        sent = fetch_live_news_sentiment(sym, market=market)
        ed = get_next_earnings(f"{sym}{m['suffix']}")
        rv, proj = projected_rel_volume(df, market)
        res = score_setup(last, "long", rs, regime["label"], relvol=rv, sentiment=sent["score"], earnings_days=days_until(ed))
        lv = compute_levels(last, "long", c["atr_mult"])

    regime_banner(regime, market)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", f"{m['ccy']}{_f(last['Close']):,.2f}")
    c2.metric("Long Score", f"{res['score']} ({res['grade']})")
    c3.metric("Wilder's RSI (14)", round(_f(last["RSI"]), 1))
    c4.metric("ADX (14)", round(_f(last["ADX"]), 1))
    c5.metric(f"vs {m['bench_name']} (63d)", f"{excess:+.1f}%" if excess is not None else "n/a")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("ATR stop", f"{m['ccy']}{lv['stop']:,.2f}", f"-{lv['risk_pct']}%")
    d2.metric("Target 2R / 3R", f"{lv['t1']:,.2f} / {lv['t2']:,.2f}")
    d3.metric("Rel. volume", f"{rv:.2f}x{' proj' if proj else ''}" if rv == rv else "n/a")
    d4.metric("News sentiment", f"{sent['label']} ({sent['score']:+.2f})", f"{sent['count']} headlines")
    st.write(f"**Setup:** {res['setup']}  |  **Next earnings:** {ed or 'unknown'}  |  "
             f"**Weekly trend:** {'Up' if _f(last['Weekly_Up'], 0) else ('Down' if _f(last['Weekly_Down'], 0) else 'Mixed')}")
    if res["why"]:
        st.markdown("**Confluence present:** " + " · ".join(res["why"]))
    if res["flags"]:
        st.warning("**Cautions:** " + " · ".join(res["flags"]))
    st.line_chart(ind[["Close", "EMA_20", "EMA_50", "SMA_200"]].tail(250).dropna())
    if sent["items"]:
        st.write("#### Recent headlines (score, age in days):")
        for it in sent["items"]:
            st.write(f"- {it['headline']}  `({it['score']:+.2f}, {it['age_days']}d)`")


def backtest_symbol(df, bench_df, min_score, atr_mult, target_r, max_hold, cost_pct, risk_pct, capital0=100000.0):
    """[CHG-22] Backtests the SAME scoring logic the scanner uses (v1 tested an unrelated RSI<45 rule).
    Entry next-day open, stop/target on intraday high/low (stop assumed first if both touched),
    chandelier trail + breakeven at +1R, time stop, costs on both sides, risk-based sizing."""
    ind = add_indicators(df)
    if bench_df is not None and len(bench_df) > 210:
        reg = regime_series(bench_df).reindex(ind.index, method="ffill").fillna("NEUTRAL")
        bclose = bench_df["Close"].reindex(ind.index, method="ffill")
        excess = (ind["Close"].pct_change(63) - bclose.pct_change(63)) * 100
    else:
        reg = pd.Series("NEUTRAL", index=ind.index)
        excess = pd.Series(np.nan, index=ind.index)
    start = 210
    if len(ind) <= start + 20:
        return None
    cash, pos, pending, trades, curve, bars_in = capital0, None, None, [], [], 0
    idx = ind.index
    for i in range(start, len(ind)):
        bar = ind.iloc[i]
        o, h, l, c = float(bar["Open"]), float(bar["High"]), float(bar["Low"]), float(bar["Close"])
        if pending is not None and pos is None:
            lv = compute_levels(pending, "long", atr_mult, entry=o, t1_r=target_r)
            risk_ps = o - lv["stop"]
            qty = math.floor(min(cash * risk_pct / 100 / risk_ps, cash / (o * (1 + cost_pct / 100)))) if risk_ps > 0 else 0
            if qty > 0:
                fee = qty * o * cost_pct / 100
                cash -= qty * o + fee
                pos = {"entry": o, "qty": qty, "stop": lv["stop"], "init_stop": lv["stop"], "R": risk_ps,
                       "target": o + target_r * risk_ps, "i0": i, "hh": h, "fee_in": fee, "date": idx[i]}
            pending = None
        if pos is not None:
            bars_in += 1
            exit_p = reason = None
            if l <= pos["stop"]:
                exit_p = min(o, pos["stop"])
                reason = "Stop" if pos["stop"] < pos["entry"] else "Trail/BE"
            elif h >= pos["target"]:
                exit_p, reason = max(o, pos["target"]), "Target"
            elif i - pos["i0"] >= max_hold:
                exit_p, reason = c, "Time"
            if exit_p is not None:
                fee_out = pos["qty"] * exit_p * cost_pct / 100
                cash += pos["qty"] * exit_p - fee_out
                pnl = (exit_p - pos["entry"]) * pos["qty"] - pos["fee_in"] - fee_out
                trades.append({"Entry Date": pos["date"].date(), "Exit Date": idx[i].date(), "Entry": round(pos["entry"], 2),
                               "Exit": round(exit_p, 2), "Qty": pos["qty"], "PnL": round(pnl, 2),
                               "R": round(pnl / (pos["R"] * pos["qty"]), 2), "Bars": i - pos["i0"] + 1, "Reason": reason})
                pos = None
            else:
                pos["hh"] = max(pos["hh"], h)
                if pos["hh"] >= pos["entry"] + pos["R"]:
                    pos["stop"] = max(pos["stop"], pos["entry"])
                pos["stop"] = max(pos["stop"], pos["hh"] - 3 * float(bar["ATR"]))
        if pos is None and pending is None and i < len(ind) - 1:
            res = score_setup(bar, "long", rs_proxy_rating(_f(excess.iloc[i])), reg.iloc[i])
            if res["score"] >= min_score:
                pending = bar
        curve.append((idx[i], cash + (pos["qty"] * c if pos else 0.0)))
    eq = pd.Series([v for _, v in curve], index=[d for d, _ in curve])
    tdf = pd.DataFrame(trades)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    bh = ind["Close"].iloc[start:]
    wins = tdf[tdf["PnL"] > 0]["PnL"] if not tdf.empty else pd.Series(dtype=float)
    losses = tdf[tdf["PnL"] <= 0]["PnL"] if not tdf.empty else pd.Series(dtype=float)
    stats = {
        "Trades": len(tdf),
        "Win rate %": round(len(wins) / len(tdf) * 100, 1) if len(tdf) else 0.0,
        "Profit factor": round(wins.sum() / abs(losses.sum()), 2) if len(losses) and losses.sum() != 0 else (float("inf") if len(wins) else 0.0),
        "Expectancy (R)": round(tdf["R"].mean(), 2) if len(tdf) else 0.0,
        "Total return %": round((eq.iloc[-1] / capital0 - 1) * 100, 2),
        "CAGR %": round(((eq.iloc[-1] / capital0) ** (1 / years) - 1) * 100, 2),
        "Buy & hold CAGR %": round(((bh.iloc[-1] / bh.iloc[0]) ** (1 / years) - 1) * 100, 2),
        "Max drawdown %": round(((eq / eq.cummax()) - 1).min() * 100, 2),
        "Time in market %": round(bars_in / len(eq) * 100, 1),
    }
    return {"stats": stats, "trades": tdf, "equity": eq, "bh": bh / bh.iloc[0] * capital0}


def render_backtester(market):
    m = MARKETS[market]
    k = market.lower()
    c = cfg()
    b1, b2, b3 = st.columns(3)
    sym = b1.text_input("Symbol for Backtest:", value="TCS" if market == "IN" else "MSFT", key=f"ti_bt_{k}").upper().strip()
    sym = SYMBOL_RENAMES.get(sym, sym)
    period = b2.selectbox("History:", ["2y", "5y", "10y"], index=1, key=f"sb_bt_period_{k}")
    min_score = b3.slider("Min entry score:", 50, 90, int(c["min_score"]), 5, key=f"sl_bt_score_{k}")
    b4, b5, b6, b7 = st.columns(4)
    target_r = b4.number_input("Target (R multiple):", 1.0, 6.0, 2.0, 0.5, key=f"ni_bt_tr_{k}")
    max_hold = b5.number_input("Max hold (bars):", 5, 120, 20, 5, key=f"ni_bt_hold_{k}")
    cost = b6.number_input("Cost per side %:", 0.0, 1.0, m["cost_pct"], 0.01, key=f"ni_bt_cost_{k}",
                           help="Brokerage + STT/exchange fees + slippage, per side.")
    risk = b7.number_input("Risk per trade %:", 0.25, 5.0, c["risk_pct"], 0.25, key=f"ni_bt_risk_{k}")
    if not st.button("Run Backtest", key=f"btn_exec_backtest_{k}"):
        return
    with st.spinner("Backtesting..."):
        data = download_many((f"{sym}{m['suffix']}", m["bench"]), period)
        df = data.get(f"{sym}{m['suffix']}")
        if df is None or len(df) < 260:
            st.error("Not enough history for this symbol (need 260+ bars).")
            return
        out = backtest_symbol(df, data.get(m["bench"]), min_score, c["atr_mult"], target_r, int(max_hold), cost, risk)
    if out is None:
        st.error("Not enough data after indicator warm-up.")
        return
    s = out["stats"]
    r1 = st.columns(5)
    for col, key in zip(r1, ["Trades", "Win rate %", "Profit factor", "Expectancy (R)", "Max drawdown %"]):
        col.metric(key, s[key])
    r2 = st.columns(4)
    for col, key in zip(r2, ["Total return %", "CAGR %", "Buy & hold CAGR %", "Time in market %"]):
        col.metric(key, s[key])
    st.line_chart(pd.DataFrame({"Strategy": out["equity"], "Buy & hold": out["bh"]}))
    if not out["trades"].empty:
        show_df(out["trades"])
    st.caption("One symbol is a small sample. Judge a rule-set across many symbols and periods; positive expectancy "
               "(R > 0) and profit factor > 1.3 after costs matter more than win rate. Past results do not predict future returns.")


def render_sizer(market):
    m = MARKETS[market]
    k = market.lower()
    c = cfg()
    st.subheader("Equity Position Size Calculator")
    s1, s2, s3 = st.columns(3)
    cap = s1.number_input(f"Total Trading Capital ({m['ccy']}):", value=float(c[f"cap_{market}"]), step=10000.0, key=f"sz_cap_{k}")
    risk = s2.number_input("Max Risk per trade (%):", value=float(c["risk_pct"]), step=0.25, key=f"sz_risk_{k}")
    entry = s3.number_input(f"Entry Price ({m['ccy']}):", value=2500.0 if market == "IN" else 150.0, step=1.0, key=f"sz_entry_{k}")
    s4, s5 = st.columns(2)
    sl = s4.number_input(f"Stop Loss Price ({m['ccy']}):", value=2450.0 if market == "IN" else 145.0, step=1.0, key=f"sz_sl_{k}")
    tgt = s5.number_input(f"Target Price ({m['ccy']}):", value=2600.0 if market == "IN" else 160.0, step=1.0, key=f"sz_tgt_{k}")
    if st.button("Calculate Position", key=f"btn_calc_position_{k}"):
        risk_per_unit = entry - sl
        if risk_per_unit <= 0:
            st.error("Entry price must be higher than Stop Loss.")
        else:
            qty = position_size(cap, risk, entry, sl)
            rr = (tgt - entry) / risk_per_unit
            st.success(f"**Recommended Quantity:** {qty} shares  |  **Reward:Risk** = {rr:.2f}R")
            st.write(f"**Capital Committed:** {m['ccy']}{qty * entry:,.2f}")
            st.write(f"**Max Loss:** :red[-{m['ccy']}{qty * risk_per_unit:,.2f}]  |  **Max Gain:** :green[+{m['ccy']}{qty * (tgt - entry):,.2f}]")
            if rr < 1.5:
                st.warning("Reward:Risk below 1.5R — most professional swing traders skip these.")


def evaluate_equity_position(row, market):
    """[CHG-23] Exit logic: initial stop → breakeven at +1R → 3xATR chandelier trail.
    MACD crossover alone is now a 'watch' (v1 exited on any MACD dip = whipsaw)."""
    m = MARKETS[market]
    sym = str(row["Symbol"])
    df = get_history(f"{sym}{m['suffix']}", "2y")  # same period as scanner → shared cache
    if df.empty or len(df) < 60:
        return None
    ind = add_indicators(df)
    last = ind.iloc[-1]
    cur, atr, e20 = _f(last["Close"]), _f(last["ATR"]), _f(last["EMA_20"])
    entry, target, sl = _f(row["Entry_Price"]), _f(row["Target_Price"]), _f(row["Stop_Loss"])
    init_stop = _f(row.get("Initial_Stop"), sl)
    R = entry - init_stop if entry - init_stop > 0 else entry * 0.02
    edt = pd.to_datetime(row.get("Entry_Date"), errors="coerce")
    since = ind[ind.index >= edt.normalize()] if pd.notna(edt) else ind.tail(1)
    if since.empty:
        since = ind.tail(1)
    hh = float(since["High"].max())
    trail = max(init_stop, sl)
    if hh >= entry + R:
        trail = max(trail, entry)
    trail = max(trail, hh - 3 * atr)
    r_now = (cur - entry) / R
    days = (datetime.now() - edt.to_pydatetime()).days if pd.notna(edt) else 0
    if cur <= trail:
        action = "🔒 TRAILING STOP HIT — book profit" if trail >= entry else "🛑 STOP LOSS — exit"
    elif cur >= target:
        action = f"🎯 TARGET HIT — book 50%, trail rest at {trail:,.2f}"
    elif cur < e20 and _f(last["MACD"]) < _f(last["Signal"]):
        action = "⚠️ TREND WEAKENING — close < EMA20 & MACD bearish; tighten/exit"
    elif _f(last["MACD"]) < _f(last["Signal"]):
        action = "👀 WATCH — MACD crossed down (no price confirmation yet)"
    elif days >= 15 and r_now < 0.5:
        action = "⏳ TIME STOP — stalled 15+ days, consider freeing capital"
    else:
        action = "✅ HOLD (trend intact)"
    qty = _f(row.get("Qty"), 0)
    return {"Row_ID": row["Row_ID"], "Symbol": sym, f"Entry ({m['ccy']})": round(entry, 2), f"Live ({m['ccy']})": round(cur, 2),
            "P&L %": round((cur / entry - 1) * 100, 2), "R now": round(r_now, 2),
            "Unrealised": round((cur - entry) * qty, 2) if qty else None,
            f"Target ({m['ccy']})": round(target, 2), f"Initial SL": round(init_stop, 2),
            f"Active Stop ({m['ccy']})": round(trail, 2), "Days": days, "Exit Signal": action}


def render_equity_monitor(market):
    m = MARKETS[market]
    k = market.lower()
    key, fpath = PORTFOLIO_KEYS[market]
    port_df = st.session_state[key]

    with st.expander("➕ Manually Add an Equity Position"):
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        m_sym = mc1.text_input("Symbol:", key=f"ti_manual_{k}_sym").upper().strip()
        m_entry = mc2.number_input(f"Entry ({m['ccy']}):", value=2500.0 if market == "IN" else 120.0, key=f"ni_manual_{k}_entry")
        m_tgt = mc3.number_input(f"Target ({m['ccy']}):", value=2625.0 if market == "IN" else 127.2, key=f"ni_manual_{k}_tgt")
        m_sl = mc4.number_input(f"Stop ({m['ccy']}):", value=2450.0 if market == "IN" else 117.0, key=f"ni_manual_{k}_sl")
        m_qty = mc5.number_input("Qty:", value=0, step=1, key=f"ni_manual_{k}_qty")
        if st.button("Save Position to Monitor", key=f"btn_save_manual_{k}"):
            if m_sym:
                new_row = pd.DataFrame([{"Row_ID": uuid.uuid4().hex[:8], "Symbol": SYMBOL_RENAMES.get(m_sym, m_sym),
                                         "Entry_Price": m_entry, "Target_Price": m_tgt, "Stop_Loss": m_sl,
                                         "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M"), "Qty": m_qty,
                                         "Initial_Stop": m_sl, "ATR_Entry": None, "Setup": "Manual"}])
                st.session_state[key] = pd.concat([st.session_state[key], new_row], ignore_index=True)
                save_csv_data(st.session_state[key], fpath)
                st.success(f"Tracking {m_sym}!")
                st.rerun()

    render_backup_restore(key, fpath, EQ_COLS, k)

    if port_df.empty:
        st.info("No active trades in portfolio.")
        return
    st.subheader("Live Exit Matrix")
    rows = []
    with st.spinner("Refreshing positions..."):
        for _, row in port_df.iterrows():
            try:
                r = evaluate_equity_position(row, market)
                if r:
                    rows.append(r)
            except Exception:
                continue
    live = pd.DataFrame(rows)
    if not live.empty:
        show_df(live.drop(columns=["Row_ID"]))

    st.markdown("---")
    st.markdown("#### Close a position")
    labels = {r["Row_ID"]: f"{r['Symbol']} | entry {r['Entry_Price']} | {r['Entry_Date']} | {r['Row_ID']}"
              for _, r in port_df.iterrows()}
    rid = st.selectbox("Select trade:", list(labels.keys()), format_func=lambda x: labels[x], key=f"sb_close_{k}")
    sel = port_df[port_df["Row_ID"] == rid].iloc[0]
    live_px = float(live.loc[live["Row_ID"] == rid, f"Live ({m['ccy']})"].iloc[0]) if (not live.empty and (live["Row_ID"] == rid).any()) else _f(sel["Entry_Price"])
    x1, x2, x3 = st.columns(3)
    exit_px = x1.number_input(f"Exit price ({m['ccy']}):", value=float(live_px), key=f"ni_exit_{k}_{rid}")  # [REV-2]
    qty = x2.number_input("Qty closed:", value=int(_f(sel.get("Qty"), 0)), step=1, key=f"ni_exitqty_{k}_{rid}")
    reason = x3.selectbox("Reason:", ["Target", "Stop loss", "Trailing stop", "Trend reversal", "Time stop", "Discretionary"], key=f"sb_reason_{k}")
    y1, y2 = st.columns(2)
    if y1.button("Close & log to Journal", key=f"btn_close_log_{k}"):
        init_stop = _f(sel.get("Initial_Stop"), _f(sel["Stop_Loss"]))
        log_to_journal(market, sel["Symbol"], "LONG", sel["Entry_Price"], exit_px, qty, sel["Entry_Date"], reason,
                       str(sel.get("Setup") or ""), init_risk=_f(sel["Entry_Price"]) - init_stop)
        st.session_state[key] = port_df[port_df["Row_ID"] != rid].reset_index(drop=True)
        save_csv_data(st.session_state[key], fpath)
        st.success("Closed and journaled.")
        st.rerun()
    if y2.button("Remove without logging", key=f"btn_remove_{k}"):
        st.session_state[key] = port_df[port_df["Row_ID"] != rid].reset_index(drop=True)
        save_csv_data(st.session_state[key], fpath)
        st.rerun()


def render_backup_restore(state_key, fpath, cols, k):
    """[CHG-24] Manual backup/restore — works even without GitHub secrets."""
    with st.expander("💾 Backup / restore this portfolio (cloud storage is ephemeral)"):
        b1, b2 = st.columns(2)
        b1.download_button("Download backup CSV", data=st.session_state[state_key].to_csv(index=False).encode("utf-8"),
                           file_name=fpath, mime="text/csv", key=f"dl_backup_{k}_{state_key}")
        up = b2.file_uploader("Restore from CSV", type=["csv"], key=f"up_restore_{k}_{state_key}")
        if up is not None and b2.button("Restore now", key=f"btn_restore_{k}_{state_key}"):
            try:
                df = pd.read_csv(up, dtype={"Row_ID": str})  # [REV-1]
                for col in cols:
                    if col not in df.columns:
                        df[col] = None
                df = _ensure_ids(df[cols].astype(object))
                st.session_state[state_key] = df
                save_csv_data(df, fpath)
                st.success(f"Restored {len(df)} rows.")
                st.rerun()
            except Exception as e:
                st.error(f"Restore failed: {e}")


# ==============================================================================
# 9. OPTIONS POSITION EVALUATION  [CHG-20]
#    v1 stored premium target/stop but NEVER compared them to a live premium,
#    and computed EMA20 on 10 bars. v2 uses live NSE LTP when reachable, else the
#    premium you type in; adds expiry and time stops.
# ==============================================================================
def _strike_from_contract(contract):
    for tok in reversed(str(contract).split()):
        v = _f(tok)
        if v == v:
            return v
    return np.nan


def evaluate_option_position(row, use_live):
    und = str(row["Underlying"])
    is_index = und in UNDERLYING_YF
    df = get_history(UNDERLYING_YF.get(und, f"{und}.NS"), "2y")
    if df.empty or len(df) < 30:
        return None
    ind = add_indicators(df)
    last = ind.iloc[-1]
    spot, e20 = _f(last["Close"]), _f(last["EMA_20"])
    typ = str(row["Option_Type"])
    strike = _f(row.get("Strike"))
    if strike != strike:
        strike = _strike_from_contract(row["Contract"])
    exp = parse_nse_date(row.get("Expiry"))
    dte = (exp - date.today()).days if exp else None
    ltp, src = None, ""
    if use_live and strike == strike:
        ch = fetch_option_chain(und, is_index)
        if ch and (exp is None or parse_nse_date(ch["expiry"]) == exp):
            hit = ch["chain"][ch["chain"]["strike"] == strike]
            if not hit.empty:
                v = _f(hit.iloc[0]["ce_ltp" if typ == "CE" else "pe_ltp"])
                if v == v and v > 0:
                    ltp, src = v, "NSE live"
    if ltp is None:
        lp = _f(row.get("Last_Premium"))
        if lp == lp and lp > 0:
            ltp, src = lp, "manual"
    entry_p, tgt, slp, und_sl = (_f(row["Entry_Premium"]), _f(row["Target_Premium"]),
                                 _f(row["Stop_Loss_Premium"]), _f(row["Underlying_SL"]))
    edt = pd.to_datetime(row.get("Entry_Date"), errors="coerce")
    held = (datetime.now() - edt.to_pydatetime()).days if pd.notna(edt) else 0
    if ltp is not None and ltp <= slp:
        status = "🛑 EXIT NOW: premium stop hit"
    elif (typ == "CE" and spot <= und_sl) or (typ == "PE" and spot >= und_sl):
        status = "🛑 EXIT NOW: underlying stop breached"
    elif ltp is not None and ltp >= tgt:
        status = "🎯 TARGET: book 50–100%, trail the rest"
    elif dte is not None and dte <= 1:
        status = "⌛ EXIT: expiry ≤ 1 day — theta/gamma risk"
    elif held >= 3 and ltp is not None and ltp < entry_p * 1.05:
        status = "⏳ TIME STOP: no progress in 3+ days — theta is eating premium"
    elif (typ == "CE" and spot < e20) or (typ == "PE" and spot > e20):
        status = "⚠️ CAUTION: spot crossed 20-EMA against you"
    else:
        status = "✅ HOLD: trend intact"
    if ltp is None:
        status += " (enter live premium for premium stops)"
    units = _f(row.get("Lots"), 1) * _f(row.get("Lot_Size"), 0)
    return {"Row_ID": row["Row_ID"], "Contract": row["Contract"], "Expiry": row.get("Expiry"), "DTE": dte,
            "Entry Prem (₹)": entry_p, "Live Prem (₹)": round(ltp, 2) if ltp is not None else None, "Prem source": src,
            "P&L %": round((ltp / entry_p - 1) * 100, 1) if (ltp is not None and entry_p) else None,
            "P&L (₹)": round((ltp - entry_p) * units, 0) if (ltp is not None and units) else None,
            "Target Prem": tgt, "SL Prem": slp, "Live Spot": round(spot, 1), "Underlying SL": und_sl,
            "Exit Signal": status}


def pick_expiry(underlying, chain, min_dte=3):
    """[CHG-16] First expiry with at least `min_dte` days left (avoid buying into expiry week decay)."""
    cands = []
    if chain and chain.get("expiries"):
        cands = [d for d in (parse_nse_date(x) for x in chain["expiries"]) if d]
    if not cands:
        cands = computed_expiries(underlying, n=3)
    today = date.today()
    for d in sorted(cands):
        if (d - today).days >= min_dte:
            return d, (d - today).days
    d = sorted(cands)[-1]
    return d, (d - today).days


# ==============================================================================
# SIDEBAR NAVIGATION & GLOBAL SETTINGS
# ==============================================================================
NAV = [
    "1. Full-Market Equity Scanner (Indian Market)",
    "2. Active Equity Portfolio & Exit Alerts (Indian Market)",
    "3. High-Probability Options Trading Desk (Indian Market)",
    "4. Active Options Exit Alert Engine (Indian Market)",
    "5. Market Regime, FII/DII & Sentiment",          # [CHG-05] renamed/extended
    "6. US Equities Swing Trading Desk (US Market)",
    "7. Active US Swing Portfolio & Exit Monitor (US Market)",
    "8. Trade Journal & Performance",                  # [CHG-32] new
]
st.sidebar.title("⚡ Global Quant Suite")
app_mode = st.sidebar.radio("Select Platform Section:", NAV, key="nav_platform_mode_selector")

with st.sidebar.expander("⚙️ Risk & signal settings", expanded=False):  # [CHG-08][CHG-10]
    _min_score = st.slider("Minimum confluence score", 50, 90, 65, 5, key="cfg_min_score")
    _atr_mult = st.slider("Max stop distance (× ATR)", 1.5, 3.5, 2.0, 0.25, key="cfg_atr_mult")
    _risk = st.number_input("Risk per trade (% of capital)", 0.25, 3.0, 1.0, 0.25, key="cfg_risk")
    _cap_in = st.number_input("India trading capital (₹)", 10000.0, 1e9, 500000.0, 10000.0, key="cfg_cap_in")
    _cap_us = st.number_input("US trading capital ($)", 1000.0, 1e8, 10000.0, 1000.0, key="cfg_cap_us")
    _nse_live = st.toggle("Try live NSE option-chain / FII data", value=True, key="cfg_nse_live",
                          help="NSE often blocks cloud servers; the app falls back automatically.")
st.session_state["_cfg"] = {"min_score": _min_score, "atr_mult": _atr_mult, "risk_pct": _risk,
                            "cap_IN": _cap_in, "cap_US": _cap_us, "nse_live": _nse_live}
st.sidebar.caption("Data: Yahoo Finance (delayed), Google News RSS, NSE public pages. "
                   "Decision-support only — not investment advice.")
if _gh_cfg() is None:
    st.sidebar.caption("💾 Storage: local only (wiped on cloud reboot). Add GitHub secrets for durable storage.")

# ##############################################################################
# SECTION 1: FULL-MARKET EQUITY SCANNER (INDIAN MARKET)
# ##############################################################################
if app_mode == NAV[0]:
    st.title("📈 Indian Equities Multi-Indicator Swing Scanner")
    st.caption("Ranks liquid NSE equities by multi-factor confluence, market regime and relative strength.")
    t_scan, t_single, t_backtest, t_sizer = st.tabs(["🚀 Market Scanner", "🔍 Stock Deep Dive", "📊 Backtester", "🧮 Position Sizer"])
    with t_scan:
        st.subheader("Automated NSE Market Scanner")
        render_scanner("IN")
    with t_single:
        st.subheader("Single Stock Multi-Factor Analysis")
        render_deep_dive("IN")
    with t_backtest:
        st.subheader("Strategy Backtest (same rules as the scanner)")
        render_backtester("IN")
    with t_sizer:
        render_sizer("IN")

# ##############################################################################
# SECTION 2: ACTIVE EQUITY PORTFOLIO & EXIT ALERTS (INDIAN MARKET)
# ##############################################################################
elif app_mode == NAV[1]:
    st.title("🛡️ Active Equity Portfolio & Dynamic Exit Monitor (NSE)")
    st.caption("Initial ATR stop → breakeven at +1R → 3×ATR chandelier trailing stop; trend-weakening and time-stop alerts.")
    render_equity_monitor("IN")

# ##############################################################################
# SECTION 3: HIGH-PROBABILITY OPTIONS TRADING DESK (INDIAN MARKET)
# ##############################################################################
elif app_mode == NAV[2]:
    st.title("🎯 High-Probability Options Trading Desk (NSE)")
    st.caption("Directional option buying only when trend, momentum, ADX ≥ 20 and volatility pricing line up. "
               "Expiry, theta and IV are now part of the decision.")
    st.info("Reality check: SEBI's FY26 study found ~88% of individual F&O traders lost money, with option buyers "
            "carrying most losses. Keep risk per trade small and prefer spreads when premiums are rich.")
    c = cfg()
    opt_choice = st.radio("Choose Trading Segment:", ["Benchmark Indices (Nifty / Bank Nifty)", "Top F&O Momentum Stocks"],
                          horizontal=True, key="rb_options_segment_in")

    if opt_choice == "Benchmark Indices (Nifty / Bank Nifty)":
        st.subheader("Benchmark Index Directional Engine")
        idx_name = st.selectbox("Select Index:", list(INDEX_DICT.keys()), key="sb_index_choice_in")
        meta = INDEX_DICT[idx_name]
        if st.button("Evaluate High-Probability Setup", key="btn_eval_index_options_in"):
            with st.spinner(f"Analyzing multi-factor confluence for {idx_name}..."):
                df_idx = get_history(meta["yf"], "2y")
                if df_idx.empty or len(df_idx) < 210:
                    st.error("Insufficient market data available for index.")
                    st.session_state.idx_eval = None
                else:
                    ind = add_indicators(df_idx)
                    last = ind.iloc[-1]
                    regime = get_market_regime("IN")
                    sent = fetch_live_news_sentiment(idx_name, market="IN")
                    st.session_state.idx_eval = {
                        "name": idx_name, "last": last, "regime": regime, "sent": sent,
                        "long": score_setup(last, "long", None, regime["label"], sentiment=sent["score"]),
                        "short": score_setup(last, "short", None, regime["label"], sentiment=sent["score"]),
                        "chain": fetch_option_chain(meta["nse"], True) if c["nse_live"] else None,
                        "ts": datetime.now().strftime("%H:%M")}

        ev = st.session_state.get("idx_eval")
        if ev and ev["name"] == idx_name:
            last, regime, sent, chain = ev["last"], ev["regime"], ev["sent"], ev["chain"]
            spot = chain["spot"] if (chain and chain.get("spot") == chain.get("spot") and chain.get("spot")) else _f(last["Close"])
            regime_banner(regime, "IN")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Spot", f"{spot:,.1f}")
            col2.metric("Wilder's RSI (14)", round(_f(last["RSI"]), 1))
            col3.metric("ADX (14)", round(_f(last["ADX"]), 1))
            col4.metric("MACD Histogram", round(_f(last["Hist"]), 2))
            col5.metric("News Sentiment", sent["label"], f"{sent['score']:+.2f}")
            s1, s2 = st.columns(2)
            s1.metric("CALL (long) score", f"{ev['long']['score']} ({ev['long']['grade']})")
            s2.metric("PUT (short) score", f"{ev['short']['score']} ({ev['short']['grade']})")

            # Volatility pricing [CHG-17]
            hv, hv_rank = _f(last["HV20"]), _f(last["HV_Rank"])
            atm_iv = chain["atm_iv"] if chain else None
            if atm_iv:
                iv, iv_src = atm_iv, "NSE ATM IV"
            elif meta["nse"] == "NIFTY" and regime.get("vix"):
                iv, iv_src = regime["vix"], "India VIX"
            else:
                iv, iv_src = hv * 1.1, "HV20 proxy"
            vix_pct = regime.get("vix_pct")
            iv_rich = (vix_pct is not None and vix_pct >= 0.8) or (atm_iv is not None and hv and atm_iv / hv >= 1.3)
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Implied vol used", f"{iv:.1f}%", iv_src)
            v2.metric("HV20 (realised)", f"{hv:.1f}%", f"{hv_rank * 100:.0f}th pct" if hv_rank == hv_rank else None)
            v3.metric("India VIX", f"{regime['vix']:.2f}" if regime.get("vix") else "n/a",
                      f"{vix_pct * 100:.0f}th pct (1y)" if vix_pct is not None else None)
            v4.metric("Premium pricing", "RICH" if iv_rich else "NORMAL/CHEAP")

            exp_d, dte = pick_expiry(meta["nse"], chain, min_dte=3)
            thr = c["min_score"]
            direction = None
            if ev["long"]["score"] >= thr and ev["long"]["score"] > ev["short"]["score"]:
                direction = "CE"
            elif ev["short"]["score"] >= thr and ev["short"]["score"] > ev["long"]["score"]:
                direction = "PE"

            st.markdown("---")
            if direction:
                res = ev["long"] if direction == "CE" else ev["short"]
                step = meta["step"]
                atm = round_to_strike(spot, step)
                strike = atm - step if direction == "CE" else atm + step  # 1 strike ITM: higher delta, less theta share
                T = max(dte, 1) / 365
                g = bs_greeks(spot, strike, T, MARKETS["IN"]["rf"], iv / 100, direction)
                live_ltp = None
                if chain and parse_nse_date(chain["expiry"]) == exp_d:
                    hit = chain["chain"][chain["chain"]["strike"] == strike]
                    if not hit.empty:
                        live_ltp = _f(hit.iloc[0]["ce_ltp" if direction == "CE" else "pe_ltp"])
                prem = live_ltp if (live_ltp and live_ltp == live_ltp) else g["price"]
                lv = compute_levels(last, "long" if direction == "CE" else "short", c["atr_mult"], entry=spot)
                d_abs = abs(g["delta"])
                theta_pct = abs(g["theta"]) / prem * 100
                exp_move = spot * iv / 100 * math.sqrt(max(dte, 1) / 365)
                # Cap the planned underlying move at the 1σ expected move to expiry: a 2R target
                # the market is unlikely to reach before expiry inflates the premium target.
                move = min(abs(lv["t1"] - spot), exp_move)
                if abs(lv["t1"] - spot) > exp_move:
                    st.warning(f"2R underlying target ({lv['t1']:,.0f}) lies beyond the 1σ expected move to expiry "
                               f"(±{exp_move:,.0f}). Premium target is capped at the expected move; consider a later expiry.")
                prem_sl = round(max(prem - d_abs * abs(spot - lv["stop"]), prem * 0.4), 1)
                prem_tgt = round(prem + d_abs * move - abs(g["theta"]) * min(3, dte), 1)
                contract_name = f"{idx_name} {exp_d.strftime('%d%b%y').upper()} {strike} {direction}"
                if iv_rich or theta_pct > 4:
                    far = round_to_strike(spot + move if direction == "CE" else spot - move, step)
                    structure = (f"**Debit spread** — Buy {strike} {direction}, Sell {far} {direction} "
                                 f"(premium is {'rich' if iv_rich else 'decaying fast'}; the short leg finances theta).")
                else:
                    structure = f"**Buy {strike} {direction}** (1 strike ITM, Δ≈{g['delta']:+.2f})."
                box = st.success if direction == "CE" else st.error
                box(f"### {'🟢 CALL' if direction == 'CE' else '🔴 PUT'} SETUP ({res['grade']}): {contract_name}")
                st.markdown(structure)
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Expiry / DTE", f"{exp_d:%d-%b}", f"{dte} days")
                p2.metric(f"Premium ({'live' if live_ltp else 'model'})", f"₹{prem:,.1f}", f"θ {theta_pct:.1f}%/day")
                p3.metric("Premium SL / Target", f"₹{prem_sl} / ₹{prem_tgt}")
                p4.metric("Expected move to expiry (1σ)", f"±{exp_move:,.0f}")
                st.write(f"**Underlying stop:** {lv['stop']:,.1f}  |  **Underlying target (2R):** {lv['t1']:,.1f}  |  "
                         f"Lot size {meta['lot']} (verify with your broker).")
                if chain:
                    if direction == "CE" and spot < chain["call_wall"] < lv["t1"]:
                        st.warning(f"Heavy call OI at {chain['call_wall']:,.0f} sits before your target — likely resistance.")
                    if direction == "PE" and lv["t1"] < chain["put_wall"] < spot:
                        st.warning(f"Heavy put OI at {chain['put_wall']:,.0f} sits before your target — likely support.")
                st.session_state.active_index_signal = {
                    "Contract": contract_name, "Underlying": meta["nse"], "Option_Type": direction, "Strike": strike,
                    "Expiry": exp_d.strftime("%d-%b-%Y"), "Lot_Size": meta["lot"], "Underlying_SL": lv["stop"],
                    "Prem": round(prem, 1), "Prem_SL": prem_sl, "Prem_Tgt": prem_tgt}
            else:
                st.session_state.active_index_signal = None
                st.warning(f"### ⚠️ NO TRADE SETUP — STAND ASIDE (best score {max(ev['long']['score'], ev['short']['score'])} < {thr})")

            with st.expander("Why? Full confluence breakdown"):
                for lab, rr in (("CALL", ev["long"]), ("PUT", ev["short"])):
                    st.markdown(f"**{lab} — {rr['score']}**: " + (" · ".join(rr["why"]) or "nothing aligned"))
                    if rr["flags"]:
                        st.markdown("&nbsp;&nbsp;Cautions: " + " · ".join(rr["flags"]))
            if chain:
                st.markdown("#### Live option-chain context (NSE)")
                o1, o2, o3, o4, o5 = st.columns(5)
                o1.metric("Expiry", chain["expiry"])
                o2.metric("PCR (OI)", chain["pcr"])
                o3.metric("Max pain", f"{chain['max_pain']:,.0f}" if chain["max_pain"] else "n/a")
                o4.metric("Call OI wall", f"{chain['call_wall']:,.0f}")
                o5.metric("Put OI wall", f"{chain['put_wall']:,.0f}")
                st.caption("OI walls = where option writers are concentrated; they often act as intraday resistance/support. "
                           "PCR is context only (contrarian readings are debated) and is not scored.")
            else:
                st.caption("Live NSE option chain unavailable from this server (NSE blocks many cloud IPs). "
                           "Premiums shown are Black-Scholes model values — check your broker's LTP before trading.")

        sig = st.session_state.get("active_index_signal")
        if sig and ev and ev["name"] == idx_name:
            a1, a2, a3, a4 = st.columns(4)
            prem_in = a1.number_input("Bought Option Premium (₹):", value=float(sig["Prem"]), step=1.0, key=f"ni_index_prem_in_opt_{sig['Contract']}")
            lots_in = a2.number_input("Lots:", value=1, min_value=1, step=1, key=f"ni_index_lots_in_opt_{sig['Contract']}")
            tgt_in = a3.number_input("Target premium (₹):", value=float(sig["Prem_Tgt"]), step=1.0, key=f"ni_index_tgt_in_opt_{sig['Contract']}")
            sl_in = a4.number_input("Stop premium (₹):", value=float(sig["Prem_SL"]), step=1.0, key=f"ni_index_sl_in_opt_{sig['Contract']}")
            risk_amt = (prem_in - sl_in) * lots_in * sig["Lot_Size"]
            st.caption(f"Risk on this position ≈ ₹{risk_amt:,.0f} "
                       f"({risk_amt / c['cap_IN'] * 100:.2f}% of India capital; your limit is {c['risk_pct']}%).")
            if st.button("Add to Options Exit Engine", key="btn_add_idx_to_exit_opt"):
                new_row = pd.DataFrame([{
                    "Row_ID": uuid.uuid4().hex[:8], "Contract": sig["Contract"], "Underlying": sig["Underlying"],
                    "Option_Type": sig["Option_Type"], "Strike": sig["Strike"], "Expiry": sig["Expiry"],
                    "Lots": lots_in, "Lot_Size": sig["Lot_Size"], "Entry_Premium": prem_in, "Target_Premium": tgt_in,
                    "Stop_Loss_Premium": sl_in, "Underlying_SL": sig["Underlying_SL"], "Last_Premium": prem_in,
                    "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")}])
                st.session_state.opt_portfolio = pd.concat([st.session_state.opt_portfolio, new_row], ignore_index=True)
                save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
                st.success(f"✅ Added {sig['Contract']} to Section 4!")

    else:
        st.subheader("F&O Stock Momentum Scanner (long & short)")
        if st.button("Scan F&O Universe for Confluence Setups", key="btn_scan_fno_momentum_in"):
            with st.spinner("Filtering F&O universe..."):
                regime = get_market_regime("IN")
                data_fno = download_many(tuple(f"{s}.NS" for s in FNO_WATCHLIST), "2y")
                feats = {s: add_indicators(data_fno[f"{s}.NS"]) for s in FNO_WATCHLIST
                         if f"{s}.NS" in data_fno and len(data_fno[f"{s}.NS"]) >= 210}
                rsr = (pd.Series({s: rs_raw_score(v["Close"]) for s, v in feats.items()}).rank(pct=True) * 98 + 1).round(0) if feats else pd.Series(dtype=float)
                fno_cands = []
                for s, ind in feats.items():
                    try:
                        last = ind.iloc[-1]
                        rv, proj = projected_rel_volume(data_fno[f"{s}.NS"], "IN")
                        lg = score_setup(last, "long", float(rsr[s]), regime["label"], relvol=rv)
                        sh = score_setup(last, "short", float(rsr[s]), regime["label"], relvol=rv)
                        best, typ = (lg, "CE") if lg["score"] >= sh["score"] else (sh, "PE")
                        if best["score"] < c["min_score"]:
                            continue
                        price = _f(last["Close"])
                        step = stock_strike_step(price)
                        atm = round_to_strike(price, step)
                        strike = atm - step if typ == "CE" else atm + step
                        exp_d, dte = pick_expiry(s, None, min_dte=7)
                        hv = _f(last["HV20"], 30.0)
                        g = bs_greeks(price, strike, max(dte, 1) / 365, MARKETS["IN"]["rf"], hv * 1.1 / 100, typ)
                        lv = compute_levels(last, "long" if typ == "CE" else "short", c["atr_mult"])
                        ed = get_next_earnings(f"{s}.NS")
                        dd = days_until(ed)
                        flags = best["flags"] + ([f"Results {ed} before expiry — IV crush risk"] if dd is not None and 0 <= dd <= dte else [])
                        fno_cands.append({
                            "Stock": s, "Score": best["score"], "Grade": best["grade"], "Setup": best["setup"],
                            "Spot Price (₹)": round(price, 2), "Recommended Contract": f"{s} {exp_d.strftime('%d%b%y').upper()} {strike} {typ}",
                            "Option_Type": typ, "Strike": strike, "Expiry": exp_d.strftime("%d-%b-%Y"), "DTE": dte,
                            "RS": int(rsr[s]), "Rel Vol": f"{rv:.2f}x{' (proj)' if proj else ''}" if rv == rv else "n/a",
                            "RSI": round(_f(last["RSI"]), 1), "ADX": round(_f(last["ADX"]), 1), "HV20 %": round(hv, 1),
                            "Underlying SL (₹)": lv["stop"], "Underlying T1 (₹)": lv["t1"],
                            "Model Prem (₹)": round(g["price"], 1), "Delta": round(g["delta"], 2),
                            "Prem SL (₹)": round(max(g["price"] - abs(g["delta"]) * abs(price - lv["stop"]), g["price"] * 0.4), 1),
                            "Prem T1 (₹)": round(g["price"] + abs(g["delta"]) * abs(lv["t1"] - price), 1),
                            "Flags": "; ".join(flags)})
                    except Exception:
                        continue
                st.session_state.active_fno_scan_results = sorted(fno_cands, key=lambda x: -x["Score"])
                st.session_state.active_fno_regime = regime

        if st.session_state.get("active_fno_regime"):
            regime_banner(st.session_state.active_fno_regime, "IN")
        results = st.session_state.get("active_fno_scan_results")
        if results:
            show_df(pd.DataFrame(results))
            st.caption("Model premiums use HV20×1.1 as an IV proxy — replace with your broker's LTP. "
                       "Strike intervals are heuristic; confirm on the live chain. Stock options expire monthly (last Tuesday).")
            st.markdown("---")
            cf1, cf2, cf3, cf4 = st.columns([2, 1, 1, 1])
            sel_fno_contract = cf1.selectbox("Select Option Contract:", [r["Recommended Contract"] for r in results], key="sb_fno_contract_sel_in")
            matched_fno = [r for r in results if r["Recommended Contract"] == sel_fno_contract][0]
            fno_prem = cf2.number_input("Option Premium (₹):", value=float(matched_fno["Model Prem (₹)"]), step=1.0, key=f"ni_fno_premium_in_desk_{sel_fno_contract}")  # [REV-2]
            fno_lot = cf3.number_input("Lot size:", value=0, min_value=0, step=1, key=f"ni_fno_lotsize_in_desk_{matched_fno['Stock']}",
                                       help="Stock lot sizes change periodically — enter from your broker.")
            fno_lots = cf4.number_input("Lots:", value=1, min_value=1, step=1, key=f"ni_fno_lots_in_desk_{sel_fno_contract}")
            if st.button("Add to Exit Engine", key="btn_add_fno_to_exit_desk"):
                scale = fno_prem / matched_fno["Model Prem (₹)"] if matched_fno["Model Prem (₹)"] else 1.0
                new_row = pd.DataFrame([{
                    "Row_ID": uuid.uuid4().hex[:8], "Contract": matched_fno["Recommended Contract"], "Underlying": matched_fno["Stock"],
                    "Option_Type": matched_fno["Option_Type"], "Strike": matched_fno["Strike"], "Expiry": matched_fno["Expiry"],
                    "Lots": fno_lots, "Lot_Size": fno_lot, "Entry_Premium": fno_prem,
                    "Target_Premium": round(matched_fno["Prem T1 (₹)"] * scale, 2),
                    "Stop_Loss_Premium": round(matched_fno["Prem SL (₹)"] * scale, 2),
                    "Underlying_SL": matched_fno["Underlying SL (₹)"], "Last_Premium": fno_prem,
                    "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")}])
                st.session_state.opt_portfolio = pd.concat([st.session_state.opt_portfolio, new_row], ignore_index=True)
                save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
                st.success(f"✅ Added {matched_fno['Recommended Contract']} to Section 4!")
        elif results is not None:
            st.info("No F&O stock met the confluence threshold. Standing aside is a position.")

# ##############################################################################
# SECTION 4: ACTIVE OPTIONS EXIT ALERT ENGINE (INDIAN MARKET)
# ##############################################################################
elif app_mode == NAV[3]:
    st.title("🛡️ Real-Time Options Exit Alert Engine (NSE)")
    st.caption("Premium stop/target (live NSE LTP or your input), underlying stop, expiry and time stops, 20-EMA caution.")
    c = cfg()
    port_df = st.session_state.opt_portfolio

    with st.expander("➕ Manually Log an Active Option Trade"):
        oc1, oc2, oc3, oc4 = st.columns(4)
        c_underlying = oc1.selectbox("Underlying Asset:", ["NIFTY", "BANKNIFTY"] + FNO_WATCHLIST, key="sb_log_und_in")
        c_type = oc2.selectbox("Option Type:", ["CE", "PE"], key="sb_log_type_in")
        _hist = get_history(UNDERLYING_YF.get(c_underlying, f"{c_underlying}.NS"), "2y")  # [REV-2] strike default near spot
        _spot = _f(_hist["Close"].iloc[-1]) if not _hist.empty else 25000.0
        _step = {v["nse"]: v["step"] for v in INDEX_DICT.values()}.get(c_underlying) or stock_strike_step(_spot)
        c_strike = oc3.number_input("Strike Price:", value=int(round_to_strike(_spot, _step)), step=int(_step), key=f"ni_log_strike_in_{c_underlying}")
        c_exp = oc4.date_input("Expiry:", value=computed_expiries(c_underlying, n=1)[0], key=f"di_log_exp_in_{c_underlying}")  # [REV-2]
        oc5, oc6, oc7, oc8, oc9 = st.columns(5)
        c_entry_prem = oc5.number_input("Entry Premium (₹):", value=120.0, step=1.0, key="ni_log_prem_in")
        c_tgt = oc6.number_input("Target Premium (₹):", value=round(c_entry_prem * 1.5, 1), step=1.0, key=f"ni_log_tgt_in_{c_entry_prem}")
        c_slp = oc7.number_input("Stop Premium (₹):", value=round(c_entry_prem * 0.7, 1), step=1.0, key=f"ni_log_slp_in_{c_entry_prem}")
        c_underlying_sl = oc8.number_input("Underlying Stop (₹):", value=24800.0, step=10.0, key="ni_log_und_sl_in")
        default_lot = {v["nse"]: v["lot"] for v in INDEX_DICT.values()}.get(c_underlying, 0)
        c_lot = oc9.number_input("Lot size:", value=int(default_lot), step=1, key=f"ni_log_lot_in_{c_underlying}")
        c_lots = st.number_input("Lots:", value=1, min_value=1, step=1, key="ni_log_lots_in")
        if st.button("Start Tracking Exit Signals", key="btn_start_tracking_options_in"):
            contract_code = f"{c_underlying} {c_exp.strftime('%d%b%y').upper()} {c_strike} {c_type}"
            new_trade = pd.DataFrame([{
                "Row_ID": uuid.uuid4().hex[:8], "Contract": contract_code, "Underlying": c_underlying, "Option_Type": c_type,
                "Strike": c_strike, "Expiry": c_exp.strftime("%d-%b-%Y"), "Lots": c_lots, "Lot_Size": c_lot,
                "Entry_Premium": c_entry_prem, "Target_Premium": c_tgt, "Stop_Loss_Premium": c_slp,
                "Underlying_SL": c_underlying_sl, "Last_Premium": c_entry_prem,
                "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M")}])
            st.session_state.opt_portfolio = pd.concat([st.session_state.opt_portfolio, new_trade], ignore_index=True)
            save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
            st.success(f"Tracking {contract_code}!")
            st.rerun()

    render_backup_restore("opt_portfolio", OPTIONS_PORTFOLIO_FILE, OPT_COLS, "opt")

    if port_df.empty:
        st.info("No active option trades currently logged.")
    else:
        st.markdown("#### Update live premiums (used when NSE live data is unavailable)")
        ed = st.data_editor(port_df[["Row_ID", "Contract", "Entry_Premium", "Last_Premium"]].astype({"Last_Premium": float, "Entry_Premium": float}),
                            disabled=["Row_ID", "Contract", "Entry_Premium"], hide_index=True, key="de_opt_prem")
        if st.button("Save premiums", key="btn_save_opt_prem"):
            upd = dict(zip(ed["Row_ID"], ed["Last_Premium"]))
            st.session_state.opt_portfolio["Last_Premium"] = st.session_state.opt_portfolio["Row_ID"].map(upd)
            save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
            st.rerun()

        st.subheader("Live Exit Status Matrix")
        live_reports = []
        with st.spinner("Refreshing option positions..."):
            for _, row in port_df.iterrows():
                try:
                    r = evaluate_option_position(row, c["nse_live"])
                    if r:
                        live_reports.append(r)
                except Exception:
                    continue
        live = pd.DataFrame(live_reports)
        if not live.empty:
            show_df(live.drop(columns=["Row_ID"]))

        st.markdown("---")
        st.markdown("#### Close a position")
        labels = {r["Row_ID"]: f"{r['Contract']} | entry ₹{r['Entry_Premium']} | {r['Row_ID']}" for _, r in port_df.iterrows()}
        rid = st.selectbox("Select trade:", list(labels.keys()), format_func=lambda x: labels[x], key="sb_close_opt")
        sel = port_df[port_df["Row_ID"] == rid].iloc[0]
        lp = _f(sel.get("Last_Premium"), _f(sel["Entry_Premium"]))
        x1, x2 = st.columns(2)
        exit_prem = x1.number_input("Exit premium (₹):", value=float(lp), key=f"ni_exit_opt_{rid}")  # [REV-2]
        reason = x2.selectbox("Reason:", ["Target", "Premium stop", "Underlying stop", "Expiry/time stop", "Discretionary"], key="sb_reason_opt")
        y1, y2 = st.columns(2)
        if y1.button("Close & log to Journal", key="btn_close_log_opt"):
            units = _f(sel.get("Lots"), 1) * _f(sel.get("Lot_Size"), 0)
            log_to_journal("IN-OPT", sel["Contract"], "LONG", sel["Entry_Premium"], exit_prem, units, sel["Entry_Date"],
                           reason, "Option buy", init_risk=_f(sel["Entry_Premium"]) - _f(sel["Stop_Loss_Premium"]))
            st.session_state.opt_portfolio = port_df[port_df["Row_ID"] != rid].reset_index(drop=True)
            save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
            st.rerun()
        if y2.button("Remove without logging", key="btn_remove_opt_trade_in"):
            st.session_state.opt_portfolio = port_df[port_df["Row_ID"] != rid].reset_index(drop=True)
            save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
            st.rerun()

# ##############################################################################
# SECTION 5: MARKET REGIME, INSTITUTIONAL FII/DII & SENTIMENT
# ##############################################################################
elif app_mode == NAV[4]:
    st.title("🏛️ Market Regime, Institutional Activity & Sentiment")
    c = cfg()
    st.subheader("Market regime (trade with the tide)")  # [CHG-05]
    for mk in ("IN", "US"):
        rg = get_market_regime(mk)
        regime_banner(rg, mk)

    st.markdown("---")
    st.subheader("Daily Institutional Cash Flow Activity (NSE/BSE)")  # [CHG-01] fixed NameError (fetch_fii_dII_live)
    fii_data, src = fetch_fii_dii_live() if c["nse_live"] else (None, None)
    if fii_data is not None:
        st.caption(f"Source: {src}")
        show_df(fii_data)
    else:
        st.info("Live FII/DII data unreachable from this server (NSE/Moneycontrol block many cloud IPs). "
                "Check nseindia.com → Reports → FII/DII Trading Activity.")

    if c["nse_live"]:
        ch = fetch_option_chain("NIFTY", True)
        if ch:
            st.subheader("NIFTY option-chain positioning")
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("PCR (OI)", ch["pcr"])
            o2.metric("Max pain", f"{ch['max_pain']:,.0f}" if ch["max_pain"] else "n/a")
            o3.metric("Call wall / Put wall", f"{ch['call_wall']:,.0f} / {ch['put_wall']:,.0f}")
            o4.metric("ATM IV", f"{ch['atm_iv']}%" if ch["atm_iv"] else "n/a")

    st.markdown("---")
    st.subheader("Live Google News Financial Sentiment Analyzer")  # [CHG-26]
    n1, n2 = st.columns([3, 1])
    query_sym = n1.text_input("Enter Stock, Sector, or Index Name:", value="Nifty 50", key="ti_nlp_query_in")
    mk = n2.selectbox("Market:", ["IN", "US"], key="sb_nlp_market")
    if st.button("Analyze News Polarity", key="btn_analyze_nlp_in"):
        with st.spinner("Parsing live RSS headlines..."):
            res = fetch_live_news_sentiment(query_sym, market=mk)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Market Headline Bias", res["label"])
            sc2.metric("Recency-weighted score (-1 to +1)", res["score"])
            sc3.metric("Headlines (last 7 days)", res["count"])
            for it in res["items"]:
                st.write(f"- {it['headline']}  `({it['score']:+.2f}, {it['age_days']}d)`")

# ##############################################################################
# SECTION 6: US EQUITIES SWING TRADING DESK
# ##############################################################################
elif app_mode == NAV[5]:
    st.title("🗽 US Equities Multi-Indicator Swing Desk")
    st.caption("Same scoring engine as the NSE desk, benchmarked to the S&P 500, with earnings-date gap-risk checks.")
    us_tab_scan, us_tab_single, us_tab_bt, us_tab_sz = st.tabs(["🚀 US Market Scanner", "🔍 US Stock Deep Dive & Sentiment",
                                                                "📊 Backtester", "🧮 Position Sizer"])
    with us_tab_scan:
        st.subheader("Automated S&P 500 / Nasdaq Swing Scanner")
        render_scanner("US")
    with us_tab_single:
        st.subheader("US Stock Deep Dive & Wall Street Sentiment")
        render_deep_dive("US")
    with us_tab_bt:
        st.subheader("Strategy Backtest (same rules as the scanner)")
        render_backtester("US")
    with us_tab_sz:
        render_sizer("US")

# ##############################################################################
# SECTION 7: ACTIVE US SWING PORTFOLIO & EXIT MONITOR
# ##############################################################################
elif app_mode == NAV[6]:
    st.title("🛡️ Active US Swing Portfolio & Exit Monitor (Wall Street)")
    st.caption("Initial ATR stop → breakeven at +1R → 3×ATR chandelier trailing stop; trend-weakening and time-stop alerts.")
    render_equity_monitor("US")

# ##############################################################################
# SECTION 8: TRADE JOURNAL & PERFORMANCE  [CHG-32]
# ##############################################################################
elif app_mode == NAV[7]:
    st.title("📓 Trade Journal & Performance Analytics")
    st.caption("Every closed trade is logged here. Expectancy by setup tells you which signals actually pay you.")
    render_backup_restore("journal", JOURNAL_FILE, JOURNAL_COLS, "jr")
    j = st.session_state.journal.copy()
    if j.empty:
        st.info("No closed trades yet. Close positions from Sections 2, 4 or 7 with 'Close & log to Journal'.")
    else:
        for col in ["Entry", "Exit", "Qty", "PnL", "PnL_Pct", "R_Multiple"]:
            j[col] = pd.to_numeric(j[col], errors="coerce")
        mk = st.selectbox("Market:", sorted(j["Market"].dropna().unique().tolist()), key="sb_journal_mkt")
        jm = j[j["Market"] == mk].copy()
        wins, losses = jm[jm["PnL"] > 0], jm[jm["PnL"] <= 0]
        ccy = "$" if mk == "US" else "₹"
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Trades", len(jm))
        k2.metric("Win rate", f"{len(wins) / len(jm) * 100:.1f}%" if len(jm) else "n/a")
        k3.metric("Avg win / loss", f"{ccy}{wins['PnL'].mean():,.0f} / {ccy}{losses['PnL'].mean():,.0f}" if len(wins) and len(losses) else "n/a")
        pf = wins["PnL"].sum() / abs(losses["PnL"].sum()) if len(losses) and losses["PnL"].sum() != 0 else None
        k4.metric("Profit factor", f"{pf:.2f}" if pf else "n/a")
        k5.metric("Expectancy (R)", f"{jm['R_Multiple'].mean():.2f}" if jm["R_Multiple"].notna().any() else "n/a")
        k6.metric("Net P&L", f"{ccy}{jm['PnL'].sum():,.0f}")
        jm["Exit_Date"] = pd.to_datetime(jm["Exit_Date"], errors="coerce")
        curve = jm.sort_values("Exit_Date").set_index("Exit_Date")["PnL"].cumsum()
        if len(curve) > 1:
            st.line_chart(curve.rename("Cumulative P&L"))
        g1, g2 = st.columns(2)
        g1.markdown("**By setup**")
        g1.dataframe(jm.groupby("Setup").agg(Trades=("PnL", "size"), Net=("PnL", "sum"), Avg_R=("R_Multiple", "mean")).round(2))
        g2.markdown("**By exit reason**")
        g2.dataframe(jm.groupby("Exit_Reason").agg(Trades=("PnL", "size"), Net=("PnL", "sum")).round(2))
        show_df(jm.sort_values("Exit_Date", ascending=False))
