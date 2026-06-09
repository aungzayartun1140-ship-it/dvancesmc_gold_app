import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom module (No.2)
from advanced_features import (
    detect_bos_choch, detect_liquidity_sweep,
    calculate_premium_discount, get_mtf_trend,
    calculate_fib_levels, calculate_indicators_basic
)

# --- Page config ---
st.set_page_config(page_title="EagleEye XAUUSD Pro", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0c0d14; color: #d1d4dc; }
.stTabs [data-baseweb="tab"] { color: #848e9c; font-size: 15px; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #f0b90b !important; border-bottom-color: #f0b90b !important; }
.css-1r6g72q { background-color: #131722; border-radius: 8px; padding: 15px; }
</style>
""", unsafe_allow_html=True)

# --- Data fetching (cached) ---
@st.cache_data(ttl=300)
def fetch_data(symbol, period, interval):
    # Map XAUUSD to Yahoo Finance symbol
    if symbol.upper() == "XAUUSD":
        yf_symbol = "GC=F"
    else:
        yf_symbol = symbol
    try:
        df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Download error: {e}")
        return None

def prepare_df(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    date_col = 'Date' if 'Date' in df.columns else 'Datetime'
    df.rename(columns={date_col: 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date_str'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    df.columns = [c.lower() for c in df.columns]
    numeric = ['open','high','low','close','volume']
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['open','high','low','close'], inplace=True)
    return df

# --- Main app ---
def main():
    st.title("🦅 EAGLEEYE XAUUSD PRO DASHBOARD")
    st.caption("BOS • CHoCH • FVG • Order Block • Liquidity Sweep • Premium/Discount • MTF • Auto RR")

    # Sidebar settings
    st.sidebar.header("⚙️ Control Panel")
    asset = st.sidebar.selectbox("Asset", ["XAUUSD (Gold)"], index=0)  # Only XAUUSD
    timeframe = st.sidebar.selectbox("Base Timeframe", ["15m", "1h", "4h", "1d"], index=1)
    period = st.sidebar.selectbox("Data Range", ["5d", "1mo", "3mo", "6mo"], index=1)
    refresh = st.sidebar.button("🔄 Refresh Data")

    if refresh:
        st.cache_data.clear()

    # Fetch base timeframe data
    df_base = fetch_data("XAUUSD", period, timeframe)
    if df_base is None or len(df_base) < 30:
        st.error("ဒေတာမလုံလောက်ပါ။ အင်တာနက်အဆက်အသွယ်စစ်ဆေးပါ။")
        return

    df_base = prepare_df(df_base)
    # Calculate basic indicators
    df_base = calculate_indicators_basic(df_base)

    # Multi Timeframe (higher timeframe = 4x base)
    if timeframe == "15m":
        ht_interval = "1h"
    elif timeframe == "1h":
        ht_interval = "4h"
    elif timeframe == "4h":
        ht_interval = "1d"
    else:
        ht_interval = "1wk"
    df_ht = fetch_data("XAUUSD", period, ht_interval)
    if df_ht is not None:
        df_ht = prepare_df(df_ht)
        ht_trend = get_mtf_trend(df_ht)
    else:
        ht_trend = "neutral"

    # Advanced feature calculations
    df_base = detect_bos_choch(df_base)                     # BOS & CHoCH
    df_base = detect_liquidity_sweep(df_base)               # Liquidity Sweep
    df_base = calculate_premium_discount(df_base)           # Premium/Discount
    fib_50, fib_618 = calculate_fib_levels(df_base)         # Fibonacci Auto

    # FVG & Order Block (already in separate functions, reused)
    fvg_list, ob_list = analyze_smc(df_base)                # from original code
    df_base['pattern'] = detect_all_patterns(df_base)       # candlestick patterns

    # Generate signal with all features
    signals = generate_composite_signal(df_base, ob_list, df_base['pattern'].tolist(),
                                         ht_trend, fib_50, fib_618)

    # --- UI Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Pro Chart & Signals", "🎯 Strategy Matrix", "📅 Economic Calendar", "📋 Dashboard Setup"])

    with tab1:
        # Plot candlestick chart with BOS/CHoCH markers
        fig = create_chart_with_features(df_base, fvg_list, ob_list)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎯 Live Signal (Auto Risk Reward)")
        if signals:
            latest = signals[-1]
            st.success(f"**{latest['type']}**  |  Entry: ${latest['entry']}  |  SL: ${latest['sl']}  |  TP: ${latest['tp']}  |  RR: {latest['rr']}")
            st.info(f"**Reason:** {latest['reason']}")
        else:
            st.info("စနစ်နှင့်ကိုက်ညီသော Signal မရှိသေးပါ။")

        # Quick stats
        col1, col2, col3 = st.columns(3)
        col1.metric("HTF Trend", ht_trend.upper())
        col2.metric("Premium/Discount", df_base['zone'].iloc[-1] if 'zone' in df_base else "N/A")
        col3.metric("Last BOS", df_base['bos'].iloc[-1] if 'bos' in df_base else "None")

    with tab2:
        st.subheader("Advanced Features Dictionary")
        features = {
            "BOS (Break of Structure)": "Swing high/low ကိုဖြတ်ကျော်ပြီး trend ဆက်ခြင်း",
            "CHoCH (Change of Character)": "Trend direction ပြောင်းလဲခြင်း အချက်ပြ",
            "Liquidity Sweep": "Stop loss hunting ဖြစ်ပွားပြီး ပြန်ပြောင်းခြင်း",
            "Premium / Discount": "Fibonacci 61.8% အထက် Premium, 38.2% အောက် Discount",
            "Multi Timeframe": "Higher timeframe trend filter",
            "Auto RR": "Risk 1% : Reward 2% (fixed ratio 1:2)"
        }
        for k, v in features.items():
            st.markdown(f"**{k}:** {v}")

    with tab3:
        st.subheader("Economic News Impact (Gold Focus)")
        st.table(pd.DataFrame([
            {"Event": "CPI", "Impact": "High", "Gold Up": "CPI < Forecast", "Gold Down": "CPI > Forecast"},
            {"Event": "NFP", "Impact": "High", "Gold Up": "Jobs < Forecast", "Gold Down": "Jobs > Forecast"},
            {"Event": "FOMC", "Impact": "Critical", "Gold Up": "Dovish", "Gold Down": "Hawkish"}
        ]))

    with tab4:
        st.subheader("Live Technical Snapshot")
        last = df_base.iloc[-1]
        st.write(f"**RSI (14):** {last.get('rsi', 'N/A')}")
        st.write(f"**MACD:** {last.get('macd', 'N/A')} | Signal: {last.get('macd_signal', 'N/A')}")
        st.write(f"**Fibonacci 50%:** ${fib_50} | **61.8%:** ${fib_618}")
        st.write(f"**Current Zone:** {last.get('zone', 'N/A')}")
        st.write(f"**Recent Liquidity Sweep:** {'Yes' if last.get('liq_sweep', False) else 'No'}")

if __name__ == "__main__":
    main()
