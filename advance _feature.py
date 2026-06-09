import pandas as pd
import numpy as np

def calculate_indicators_basic(df):
    """RSI, MACD, Moving Averages"""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    # EMAs
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    return df

def detect_bos_choch(df, lookback=5):
    """Break of Structure (BOS) and Change of Character (CHoCH)"""
    df['swing_high'] = df['high'].rolling(window=lookback, center=True).max()
    df['swing_low'] = df['low'].rolling(window=lookback, center=True).min()
    df['bos'] = "none"
    df['choch'] = False

    for i in range(lookback+1, len(df)-lookback):
        # BOS bullish: price breaks above previous swing high
        if df['high'].iloc[i] > df['swing_high'].iloc[i-1] and df['close'].iloc[i] > df['open'].iloc[i]:
            df.loc[df.index[i], 'bos'] = "bullish_bos"
        # BOS bearish: price breaks below previous swing low
        elif df['low'].iloc[i] < df['swing_low'].iloc[i-1] and df['close'].iloc[i] < df['open'].iloc[i]:
            df.loc[df.index[i], 'bos'] = "bearish_bos"

    # CHoCH: after a BOS, price reverses and breaks opposite swing point
    for i in range(2, len(df)-2):
        if df['bos'].iloc[i-1] == "bullish_bos" and df['low'].iloc[i] < df['swing_low'].iloc[i-2]:
            df.loc[df.index[i], 'choch'] = True
        elif df['bos'].iloc[i-1] == "bearish_bos" and df['high'].iloc[i] > df['swing_high'].iloc[i-2]:
            df.loc[df.index[i], 'choch'] = True
    return df

def detect_liquidity_sweep(df, threshold=1.002):
    """Detect when price spikes above recent high or below recent low then closes back"""
    df['liq_sweep'] = False
    lookback = 10
    for i in range(lookback, len(df)-1):
        recent_high = df['high'].iloc[i-lookback:i].max()
        recent_low = df['low'].iloc[i-lookback:i].min()
        # Sweep above high (bull trap)
        if df['high'].iloc[i] > recent_high * threshold and df['close'].iloc[i] < df['open'].iloc[i]:
            df.loc[df.index[i], 'liq_sweep'] = "bearish_sweep"
        # Sweep below low (bear trap)
        elif df['low'].iloc[i] < recent_low * (2 - threshold) and df['close'].iloc[i] > df['open'].iloc[i]:
            df.loc[df.index[i], 'liq_sweep'] = "bullish_sweep"
    return df

def calculate_premium_discount(df):
    """Premium = above 61.8% of range, Discount = below 38.2%"""
    high_52 = df['high'].rolling(50).max()
    low_52 = df['low'].rolling(50).min()
    range_52 = high_52 - low_52
    df['zone'] = "neutral"
    for i in range(50, len(df)):
        pos = (df['close'].iloc[i] - low_52.iloc[i]) / (range_52.iloc[i] + 1e-10)
        if pos > 0.618:
            df.loc[df.index[i], 'zone'] = "Premium (Overbought)"
        elif pos < 0.382:
            df.loc[df.index[i], 'zone'] = "Discount (Oversold)"
        else:
            df.loc[df.index[i], 'zone'] = "Fair Value"
    return df

def calculate_fib_levels(df):
    """Auto Fibonacci using recent swing high/low"""
    high_all = df['high'].max()
    low_all = df['low'].min()
    diff = high_all - low_all
    fib_50 = high_all - diff * 0.5
    fib_618 = high_all - diff * 0.618
    return round(fib_50, 2), round(fib_618, 2)

def get_mtf_trend(df_ht):
    """Higher timeframe trend: based on EMA20 and price position"""
    if len(df_ht) < 20:
        return "neutral"
    last_close = df_ht['close'].iloc[-1]
    ema20 = df_ht['close'].ewm(span=20).mean().iloc[-1]
    if last_close > ema20 * 1.005:
        return "bullish"
    elif last_close < ema20 * 0.995:
        return "bearish"
    else:
        return "neutral"

# --- Reuse original SMC functions (FVG, OB) ---
def analyze_smc(df):
    fvg = [None] * len(df)
    ob = [None] * len(df)
    for i in range(2, len(df)):
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            fvg[i] = ('bullish', df['high'].iloc[i-2], df['low'].iloc[i])
        elif df['high'].iloc[i] < df['low'].iloc[i-2]:
            fvg[i] = ('bearish', df['low'].iloc[i-2], df['high'].iloc[i])
    for i in range(5, len(df)-5):
        if df['low'].iloc[i] < df['low'].iloc[i-5:i].min():
            ob[i] = ('bullish', df['low'].iloc[i-1], df['high'].iloc[i-1], i-1)
        elif df['high'].iloc[i] > df['high'].iloc[i-5:i].max():
            ob[i] = ('bearish', df['low'].iloc[i-1], df['high'].iloc[i-1], i-1)
    return fvg, ob

def detect_all_patterns(df):
    patterns = [None] * len(df)
    for i in range(2, len(df)):
        O, H, L, C = df['open'].iloc[i], df['high'].iloc[i], df['low'].iloc[i], df['close'].iloc[i]
        O1, H1, L1, C1 = df['open'].iloc[i-1], df['high'].iloc[i-1], df['low'].iloc[i-1], df['close'].iloc[i-1]
        body = abs(C-O)
        total_range = H-L if H-L>0 else 0.001
        if H<H1 and L>L1:
            patterns[i] = "Inside Bar ⬜"
        elif body <= total_range*0.1:
            patterns[i] = "Doji ⏳"
        elif (O-L if C>=O else C-L) > body*2 and (H-C if C>=O else H-O) < body*0.5:
            patterns[i] = "Hammer/PinBar 🔨"
        elif (H-C if C>=O else H-O) > body*2 and (O-L if C>=O else C-L) < body*0.5:
            patterns[i] = "Shooting Star 💫"
        elif C1<O1 and C>O and C>=O1 and O<=C1:
            patterns[i] = "Bullish Engulfing 🟢"
        elif C1>O1 and C<O and C<=O1 and O>=C1:
            patterns[i] = "Bearish Engulfing 🔴"
    return patterns

def generate_composite_signal(df, ob_list, patterns, ht_trend, fib_50, fib_618):
    signals = []
    for i in range(10, len(df)):
        # Gather conditions
        bos_bull = df['bos'].iloc[i] == "bullish_bos" if 'bos' in df else False
        bos_bear = df['bos'].iloc[i] == "bearish_bos" if 'bos' in df else False
        choch = df['choch'].iloc[i] if 'choch' in df else False
        liq_sweep_bull = df['liq_sweep'].iloc[i] == "bullish_sweep" if 'liq_sweep' in df else False
        liq_sweep_bear = df['liq_sweep'].iloc[i] == "bearish_sweep" if 'liq_sweep' in df else False
        zone = df['zone'].iloc[i] if 'zone' in df else "neutral"
        ob_bull = ob_list[i] and ob_list[i][0]=='bullish'
        ob_bear = ob_list[i] and ob_list[i][0]=='bearish'
        pattern_bull = patterns[i] in ["Hammer/PinBar 🔨", "Bullish Engulfing 🟢"]
        pattern_bear = patterns[i] in ["Bearish Engulfing 🔴", "Shooting Star 💫"]
        rsi = df['rsi'].iloc[i] if 'rsi' in df else 50
        macd_bull = df['macd'].iloc[i] > df['macd_signal'].iloc[i] if 'macd' in df else False

        # Buy signal logic
        if ht_trend != "bearish" and (bos_bull or liq_sweep_bull or ob_bull) and pattern_bull and (zone == "Discount" or rsi < 40):
            entry = df['close'].iloc[i]
            sl = df['low'].iloc[i-2:i].min()  # recent low
            tp = entry + 2 * (entry - sl)     # RR 1:2
            signals.append({
                'time': df['date_str'].iloc[i],
                'type': 'BUY 🟢',
                'entry': round(entry,2),
                'sl': round(sl,2),
                'tp': round(tp,2),
                'rr': "1:2",
                'reason': f"BOS+{patterns[i]}+Discount Zone+RSI={round(rsi,1)} + HTF {ht_trend}"
            })
        # Sell signal logic
        elif ht_trend != "bullish" and (bos_bear or liq_sweep_bear or ob_bear) and pattern_bear and (zone == "Premium" or rsi > 60):
            entry = df['close'].iloc[i]
            sl = df['high'].iloc[i-2:i].max()
            tp = entry - 2 * (sl - entry)
            signals.append({
                'time': df['date_str'].iloc[i],
                'type': 'SELL 🔴',
                'entry': round(entry,2),
                'sl': round(sl,2),
                'tp': round(tp,2),
                'rr': "1:2",
                'reason': f"CHoCH/LiqSweep+{patterns[i]}+Premium Zone+RSI={round(rsi,1)} + HTF {ht_trend}"
            })
    return signals

# Chart creation function (reused from original with BOS/CHoCH markers)
def create_chart_with_features(df, fvg_list, ob_list):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['date_str'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="XAUUSD", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ))
    # Add FVG and OB zones (same as before)
    for i in range(len(df)):
        if fvg_list[i]:
            color = 'rgba(38,166,154,0.12)' if fvg_list[i][0]=='bullish' else 'rgba(239,83,80,0.12)'
            fig.add_shape(type="rect", x0=df['date_str'].iloc[i], x1=df['date_str'].iloc[min(i+6, len(df)-1)],
                          y0=fvg_list[i][1], y1=fvg_list[i][2], fillcolor=color, line_width=0)
        if ob_list[i]:
            color = 'rgba(41,98,255,0.15)' if ob_list[i][0]=='bullish' else 'rgba(255,109,0,0.15)'
            fig.add_shape(type="rect", x0=df['date_str'].iloc[ob_list[i][3]], x1=df['date_str'].iloc[i],
                          y0=ob_list[i][1], y1=ob_list[i][2], fillcolor=color, line_width=1)
    # Add BOS markers
    bos_idx = df[df['bos']!="none"].index
    for idx in bos_idx:
        y_pos = df['high'].iloc[idx] if df['bos'].iloc[idx]=="bullish_bos" else df['low'].iloc[idx]
        symbol = "▲" if df['bos'].iloc[idx]=="bullish_bos" else "▼"
        fig.add_annotation(x=df['date_str'].iloc[idx], y=y_pos, text=symbol, showarrow=False,
                           font=dict(size=16, color="yellow"))
    return fig
