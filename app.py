import os
import time
import asyncio
import logging
import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple, List
import uuid
import concurrent.futures
from collections import Counter
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from flask import Flask
import requests
import csv
import sys
import unittest
app = Flask(__name__)
CONFIG = {
    "SYMBOLS": [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "BCH/USDT"
    ],
    "TIMEFRAMES": {"5m": 350, "15m": 350, "1h": 300, "4h": 250},
    "SCAN_INTERVAL": 12,
    "LOCKED_INTERVAL_MIN": 120,
    "MIN_SCORE": 80,
    "MIN_RR": 2.5,
    "RISK_PCT": 0.01,
    "LEVERAGE": 5,
    "MAX_OPEN": 1,
    "RISK_MODEL": {
        "SL_ATR_MULT": 1.0,
        "TP_PCT_1": 0.01,
        "TP_PCT_2": 0.02,
    },
    "MOMENTUM_RISK_PCT_MULT": 0.55,
    "MOMENTUM_TP_PCT_1": 0.007,
    "MOMENTUM_TP_PCT_2": 0.011,
    "MAX_SIGNALS_PER_DAY": 2,
    "TRADE_SESSIONS_UTC": [
        {"name": "ASIA", "start": "00:00", "end": "07:00"},
        {"name": "LONDON", "start": "07:00", "end": "11:00"},
        {"name": "NEWYORK", "start": "13:00", "end": "17:00"},
    ],
    "VOL_REGIME_MIN": 0.0015,
    "REGIME_THRESHOLD": 0.003,
    "RECENT_MOVE_PCT": 0.01,
    "RECENT_MOVE_CANDLES": 3,
    "ORDER_BUFFER_SEC": 90,
    "TELEGRAM_TOKEN": "8441346951:AAGRjh5GQaResRakjmdre3iVPvXYdoqEP5g",
    "TELEGRAM_CHAT_ID": "8557187571",
    "API_KEY": None,
    "API_SECRET": None,
    "PRICE_DECIMALS": {
        "BTC/USDT": 1, "ETH/USDT": 2, "BNB/USDT": 2, "SOL/USDT": 3,
        "XRP/USDT": 4, "ADA/USDT": 4, "AVAX/USDT": 2, "DOT/USDT": 3,
        "LINK/USDT": 3, "BCH/USDT": 2
    },
    "LOG_FILE": "signals_log.csv",
    "DIAG_INTERVAL_SEC": 3600,
    "MAX_RETRY": 3,
    "BACKOFF_SEC": 5,
    "LATENCY_BUDGET_SEC": 12,
    "STALE_DATA_MIN_PER_TF": {"5m": 10, "15m": 30, "1h": 120, "4h": 480},
    "CIRCUIT_BREAK_ERRORS": 5,
    "SIGNAL_EXPIRE_HOURS": 4,
    "VALIDITY_MIN": 60,
    "ENTRY_ZONE_BUFFER_PCT": 0.002,
    "VWAP_DEVIATION_MULT": 2.0,
    "SLOPE_THRESHOLD": 0.0005,
    "ATR_CHANGE_THRESHOLD": 0.1,
    "SNIPER_PROXIMITY_MULT": 0.6,
    "PULLBACK_PROXIMITY_MULT": 0.3,
    "RELAXATION_HOURS": 3,
    "RELAXATION_PCT": 0.9,
    "MOMENTUM_MIN_STRENGTH_MULT": 0.6,
    "MOMENTUM_MAX_EXTENSION_MULT": 2.2,
    "MOMENTUM_MICRO_PULLBACK_MULT": 0.35,
}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def json_log(level, msg, extra=None):
    log_dict = {"timestamp": datetime.now(timezone.utc).isoformat(), "level": level, "message": msg}
    if extra:
        log_dict.update(extra)
    logger.log(getattr(logging, level.upper()), json.dumps(log_dict))
exchange = ccxt_async.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
STATE = {
    "signals_today": 0,
    "last_reset": datetime.now(timezone.utc).date(),
    "active_signals": [],
    "locked_symbols": {},
    "last_diag": datetime.now(timezone.utc),
    "error_count": 0,
    "btc_mode": None,
    "last_btc_check": None,
    "last_signal_time": None,
}
@dataclass
class Signal:
    symbol: str
    direction: str
    entry_zone: str
    sl: float
    tp1: float
    tp2: float
    rr: float
    score: int
    reason: str
    tf_alignment: str
    confidence: str
    validity_min: int
    creation_time: datetime
    is_momentum: bool = False
    size: Optional[float] = None
async def retry_fetch(func, *args, **kwargs):
    for attempt in range(CONFIG["MAX_RETRY"]):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            json_log("warning", f"Retry {attempt+1}/{CONFIG['MAX_RETRY']} for {func.__name__}", {"error": str(e)})
            await asyncio.sleep(CONFIG["BACKOFF_SEC"] * (2 ** attempt))
    raise Exception(f"Failed after {CONFIG['MAX_RETRY']} retries")
async def load_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    raw = await retry_fetch(exchange.fetch_ohlcv, symbol, timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.iloc[:-1].reset_index(drop=True)
    if len(df) < 100:
        return pd.DataFrame()
    return df
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def atr(df, n=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(n).mean()
def rsi(df, n=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(n).mean()
    loss = -delta.where(delta < 0, 0).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
def calculate_vwap(df):
    typical = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap
def in_trade_session(now_utc: datetime) -> bool:
    current_time = now_utc.time()
    for sess in CONFIG["TRADE_SESSIONS_UTC"]:
        start = datetime.strptime(sess["start"], "%H:%M").time()
        end = datetime.strptime(sess["end"], "%H:%M").time()
        if start <= current_time <= end:
            return True
    return False
def get_session(now_utc: datetime) -> str:
    current_time = now_utc.time()
    for sess in CONFIG["TRADE_SESSIONS_UTC"]:
        start = datetime.strptime(sess["start"], "%H:%M").time()
        end = datetime.strptime(sess["end"], "%H:%M").time()
        if start <= current_time <= end:
            return sess["name"]
    return "OTHER"
def get_bias(df: pd.DataFrame, tf: str) -> str:
    lengths = {'5m': 100, '15m': 100, '1h': 200, '4h': 200}
    if len(df) < lengths.get(tf, 100):
        return "NEUTRAL"
    ema200 = ema(df['close'], 200)
    price = df['close'].iloc[-1]
    slope = ema200.pct_change().tail(10).mean()
    if price > ema200.iloc[-1] and slope > CONFIG["SLOPE_THRESHOLD"]:
        return "BULL"
    elif price < ema200.iloc[-1] and slope < -CONFIG["SLOPE_THRESHOLD"]:
        return "BEAR"
    return "NEUTRAL"
def get_structure(df: pd.DataFrame, tf: str) -> str:
    order = 3 if tf in ['5m', '15m'] else 5
    lows_idx = argrelextrema(df['low'].values, np.less_equal, order=order)[0]
    highs_idx = argrelextrema(df['high'].values, np.greater_equal, order=order)[0]
    if len(lows_idx) < 2 or len(highs_idx) < 2:
        return "UNKNOWN"
    last_ll = df['low'].iloc[lows_idx[-1]]
    prev_ll = df['low'].iloc[lows_idx[-2]]
    last_hh = df['high'].iloc[highs_idx[-1]]
    prev_hh = df['high'].iloc[highs_idx[-2]]
    if last_ll > prev_ll and last_hh > prev_hh:
        return "INTACT_BULL"
    elif last_ll < prev_ll and last_hh < prev_hh:
        return "INTACT_BEAR"
    return "BROKEN"
def get_atr_regime(atr_series: pd.Series) -> str:
    atr_pct_change = atr_series.pct_change().tail(5).mean()
    if atr_pct_change > CONFIG["ATR_CHANGE_THRESHOLD"]:
        return "EXPANDING"
    elif atr_pct_change < -CONFIG["ATR_CHANGE_THRESHOLD"]:
        return "COLLAPSING"
    return "FLAT"
def detect_divergence(df: pd.DataFrame, rsi_series: pd.Series, direction: str) -> bool:
    order = 3
    if direction == "LONG":
        lows_idx = argrelextrema(df['low'].values, np.less, order=order)[0]
        if len(lows_idx) >= 2:
            last = lows_idx[-1]
            prev = lows_idx[-2]
            rsi_lows = rsi_series.iloc[lows_idx]
            if df['low'].iloc[last] < df['low'].iloc[prev] and rsi_lows.iloc[-1] > rsi_lows.iloc[-2]:
                return True
    elif direction == "SHORT":
        highs_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]
        if len(highs_idx) >= 2:
            last = highs_idx[-1]
            prev = highs_idx[-2]
            rsi_highs = rsi_series.iloc[highs_idx]
            if df['high'].iloc[last] > df['high'].iloc[prev] and rsi_highs.iloc[-1] < rsi_highs.iloc[-2]:
                return True
    return False
def determine_market_mode(dfs: Dict[str, pd.DataFrame]) -> str:
    d1h = dfs['1h']
    d15 = dfs['15m']
    atr1h = atr(d1h).iloc[-1]
    atr_regime = get_atr_regime(atr(d1h))
    price = d1h['close'].iloc[-1]
    vol = atr1h / price
    if vol < CONFIG["VOL_REGIME_MIN"]:
        return "NO-TRADE"
    bias1h = get_bias(d1h, '1h')
    bias15 = get_bias(d15, '15m')
    structure1h = get_structure(d1h, '1h')
    structure15 = get_structure(d15, '15m')
    ema200_1h = ema(d1h['close'], 200)
    slope = ema200_1h.pct_change().tail(10).mean()
    vwap = calculate_vwap(d1h).iloc[-1]
    if structure1h == "BROKEN" or structure15 == "BROKEN":
        return "TRANSITION"
    if atr_regime == "COLLAPSING" and abs(price - vwap) / price < 0.005 and structure1h in ["BROKEN", "UNKNOWN"]:
        return "NO-TRADE"
    if abs(slope) > CONFIG["SLOPE_THRESHOLD"] and structure1h.startswith("INTACT") and atr_regime != "COLLAPSING":
        return "TREND"
    elif abs(slope) < CONFIG["SLOPE_THRESHOLD"] and abs(price - vwap) / price < 0.005 and atr_regime == "FLAT":
        return "RANGE"
    return "TRANSITION"
def detect_fvg(df, atr_series):
    fvg_list = []
    for i in range(2, len(df)):
        first_high = df['high'].iloc[i-2]
        first_low = df['low'].iloc[i-2]
        middle_open = df['open'].iloc[i-1]
        middle_close = df['close'].iloc[i-1]
        middle_high = df['high'].iloc[i-1]
        middle_low = df['low'].iloc[i-1]
        third_low = df['low'].iloc[i]
        third_high = df['high'].iloc[i]
        middle_range = middle_high - middle_low
        atr_val = atr_series.iloc[i-1]
        if middle_range <= 1.5 * atr_val:
            continue
        prev_hh = df['high'].iloc[:i-1].rolling(20).max().iloc[-1]
        prev_ll = df['low'].iloc[:i-1].rolling(20).min().iloc[-1]
        break_bull = middle_high > prev_hh
        break_bear = middle_low < prev_ll
        if third_low > first_high and (break_bull or break_bear):
            fvg_list.append(('bullish', first_high, third_low, i))
        elif third_high < first_low and (break_bull or break_bear):
            fvg_list.append(('bearish', third_high, first_low, i))
        else:
            fvg_list.append(None)
    return fvg_list
def find_fvg(dfs: Dict[str, pd.DataFrame], bias: str) -> Tuple[Optional[float], Optional[float]]:
    d15 = dfs['15m']
    atr15 = atr(d15)
    fvg_list = detect_fvg(d15, atr15)
    fvg_up_mid = None
    fvg_dn_mid = None
    for idx in range(len(fvg_list)-1, -1, -1):
        f = fvg_list[idx]
        if f is not None:
            type_, bottom, top, i = f
            mid = (bottom + top) / 2
            subsequent = d15.iloc[i+1:]
            if type_ == 'bullish' and bias == "BULL" and (len(subsequent) == 0 or subsequent['low'].min() >= bottom):
                fvg_up_mid = mid
                break
            elif type_ == 'bearish' and bias == "BEAR" and (len(subsequent) == 0 or subsequent['high'].max() <= top):
                fvg_dn_mid = mid
                break
    return fvg_up_mid, fvg_dn_mid
def find_order_blocks(dfs: Dict[str, pd.DataFrame]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    d15 = dfs['15m']
    order = 3
    swing_lows = argrelextrema(d15['low'].values, np.less, order=order)[0]
    swing_highs = argrelextrema(d15['high'].values, np.greater, order=order)[0]
    bull_ob = None
    bear_ob = None
    if len(swing_lows) > 1:
        last_low_idx = swing_lows[-1]
        if d15['low'].iloc[last_low_idx+1:].min() >= d15['low'].iloc[last_low_idx]:
            bull_ob = (d15['low'].iloc[last_low_idx], d15['high'].iloc[last_low_idx])
    if len(swing_highs) > 1:
        last_high_idx = swing_highs[-1]
        if d15['high'].iloc[last_high_idx+1:].max() <= d15['high'].iloc[last_high_idx]:
            bear_ob = (d15['low'].iloc[last_high_idx], d15['high'].iloc[last_high_idx])
    return bull_ob, bear_ob
def sweep_liquidity(dfs: Dict[str, pd.DataFrame], direction: str) -> bool:
    d5 = dfs.get('5m')
    if d5 is None or d5.empty:
        return False
    prev_hh = d5['high'].rolling(20).max().shift(1).iloc[-1]
    prev_ll = d5['low'].rolling(20).min().shift(1).iloc[-1]
    recent_highs = d5['high'].iloc[-3:]
    recent_lows = d5['low'].iloc[-3:]
    recent_close = d5['close'].iloc[-1]
    if direction == "LONG":
        if min(recent_lows) < prev_ll and recent_close > prev_ll:
            return True
    elif direction == "SHORT":
        if max(recent_highs) > prev_hh and recent_close < prev_hh:
            return True
    return False
def calculate_sniper_score(symbol: str, dfs: Dict[str, pd.DataFrame], now: datetime) -> Tuple[Optional[Signal], Dict, str]:
    skip_reason = "Unknown"
    try:
        d5 = dfs['5m']
        d15 = dfs['15m']
        d1h = dfs['1h']
        price = d15['close'].iloc[-1]
        atr15 = atr(d15).iloc[-1]
        if pd.isna(atr15) or atr15 is None:
            return None, {}, "Invalid ATR"
        vol_regime = atr15 / price
        rsi_series_15 = rsi(d15)
        rsi15 = rsi_series_15.iloc[-1]
        mode = determine_market_mode(dfs)
        if mode == "NO-TRADE":
            return None, {}, f"Market mode: {mode}"
        if mode == "TRANSITION" and get_bias(d1h, '1h') == "NEUTRAL":
            return None, {}, "No clear bias in transition"
        recent_range = d15['high'].iloc[-1] - d15['low'].iloc[-1]
        news_impulse = recent_range > 3 * atr15
        tf_biases = {tf: get_bias(df, tf) for tf, df in dfs.items() if tf != '5m'}
        bias = tf_biases['1h']
        if mode == "TREND":
            if bias == "NEUTRAL" or tf_biases['15m'] != bias:
                return None, {}, "Bias conflict"
        btc_mode = STATE["btc_mode"] or "TREND"
        if symbol != "BTC/USDT" and btc_mode == "NO-TRADE":
            return None, {}, "BTC no-trade"
        if symbol != "BTC/USDT":
            if STATE.get("btc_structure", "") == "BROKEN" and STATE.get("btc_vol", 0) > CONFIG["REGIME_THRESHOLD"]:
                return None, {}, "BTC extreme instability"
            if STATE.get("btc_atr_regime", "") == "EXPANDING":
                if STATE.get("btc_bias", "NEUTRAL") == "BEAR" and bias == "LONG":
                    return None, {}, "BTC violent bear - no alt longs"
                if STATE.get("btc_bias", "NEUTRAL") == "BULL" and bias == "SHORT":
                    return None, {}, "BTC violent bull - no alt shorts"
        score_breakdown = {"htf": 30 if mode == "TREND" else 20, "structure": 0, "fvg/ob": 0, "sweep/div": 0, "vol": 0, "rsi": 0, "vwap": 0}
        direction = None
        bull_ob, bear_ob = find_order_blocks(dfs)
        fvg_up, fvg_dn = find_fvg(dfs, bias) if mode == "TREND" else (None, None)
        vwap15 = calculate_vwap(d15).iloc[-1]
        atr5 = atr(d5).iloc[-1]
        if pd.isna(atr5):
            atr5 = atr15
        entry_level = None
        intended_entry = None
        is_pullback = False
        is_range_extreme = False
        is_vwap_reversion = False
        setup_type = ""
        tp_pct_1 = CONFIG["RISK_MODEL"]["TP_PCT_1"]
        tp_pct_2 = CONFIG["RISK_MODEL"]["TP_PCT_2"]
        atr_regime_15 = get_atr_regime(atr(d15))
        if mode == "TREND":
            direction = "LONG" if bias == "BULL" else "SHORT"
            structure15 = get_structure(d15, '15m')
            if (bias == "BULL" and structure15 != "INTACT_BULL") or (bias == "BEAR" and structure15 != "INTACT_BEAR"):
                return None, {}, "Structure not intact"
            score_breakdown["structure"] = 20
            ema50_15 = ema(d15['close'], 50).iloc[-1]
            pullback_level = None
            has_fvg = False
            if bias == "BULL":
                if fvg_up:
                    pullback_level = fvg_up
                    has_fvg = True
                else:
                    pullback_level = ema50_15
                if pullback_level and abs(price - pullback_level) <= CONFIG["PULLBACK_PROXIMITY_MULT"] * atr15:
                    is_pullback = True
                    entry_level = pullback_level
                    intended_entry = pullback_level
                    score_breakdown["fvg/ob"] = 25 if has_fvg else 20
            else:
                if fvg_dn:
                    pullback_level = fvg_dn
                    has_fvg = True
                else:
                    pullback_level = ema50_15
                if pullback_level and abs(price - pullback_level) <= CONFIG["PULLBACK_PROXIMITY_MULT"] * atr15:
                    is_pullback = True
                    entry_level = pullback_level
                    intended_entry = pullback_level
                    score_breakdown["fvg/ob"] = 25 if has_fvg else 20
            if has_fvg:
                bull_ob = None if bias == "BULL" else bull_ob
                bear_ob = None if bias == "BEAR" else bear_ob
            if is_pullback:
                setup_type = "Trend Pullback"
        elif mode == "RANGE":
            prev_ll = d15['low'].rolling(20).min().shift(1).iloc[-1]
            prev_hh = d15['high'].rolling(20).max().shift(1).iloc[-1]
            is_sweep_long = sweep_liquidity(dfs, "LONG")
            is_sweep_short = sweep_liquidity(dfs, "SHORT")
            if is_sweep_long or is_sweep_short:
                is_range_extreme = True
                if is_sweep_long:
                    direction = "LONG"
                    entry_level = prev_ll
                    has_div = detect_divergence(d15, rsi_series_15, "LONG")
                    score_breakdown["sweep/div"] = 25 if has_div else 10
                else:
                    direction = "SHORT"
                    entry_level = prev_hh
                    has_div = detect_divergence(d15, rsi_series_15, "SHORT")
                    score_breakdown["sweep/div"] = 25 if has_div else 10
            prev_dist_to_vwap = abs(d15['close'].iloc[-10:-1] - vwap15) / d15['close'].iloc[-10:-1]
            max_prev_dist = prev_dist_to_vwap.max()
            deviated = max_prev_dist > CONFIG["VWAP_DEVIATION_MULT"] * (atr15 / price)
            prev_stretch_up = d15['close'].iloc[-10:-1].max() > vwap15
            if deviated and atr_regime_15 == "FLAT" and not (is_sweep_long or is_sweep_short):
                is_vwap_reversion = True
                score_breakdown["vwap"] = 25
                entry_level = vwap15
                direction = "LONG" if prev_stretch_up else "SHORT"
                tp_pct_1 = 0.01
                tp_pct_2 = 0.015
            if is_range_extreme:
                setup_type = "Range Extreme"
            elif is_vwap_reversion:
                setup_type = "VWAP Reversion"
        if not (is_pullback or is_range_extreme or is_vwap_reversion):
            return None, {}, "No valid sniper setup"
        if news_impulse and is_range_extreme:
            return None, {}, "News impulse"
        if abs(price - entry_level) > CONFIG["SNIPER_PROXIMITY_MULT"] * 1.6 * atr15:
            return None, {"intended_entry": intended_entry, "direction": direction}, "Too far from entry level"
        if d15['volume'].iloc[-1] > d15['volume'].rolling(20).mean().iloc[-1] * 1.5:
            score_breakdown["vol"] = 10
        if mode == "RANGE":
            if (direction == "LONG" and rsi15 < 40) or (direction == "SHORT" and rsi15 > 60):
                score_breakdown["rsi"] = 3
        session = get_session(now)
        session_bonus = 0
        if session == "ASIA":
            session_bonus += int(score_breakdown.get("sweep/div", 0) * 0.2)
            session_bonus += int(score_breakdown.get("htf", 0) * -0.2)
            if setup_type in ["Range Extreme", "VWAP Reversion"]:
                session_bonus += 10
            if atr_regime_15 != "EXPANDING":
                session_bonus -= 10
        elif session == "LONDON":
            if is_pullback:
                session_bonus += int(score_breakdown.get("structure", 0) * 0.2)
                session_bonus += int(score_breakdown.get("fvg/ob", 0) * 0.2)
            if is_vwap_reversion and atr_regime_15 != "FLAT":
                session_bonus -= 10
        elif session == "NEWYORK":
            session_bonus += int(score_breakdown.get("vol", 0) * 0.2)
            if atr_regime_15 == "EXPANDING":
                session_bonus += 10
        score = sum(score_breakdown.values()) + session_bonus
        if atr_regime_15 == "COLLAPSING":
            score -= 10
        if vol_regime < CONFIG["VOL_REGIME_MIN"] * 0.8:
            score -= 10
        min_score = CONFIG["MIN_SCORE"]
        if STATE["last_signal_time"] and now - STATE["last_signal_time"] > timedelta(hours=CONFIG["RELAXATION_HOURS"]):
            min_score *= CONFIG["RELAXATION_PCT"]
        score = int(score)
        if score < min_score:
            return None, {}, f"Low score: {score}"
        buffer = max(0.0015 * price, 0.2 * atr15)
        entry_low = entry_level - buffer
        entry_high = entry_level + buffer
        entry_zone = f"{entry_low:.4f} - {entry_high:.4f}"
        sl_mult = CONFIG["RISK_MODEL"]["SL_ATR_MULT"]
        if direction == "LONG":
            sl = bull_ob[0] - 0.1 * atr5 if bull_ob else entry_level - sl_mult * atr5
            tp1 = entry_level + tp_pct_1 * entry_level
            tp2 = entry_level + tp_pct_2 * entry_level
        else:
            sl = bear_ob[1] + 0.1 * atr5 if bear_ob else entry_level + sl_mult * atr5
            tp1 = entry_level - tp_pct_1 * entry_level
            tp2 = entry_level - tp_pct_2 * entry_level
        if is_vwap_reversion:
            atr_mult_tp1 = 1.0
            atr_mult_tp2 = 1.5
            if direction == "LONG":
                tp1 = vwap15 + atr_mult_tp1 * atr15
                tp2 = vwap15 + atr_mult_tp2 * atr15
            else:
                tp1 = vwap15 - atr_mult_tp1 * atr15
                tp2 = vwap15 - atr_mult_tp2 * atr15
        small_value = 1e-6 * price
        if abs(sl - entry_level) < small_value:
            return None, {}, "SL too close"
        rr = abs(tp2 - entry_level) / abs(sl - entry_level)
        if rr < CONFIG["MIN_RR"]:
            return None, {}, f"Low RR: {rr}"
        reason_map = {
            "htf": "HTF Bias alignment",
            "structure": "Structure intact",
            "fvg/ob": "FVG/OB confluence",
            "sweep/div": "Sweep/Divergence confirmation",
            "vol": "Volume surge",
            "rsi": "RSI momentum",
            "vwap": "VWAP deviation"
        }
        reasons = [reason_map[k] for k, v in score_breakdown.items() if v > 0][:2]
        reason_str = " - ".join(reasons)
        tf_align_str = "/".join(tf_biases[tf][0] for tf in ["15m", "1h", "4h"])
        confidence = "High" if score >= 90 else "Medium"
        creation_time = datetime.now(timezone.utc)
        sig = Signal(
            symbol=symbol,
            direction=direction,
            entry_zone=entry_zone,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            rr=rr,
            score=score,
            reason=f"{setup_type}: {reason_str}",
            tf_alignment=tf_align_str,
            confidence=confidence,
            validity_min=CONFIG["VALIDITY_MIN"],
            creation_time=creation_time,
            is_momentum=False
        )
        return sig, score_breakdown, ""
    except Exception as e:
        return None, {}, "Exception in sniper"
def calculate_momentum_score(symbol: str, dfs: Dict[str, pd.DataFrame], intended_entry: float, direction: str, now: datetime) -> Tuple[Optional[Signal], str]:
    d5 = dfs['5m']
    d15 = dfs['15m']
    price = d15['close'].iloc[-1]
    atr15 = atr(d15).iloc[-1]
    if pd.isna(atr15):
        return None, "Invalid ATR"
    vol_regime = atr15 / price
    if vol_regime < CONFIG["VOL_REGIME_MIN"]:
        return None, "Low volatility for momentum"
    mode = determine_market_mode(dfs)
    atr_regime_15 = get_atr_regime(atr(d15))
    if (mode not in ["TREND", "TRANSITION"] or atr_regime_15 != "EXPANDING") or (mode == "TRANSITION" and get_bias(dfs['1h'], '1h') == "NEUTRAL"):
        return None, "Not suitable for momentum"
    btc_mode = STATE["btc_mode"] or "TREND"
    if symbol != "BTC/USDT" and btc_mode == "NO-TRADE":
        return None, "BTC no-trade"
    if symbol != "BTC/USDT":
        if STATE.get("btc_structure", "") == "BROKEN" and STATE.get("btc_vol", 0) > CONFIG["REGIME_THRESHOLD"]:
            return None, "BTC extreme instability"
        if STATE.get("btc_atr_regime", "") == "EXPANDING":
            if STATE.get("btc_bias", "NEUTRAL") == "BEAR" and direction == "LONG":
                return None, "BTC violent bear - no alt longs"
            if STATE.get("btc_bias", "NEUTRAL") == "BULL" and direction == "SHORT":
                return None, "BTC violent bull - no alt shorts"
    session = get_session(now)
    if session == "ASIA" and atr_regime_15 != "EXPANDING":
        return None, "Asia low momentum"
    recent_range = d15['high'].iloc[-1] - d15['low'].iloc[-1]
    if recent_range > 3 * atr15:
        return None, "News impulse"
    distance_from_level = abs(price - intended_entry)
    if distance_from_level < CONFIG["MOMENTUM_MIN_STRENGTH_MULT"] * atr15:
        return None, "Not enough move away"
    if distance_from_level > CONFIG["MOMENTUM_MAX_EXTENSION_MULT"] * atr15:
        return None, "Overextended"
    rsi_series_15 = rsi(d15)
    rsi15 = rsi_series_15.iloc[-1]
    if (direction == "LONG" and rsi15 > 70) or (direction == "SHORT" and rsi15 < 30):
        return None, "Momentum exhaustion"
    quality_count = 0
    last_5m = d5.iloc[-1]
    body_ratio = abs(last_5m['close'] - last_5m['open']) / (last_5m['high'] - last_5m['low'] + 1e-8)
    if body_ratio >= 0.7:
        quality_count += 1
    if d15['volume'].iloc[-1] >= 1.8 * d15['volume'].rolling(20).mean().iloc[-1]:
        quality_count += 1
    prev_hh = d15['high'].rolling(20).max().shift(1).iloc[-1]
    prev_ll = d15['low'].rolling(20).min().shift(1).iloc[-1]
    if direction == "LONG" and price > prev_hh:
        quality_count += 1
    elif direction == "SHORT" and price < prev_ll:
        quality_count += 1
    vwap15 = calculate_vwap(d15).iloc[-1]
    if direction == "LONG" and price > vwap15 and (price - vwap15) / price > 0.005:
        quality_count += 1
    elif direction == "SHORT" and price < vwap15 and (vwap15 - price) / price > 0.005:
        quality_count += 1
    if quality_count < 2:
        return None, "Insufficient momentum quality"
    atr5 = atr(d5).iloc[-1]
    if pd.isna(atr5):
        atr5 = atr15
    recent_candles = CONFIG["RECENT_MOVE_CANDLES"]
    if direction == "LONG":
        recent_high = d5['high'].iloc[-recent_candles:].max()
        if (recent_high - price) / atr5 < 0.1:
            return None, "No pause in momentum"
    else:
        recent_low = d5['low'].iloc[-recent_candles:].min()
        if (price - recent_low) / atr5 < 0.1:
            return None, "No pause in momentum"
    micro_pullback = CONFIG["MOMENTUM_MICRO_PULLBACK_MULT"] * atr15
    if direction == "LONG":
        if price < intended_entry + micro_pullback:
            return None, "No micro pullback yet"
        entry_level = price - micro_pullback
    else:
        if price > intended_entry - micro_pullback:
            return None, "No micro pullback yet"
        entry_level = price + micro_pullback
    buffer = max(0.0015 * price, 0.2 * atr15)
    entry_low = entry_level - buffer
    entry_high = entry_level + buffer
    entry_zone = f"{entry_low:.4f} - {entry_high:.4f}"
    buffer_mult = 0.5 * CONFIG["SNIPER_PROXIMITY_MULT"]
    if direction == "LONG":
        sl = intended_entry - buffer_mult * atr15
        tp1 = entry_level * (1 + CONFIG["MOMENTUM_TP_PCT_1"])
        tp2 = entry_level * (1 + CONFIG["MOMENTUM_TP_PCT_2"])
    else:
        sl = intended_entry + buffer_mult * atr15
        tp1 = entry_level * (1 - CONFIG["MOMENTUM_TP_PCT_1"])
        tp2 = entry_level * (1 - CONFIG["MOMENTUM_TP_PCT_2"])
    rr = abs(tp2 - entry_level) / abs(sl - entry_level)
    if rr < 1.8:
        return None, "Momentum RR too low"
    momentum_score = 85 if quality_count >= 3 else 82
    creation_time = datetime.now(timezone.utc)
    sig = Signal(
        symbol=symbol,
        direction=direction,
        entry_zone=entry_zone,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        rr=rr,
        score=momentum_score,
        reason="Momentum Continuation: Trend expansion - micro pullback entry",
        tf_alignment="MOMENTUM",
        confidence="High",
        validity_min=CONFIG["VALIDITY_MIN"],
        creation_time=creation_time,
        is_momentum=True
    )
    return sig, ""
def calculate_score(symbol: str, dfs: Dict[str, pd.DataFrame]) -> Tuple[Optional[Signal], Dict, str]:
    now = datetime.now(timezone.utc)
    sniper_sig, context, reason = calculate_sniper_score(symbol, dfs, now)
    if sniper_sig:
        return sniper_sig, context, ""
    if reason != "Too far from entry level":
        return None, {}, reason
    intended_entry = context.get("intended_entry")
    direction = context.get("direction")
    if intended_entry is None or direction is None:
        return None, {}, "Failed to extract momentum context"
    momentum_sig, m_reason = calculate_momentum_score(symbol, dfs, intended_entry, direction, now)
    if momentum_sig:
        return momentum_sig, {}, ""
    return None, {}, m_reason or "No momentum setup"
async def fetch_balance():
    if CONFIG["API_KEY"] and CONFIG["API_SECRET"]:
        exchange.apiKey = CONFIG["API_KEY"]
        exchange.secret = CONFIG["API_SECRET"]
        bal = await exchange.fetch_balance()
        return bal.get('USDT', {}).get('free', 0)
    return None
def calculate_size(sig: Signal, balance: float) -> float:
    if not balance:
        return 0
    entry_parts = sig.entry_zone.split('-')
    entry_mid = (float(entry_parts[0]) + float(entry_parts[1])) / 2
    distance = abs(entry_mid - sig.sl) / entry_mid
    risk_pct = CONFIG["RISK_PCT"] * CONFIG["MOMENTUM_RISK_PCT_MULT"] if sig.is_momentum else CONFIG["RISK_PCT"]
    size = (balance * risk_pct) / (distance * CONFIG["LEVERAGE"])
    return size
def send_tele(msg: str):
    msg = msg[:3900]
    if CONFIG["TELEGRAM_TOKEN"] == "YOUR_TELEGRAM_TOKEN" or CONFIG["TELEGRAM_CHAT_ID"] == "YOUR_CHAT_ID":
        json_log("warning", "Telegram placeholders not set, skipping send")
        return
    try:
        url = f"https://api.telegram.org/bot{CONFIG['TELEGRAM_TOKEN']}/sendMessage"
        params = {"chat_id": CONFIG["TELEGRAM_CHAT_ID"], "text": msg}
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        json_log("error", "Telegram send failed", {"error": str(e)})
def log_signal(sig: Signal, breakdown: Dict):
    file_exists = os.path.isfile(CONFIG["LOG_FILE"])
    with open(CONFIG["LOG_FILE"], 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "direction", "entry_zone", "sl", "tp1", "tp2", "rr", "score", "reason"])
        writer.writerow([datetime.now(timezone.utc), sig.symbol, sig.direction, sig.entry_zone, sig.sl, sig.tp1, sig.tp2, sig.rr, sig.score, sig.reason])
async def diagnostic():
    try:
        dfs = {}
        for tf, lim in CONFIG["TIMEFRAMES"].items():
            df = await load_ohlcv("BTC/USDT", tf, lim)
            if df.empty:
                raise ValueError(f"Empty DF for {tf}")
            rsi_val = rsi(df).iloc[-1]
            if not 0 < rsi_val < 100:
                raise ValueError(f"Invalid RSI {rsi_val} on {tf}")
        json_log("info", "Diagnostic passed")
    except Exception as e:
        err_msg = f"Diagnostic failed: {str(e)}"
        send_tele(err_msg)
        json_log("error", err_msg)
async def main():
    json_log("info", "Bot started")
    balance = await fetch_balance()
    try:
        while True:
            try:
                cid = str(uuid.uuid4())
                start_time = time.time()
                now_utc = datetime.now(timezone.utc)
                if now_utc.date() != STATE["last_reset"]:
                    STATE["last_reset"] = now_utc.date()
                    STATE["signals_today"] = 0
                STATE["active_signals"] = [
                    (s, t) for s, t in STATE["active_signals"]
                    if now_utc - t < timedelta(minutes=CONFIG["VALIDITY_MIN"])
                ]
                STATE["locked_symbols"] = {
                    sym: t for sym, t in STATE["locked_symbols"].items()
                    if now_utc < t
                }
                if len(STATE["active_signals"]) >= CONFIG["MAX_OPEN"]:
                    no_trade_msg = f"NO TRADE — Max open trades reached ({CONFIG['MAX_OPEN']})"
                    print(no_trade_msg)
                    await asyncio.sleep(CONFIG["SCAN_INTERVAL"])
                    continue
                if STATE["signals_today"] >= CONFIG["MAX_SIGNALS_PER_DAY"]:
                    no_trade_msg = "NO TRADE — Max signals per day reached"
                    print(no_trade_msg)
                    await asyncio.sleep(CONFIG["SCAN_INTERVAL"] * 4)
                    continue
                if not in_trade_session(now_utc):
                    json_log("info", "Outside trade session", {"cid": cid})
                    await asyncio.sleep(CONFIG["SCAN_INTERVAL"])
                    continue
                if now_utc - STATE["last_diag"] > timedelta(seconds=CONFIG["DIAG_INTERVAL_SEC"]):
                    await diagnostic()
                    STATE["last_diag"] = now_utc
                if STATE["last_btc_check"] is None or now_utc - STATE["last_btc_check"] > timedelta(minutes=15):
                    btc_dfs = {}
                    for tf, lim in CONFIG["TIMEFRAMES"].items():
                        df = await load_ohlcv("BTC/USDT", tf, lim)
                        if not df.empty:
                            btc_dfs[tf] = df
                    if btc_dfs:
                        STATE["btc_mode"] = determine_market_mode(btc_dfs)
                        d1h = btc_dfs['1h']
                        STATE["btc_bias"] = get_bias(d1h, '1h')
                        STATE["btc_atr_regime"] = get_atr_regime(atr(d1h))
                        STATE["btc_vol"] = atr(d1h).iloc[-1] / d1h['close'].iloc[-1]
                        STATE["btc_structure"] = get_structure(d1h, '1h')
                        STATE["last_btc_check"] = now_utc
                async def load_symbol(sym):
                    if sym in STATE["locked_symbols"] and now_utc < STATE["locked_symbols"][sym]:
                        return None
                    sym_dfs = {}
                    for tf, lim in CONFIG["TIMEFRAMES"].items():
                        df = await load_ohlcv(sym, tf, lim)
                        if df.empty:
                            return None
                        sym_dfs[tf] = df
                    return sym, sym_dfs
                tasks = [load_symbol(sym) for sym in CONFIG["SYMBOLS"]]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                sym_dfs_list = [r for r in results if not isinstance(r, Exception) and r]
                if not sym_dfs_list:
                    raise Exception("No data loaded")
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(sym_dfs_list))) as executor:
                    futures = [executor.submit(calculate_score, sym, dfs) for sym, dfs in sym_dfs_list]
                    scores = []
                    skip_reasons = Counter()
                    for future in concurrent.futures.as_completed(futures):
                        sig, breakdown, reason = future.result()
                        if sig:
                            scores.append((sig, breakdown))
                        elif reason:
                            skip_reasons[reason] += 1
                if scores:
                    best_sig, best_breakdown = max(scores, key=lambda x: x[0].score)
                    if now_utc - best_sig.creation_time > timedelta(minutes=CONFIG["VALIDITY_MIN"]):
                        print("Signal expired, skipping")
                        continue
                    best_sig.size = calculate_size(best_sig, balance) if balance else None
                    decimals = CONFIG['PRICE_DECIMALS'].get(best_sig.symbol, 3)
                    dir_str = best_sig.direction
                    prefix = "MOMENTUM CONTINUATION ⚡\n" if best_sig.is_momentum else "TRADE SIGNAL\n"
                    note = "\nNote: Continuation trade — not pullback entry\nReduced risk applied (55% of normal)" if best_sig.is_momentum else ""
                    msg = (
                        f"{prefix}"
                        f"Pair: {best_sig.symbol}\n"
                        f"Direction: {dir_str}\n"
                        f"Move Target: 0.7–1.1%{note}\n"
                        f"Entry Zone: {best_sig.entry_zone}\n"
                        f"Stop Loss: {best_sig.sl:.{decimals}f} (structural)\n"
                        f"Take Profit 1: {best_sig.tp1:.{decimals}f}\n"
                        f"Take Profit 2: {best_sig.tp2:.{decimals}f}\n"
                        f"Confidence: {best_sig.confidence}\n"
                        f"Reasoning: {best_sig.reason}\n"
                        f"Timeframe Alignment: {best_sig.tf_alignment}\n"
                        f"Validity: next {best_sig.validity_min} minutes\n"
                        f"Latency (ms): {(time.time() - start_time) * 1000:.0f}"
                    )
                    if best_sig.size:
                        msg += f"\nSize: {best_sig.size:.4f}"
                    send_tele(msg)
                    print(msg)
                    log_signal(best_sig, {})
                    STATE["signals_today"] += 1
                    STATE["active_signals"].append((best_sig.symbol, now_utc))
                    STATE["locked_symbols"][best_sig.symbol] = now_utc + timedelta(minutes=CONFIG["LOCKED_INTERVAL_MIN"])
                    STATE["last_signal_time"] = now_utc
                    STATE["error_count"] = 0
                else:
                    common_reason = skip_reasons.most_common(1)[0][0] if skip_reasons else "No setups"
                    no_trade_msg = f"NO TRADE — {common_reason}"
                    print(no_trade_msg)
                    STATE["error_count"] = 0
                latency = time.time() - start_time
                if latency > CONFIG["LATENCY_BUDGET_SEC"]:
                    json_log("warning", "Latency exceeded budget", {"latency": latency, "cid": cid})
                json_log("info", "Scan complete", {"latency": latency, "cid": cid})
                await asyncio.sleep(CONFIG["SCAN_INTERVAL"])
            except Exception as e:
                json_log("error", str(e))
                STATE["error_count"] += 1
                if STATE["error_count"] >= CONFIG["CIRCUIT_BREAK_ERRORS"]:
                    send_tele("Circuit break: too many errors")
                    break
                await asyncio.sleep(CONFIG["BACKOFF_SEC"])
    finally:
        await exchange.close()
@app.route('/')
def home():
    return "Bot Awake"
class TestBot(unittest.TestCase):
    def test_ema(self):
        close = [22, 23, 24, 25, 26]
        df = pd.DataFrame({'close': close})
        ema9 = ema(df['close'], 9)
        expected = np.array([22., 22.2, 22.56, 23.048, 23.6384])
        np.testing.assert_allclose(ema9.values, expected, rtol=1e-5)
    def test_atr(self):
        data = {'open': [10, 11, 12], 'high': [11, 12, 13], 'low': [9, 10, 11], 'close': [10.5, 11.5, 12.5]}
        df = pd.DataFrame(data)
        atr_val = atr(df, 2).iloc[-1]
        self.assertGreater(atr_val, 0)
    def test_rsi(self):
        close = [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
        df = pd.DataFrame({'close': close})
        rsi_val = rsi(df).iloc[-1]
        self.assertTrue(0 < rsi_val < 100)
    def test_get_bias(self):
        close = np.arange(1, 201) + 100
        df = pd.DataFrame({'close': close})
        bias = get_bias(df, '1h')
        self.assertEqual(bias, "BULL")
    def test_get_structure(self):
        low = [10, 9, 11, 10, 12]
        high = [15, 14, 16, 15, 17]
        df = pd.DataFrame({'low': low, 'high': high})
        struct = get_structure(df, '15m')
        self.assertEqual(struct, "INTACT_BULL")
    def test_get_session(self):
        mock_time = datetime(2023, 1, 1, 3, 0, tzinfo=timezone.utc)
        session = get_session(mock_time)
        self.assertEqual(session, "ASIA")
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        unittest.main(argv=sys.argv)
    else:
        from threading import Thread
        Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 8080}).start()
        asyncio.run(main())
