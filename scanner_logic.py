import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger("PEAD_System.ScannerLogic")

class VectorizedScannerEngine:
    """
    High-Performance, Multi-Layered Matrix Scanner Engine.
    Computes 9 distinct technical setups simultaneously using vectorized operations
    driven dynamically by parameters in the config.json file.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scanner_settings = config.get("scanner_settings", {})
        self.risk_settings = config.get("risk_management", {})
        
    def compute_daily_matrices(self, df_daily: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Pass 1: Sweeps all assets simultaneously using matrix operations.
        Validates setups and exposes underlying data structures for the ranking layer.
        """
        if df_daily.empty:
            return df_daily, {}

        # 1. Sort the entire data pool by asset and date to ensure technical indicators calculate correctly
        df = df_daily.sort_values(by=['ticker', 'date']).copy()
        
        # 2. Extract dynamic configuration parameters
        rsi_p = self.scanner_settings.get("rsi_period", 14)
        bb_p = self.scanner_settings.get("bb_period", 20)
        bb_s = self.scanner_settings.get("bb_std", 2.0)
        vol_ma_len = self.scanner_settings.get("volume_ma_length", 20)
        breakout_lookback = self.scanner_settings.get("breakout_lookback", 20)
        atr_p = self.risk_settings.get("atr_lookback", 14)
        min_price = self.risk_settings.get("min_share_price", 5.0)

        # Apply basic asset risk filters first to reduce computation overhead
        df = df[df['close'] >= min_price].copy()
        if df.empty:
            return df, {}

        # =========================================================================
        # MATRIX ENGINE CALCULATIONS (Sequence protected to eliminate KeyErrors)
        # =========================================================================
        
        # Moving Average Vectors
        df['sma_20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=bb_p).mean())
        df['sma_50'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=50).mean())
        df['sma_200'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=200).mean())
        
        # True Range and Average True Range (ATR)
        df['prev_close'] = df.groupby('ticker')['close'].shift(1)
        df['high_low'] = df['high'] - df['low']
        df['high_pc'] = (df['high'] - df['prev_close']).abs()
        df['low_pc'] = (df['low'] - df['prev_close']).abs()
        df['true_range'] = df[['high_low', 'high_pc', 'low_pc']].max(axis=1)
        df['atr'] = df.groupby('ticker')['true_range'].transform(lambda x: x.rolling(window=atr_p).mean())
        
        # Bollinger Bands Matrix Calculation
        df['bb_std_val'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=bb_p).std())
        df['bb_upper'] = df['sma_20'] + (df['bb_std_val'] * bb_s)
        df['bb_lower'] = df['sma_20'] - (df['bb_std_val'] * bb_s)
        df['bandwidth'] = (df['bb_upper'] - df['bb_lower']) / (df['sma_20'] + 1e-10)
        df['prev_bandwidth'] = df.groupby('ticker')['bandwidth'].shift(1)
        
        # Relative Strength Index (RSI) Vectorized Block
        df['change'] = df['close'] - df['prev_close']
        df['gain'] = np.where(df['change'] > 0, df['change'], 0.0)
        df['loss'] = np.where(df['change'] < 0, -df['change'], 0.0)
        df['avg_gain'] = df.groupby('ticker')['gain'].transform(lambda x: x.rolling(window=rsi_p).mean())
        df['avg_loss'] = df.groupby('ticker')['loss'].transform(lambda x: x.rolling(window=rsi_p).mean())
        df['rs'] = df['avg_gain'] / (df['avg_loss'] + 1e-10)
        df['rsi_val'] = 100 - (100 / (1 + df['rs']))
        df['prev_rsi'] = df.groupby('ticker')['rsi_val'].shift(1)
        
        # Momentum Vectors & Donchian Breakout Highs
        df['prev_close_10'] = df.groupby('ticker')['close'].shift(10)
        df['roc_10'] = ((df['close'] - df['prev_close_10']) / (df['prev_close_10'] + 1e-10)) * 100
        df['prev_roc_10'] = df.groupby('ticker')['roc_10'].shift(1)
        df['donchian_high'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(window=breakout_lookback).max())
        df['prev_donchian_high'] = df.groupby('ticker')['donchian_high'].shift(1)
        
        # Volume Baselines
        df['volume_sma'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=vol_ma_len).mean())
        
        # Candlestick Formations
        df['prev_open'] = df.groupby('ticker')['open'].shift(1)

        # =========================================================================
        # 9 DISTINCT TECHNICAL SETUPS EVALUATION LAYER
        # =========================================================================
        df['setup_1'] = (df['sma_50'] > df['sma_200']).astype(int)
        df['setup_2'] = (df['close'] > df['sma_20']).astype(int)
        df['setup_3'] = ((df['close'] > df['bb_upper']) & (df['bandwidth'] > df['prev_bandwidth'])).astype(int)
        df['setup_4'] = ((df['prev_rsi'] < 30) & (df['rsi_val'] >= 30)).astype(int)
        df['setup_5'] = ((df['change'] > 0) & (df['volume'] > df['volume_sma'] * 1.5)).astype(int)
        df['setup_6'] = (df['high'] > df['prev_donchian_high']).astype(int)
        df['setup_7'] = ((df['roc_10'] > 0) & (df['roc_10'] > df['prev_roc_10'])).astype(int)
        df['setup_8'] = (df['true_range'] > df['atr'] * 1.3).astype(int)
        df['setup_9'] = ((df['prev_close'] < df['prev_open']) & 
                         (df['close'] > df['open']) & 
                         (df['close'] >= df['prev_open']) & 
                         (df['open'] <= df['prev_close'])).astype(int)

        setup_cols = [f'setup_{i}' for i in range(1, 10)]
        df['total_setups_triggered'] = df[setup_cols].sum(axis=1)

        # Pull the absolute newest completed daily row for each individual asset
        latest_bars = df.groupby('ticker').last().reset_index()
        shortlist_signals = {}

        # Human-readable dictionary matching our exact mathematical triggers
        explanations_catalog = {
            'setup_1': "Golden Cross Trend Alignment: The 50-day moving average is floating cleanly above the 200-day moving average.",
            'setup_2': "Short-Term Trend Guard: Asset price is closing completely above the baseline 20-day simple moving average.",
            'setup_3': "Volatility Band Squeeze Breakout: Price cleared upper Bollinger Band limits while volatility expanded.",
            'setup_4': "RSI Oversold Mean Reversion: The Relative Strength Index crossed above the 30 floor, confirming buyer return.",
            'setup_5': "Institutional Volume Surge: Intraday tracking volume surged over 150% above its trailing baseline average.",
            'setup_6': "Donchian Range Expansion Breakout: Asset price established fresh highs above the trailing lookback ceiling.",
            'setup_7': "Momentum Vector Acceleration: Price Rate of Change curve steepened positively over the trailing 10 sessions.",
            'setup_8': "ATR Volatility Expansion Check: The true trading range expanded past baseline volatility bounds.",
            'setup_9': "Bullish Engulfing Candlestick Reversal: Daily closing print completely swallowed the prior session distribution body."
        }

        for _, row in latest_bars.iterrows():
            ticker = row['ticker']
            triggered_setups = []
            justifications = []

            for s_id in range(1, 10):
                col = f'setup_{s_id}'
                if row[col] == 1:
                    triggered_setups.append(col.upper())
                    justifications.append(explanations_catalog[col])

            # Trend Confirmation Guard: Stacked Moving Average check adds institutional validation
            if row['close'] > row['sma_20'] > row['sma_50'] > row['sma_200']:
                justifications.append("Stacked Moving Averages Alignment: Price > 20 SMA > 50 SMA > 200 SMA structural configuration met.")

            # Filter out assets completely flat or missing triggers
            if triggered_setups and (row['close'] > row['open']):
                shortlist_signals[ticker] = {
                    "setups": triggered_setups,
                    "justifications": justifications,
                    "close": float(row['close']),
                    "atr": float(row['atr'] if not np.isnan(row['atr']) else 0.0),
                    "total_setups": int(row['total_setups_triggered']),
                    "roc_val": float(row['roc_10'] if not np.isnan(row['roc_10']) else 0.0),
                    # Pass the pre-existing database sentiment metrics up to the orchestration layer
                    "avg_sentiment": float(row['avg_sentiment'] if 'avg_sentiment' in row and not np.isnan(row['avg_sentiment']) else 0.0),
                    "news_volume": float(row['news_volume'] if 'news_volume' in row and not np.isnan(row['news_volume']) else 0.0)
                }

        return df, shortlist_signals

    def verify_micro_confluence(self, df_hourly: pd.DataFrame) -> Tuple[bool, str]:
        """
        Pass 2: Hourly Micro-Structure Verification.
        Checks for local support signals on the lower timeframe chart.
        """
        if df_hourly.empty or len(df_hourly) < 5:
            return False, "Insufficient intraday data available to confirm localized footprint."
        
        df = df_hourly.copy()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        last_row = df.iloc[-1]
        
        candle_body = abs(last_row['close'] - last_row['open'])
        candle_range = last_row['high'] - last_row['low']
        lower_wick = min(last_row['open'], last_row['close']) - last_row['low']
        
        has_buyer_support = lower_wick > (candle_body * 0.4) if candle_range > 0 else False
        is_above_trend = last_row['close'] > last_row['sma_20']
        
        if has_buyer_support or is_above_trend:
            confluence_reasons = []
            if is_above_trend: 
                confluence_reasons.append("Hourly price trading cleanly above Local 20-Bar SMA.")
            if has_buyer_support: 
                confluence_reasons.append("Intraday wick profile signals dynamic limit-buying order book support.")
            return True, " & ".join(confluence_reasons)
            
        return False, "Hourly posture tracking flat/bearish; lower timeframe entry timing window closed."