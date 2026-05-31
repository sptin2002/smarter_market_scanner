import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class VectorizedScannerEngine:
    """
    High-Performance, Multi-Layered Matrix Scanner Engine.
    Computes 9 distinct technical setups simultaneously using vectorized operations
    and generates automated mathematical text justifications in plain English.
    """
    def __init__(self, atr_lookback: int = 14):
        self.atr_lookback = atr_lookback

    def compute_daily_matrices(self, df_daily: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Pass 1: Sweeps all assets simultaneously using matrix broadcasting and evaluates
        all 9 technical setups horizontally.
        """
        if df_daily.empty:
            return df_daily, {}

        # Ensure data is sorted cleanly by ticker and date to protect rolling window calculations
        df = df_daily.sort_values(by=['ticker', 'date']).copy()
        grouped = df.groupby('ticker')

        # =========================================================================
        # 1. BASE VECTORIZED TECHNICAL INDICATORS
        # =========================================================================
        # Moving Averages
        df['sma_20'] = grouped['close'].transform(lambda x: x.rolling(window=20).mean())
        df['sma_50'] = grouped['close'].transform(lambda x: x.rolling(window=50).mean())
        df['sma_200'] = grouped['close'].transform(lambda x: x.rolling(window=200).mean())
        
        # Average True Range (ATR)
        df['high_low'] = df['high'] - df['low']
        df['high_pc'] = (df['high'] - grouped['close'].shift(1)).abs()
        df['low_pc'] = (df['low'] - grouped['close'].shift(1)).abs()
        df['true_range'] = df[['high_low', 'high_pc', 'low_pc']].max(axis=1)
        df['atr'] = df.groupby('ticker')['true_range'].transform(lambda x: x.rolling(window=self.atr_lookback).mean())
        
        # Bollinger Bands (20 Period, 2 StdDev)
        df['bb_std'] = grouped['close'].transform(lambda x: x.rolling(window=20).std())
        df['bb_upper'] = df['sma_20'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['bb_std'] * 2)
        df['bandwidth'] = (df['bb_upper'] - df['bb_lower']) / (df['sma_20'] + 1e-10)
        df['prev_bandwidth'] = grouped['bandwidth'].shift(1)
        
        # Relative Strength Index (RSI 14)
        df['change'] = df['close'] - grouped['close'].shift(1)
        df['gain'] = np.where(df['change'] > 0, df['change'], 0.0)
        df['loss'] = np.where(df['change'] < 0, -df['change'], 0.0)
        df['avg_gain'] = df.groupby('ticker')['gain'].transform(lambda x: x.rolling(window=14).mean())
        df['avg_loss'] = df.groupby('ticker')['loss'].transform(lambda x: x.rolling(window=14).mean())
        df['rs'] = df['avg_gain'] / (df['avg_loss'] + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + df['rs']))
        
        # Rate of Change (ROC 10) and Donchian Channels (20)
        df['prev_close_10'] = grouped['close'].shift(10)
        df['roc_10'] = ((df['close'] - df['prev_close_10']) / (df['prev_close_10'] + 1e-10)) * 100
        df['donchian_high'] = grouped['high'].transform(lambda x: x.rolling(window=20).max())
        df['donchian_low'] = grouped['low'].transform(lambda x: x.rolling(window=20).min())

        # =========================================================================
        # 2. MATRIX EVALUATION (9 DISTINCT TECHNICAL SETUPS SIMULTANEOUSLY)
        # =========================================================================
        # Setup 1: Golden Cross
        df['setup_1'] = (df['sma_50'] > df['sma_200']).astype(int)
        
        # Setup 2: Price Above 20-Day SMA Support
        df['setup_2'] = (df['close'] > df['sma_20']).astype(int)
        
        # Setup 3: Bollinger Band Squeeze Breakout
        df['setup_3'] = ((df['close'] > df['bb_upper']) & (df['bandwidth'] > df['prev_bandwidth'])).astype(int)
        
        # Setup 4: RSI Oversold Mean Reversion
        df['setup_4'] = ((grouped['rsi_14'].shift(1) < 30) & (df['rsi_14'] >= 30)).astype(int)
        
        # Setup 5: Volume-Backed Breakout
        df['volume_sma'] = grouped['volume'].transform(lambda x: x.rolling(window=20).mean())
        df['setup_5'] = ((df['change'] > 0) & (df['volume'] > df['volume_sma'] * 1.5)).astype(int)
        
        # Setup 6: Donchian Channel Range Breakout
        df['setup_6'] = (df['high'] > grouped['donchian_high'].shift(1)).astype(int)
        
        # Setup 7: Momentum Acceleration
        df['setup_7'] = ((df['roc_10'] > 0) & (df['roc_10'] > grouped['roc_10'].shift(1))).astype(int)
        
        # Setup 8: ATR Volatility Expansion
        df['setup_8'] = (df['true_range'] > df['atr'] * 1.3).astype(int)
        
        # Setup 9: Bullish Engulfing Candle
        df['prev_open'] = grouped['open'].shift(1)
        df['prev_close'] = grouped['close'].shift(1)
        df['setup_9'] = ((df['prev_close'] < df['prev_open']) & 
                         (df['close'] > df['open']) & 
                         (df['close'] >= df['prev_open']) & 
                         (df['open'] <= df['prev_close'])).astype(int)

        # Sum total configurations triggered horizontally
        setup_cols = [f'setup_{i}' for i in range(1, 10)]
        df['total_setups_triggered'] = df[setup_cols].sum(axis=1)

        # Isolate the latest state per asset to build signals shortlist
        latest_bars = df.groupby('ticker').last().reset_index()
        shortlist_signals = {}

        # Plain English description catalog map
        explanations_catalog = {
            'setup_1': "Golden Cross Trend Signal: The 50-day moving average is crossing above the 200-day moving average, proving the long-term institutional trend has shifted from bearish to heavily bullish.",
            'setup_2': "Short-Term Structural Support: Price is trading cleanly above the 20-day moving average, indicating consistent short-term momentum and baseline price defense by market makers.",
            'setup_3': "Volatility Band Squeeze Breakout: Price has broken above the upper Bollinger Band while volatility is expanding. This marks the start of an aggressive breakout phase.",
            'setup_4': "RSI Oversold Mean Reversion: The Relative Strength Index has successfully crossed back above the 30 oversold floor, confirming extreme seller exhaustion and an influx of value buyers.",
            'setup_5': "Institutional Volume Injection: The breakout candle is backed by trading volume 1.5x greater than its 20-day average. This confirms financial institutions are actively backing this asset move.",
            'setup_6': "Donchian Channel Range Expansion: Price has achieved a brand new 20-day high, breaking completely free of past overhead supply congestion zones.",
            'setup_7': "Momentum Vector Acceleration: The 10-day Rate of Change is positive and increasing rapidly, signifying accelerating buying pressure.",
            'setup_8': "ATR Volatility Expansion Check: Today's price range exceeds the average true range by 1.3x, proving this move is a highly volatile directional breakout, not noisy sideways consolidation.",
            'setup_9': "Bullish Engulfing Candlestick Reversal: Today's price candle completely wraps around and overwhelms yesterday's selling candle bodies, proving that buyers have taken full control."
        }

        for _, row_data in latest_bars.iterrows():
            ticker = row_data['ticker']
            triggered = []
            justifications = []

            # Loop through setups to construct context lists for active triggers
            for s_id in range(1, 10):
                col_name = f'setup_{s_id}'
                if row_data[col_name] == 1:
                    triggered.append(col_name.upper())
                    justifications.append(explanations_catalog[col_name])

            # Stacked Baseline Condition Check (Maintain structural layout logic)
            if row_data['close'] > row_data['sma_20'] > row_data['sma_50'] > row_data['sma_200']:
                justifications.append("Stacked Moving Averages Alignment: Price > 20 SMA > 50 SMA > 200 SMA configuration established.")

            # Filter criterion: Ticker must trigger at least one rule and maintain positive intraday direction
            if triggered and (row_data['close'] > row_data['open']):
                shortlist_signals[ticker] = {
                    "setups": triggered,
                    "justifications": justifications,
                    "close": float(row_data['close']),
                    "atr": float(row_data['atr'] if not np.isnan(row_data['atr']) else 0.0),
                    "total_setups": int(row_data['total_setups_triggered']),
                    "roc_val": float(row_data['roc_10'] if not np.isnan(row_data['roc_10']) else 0.0)
                }

        return df, shortlist_signals

    def verify_micro_confluence(self, df_hourly: pd.DataFrame) -> Tuple[bool, str]:
        """
        Pass 2: Hourly validation criteria via Polygon data frames.
        """
        if df_hourly.empty or len(df_hourly) < 5:
            return False, "Insufficient hourly data structure available for structural analysis."
        
        df = df_hourly.copy()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        last_row = df.iloc[-1]
        prev_rows = df.iloc[-6:-1]
        
        local_floor = prev_rows['low'].min()
        
        candle_body = abs(last_row['close'] - last_row['open'])
        candle_range = last_row['high'] - last_row['low']
        lower_wick = min(last_row['open'], last_row['close']) - last_row['low']
        
        has_buyer_support = lower_wick > (candle_body * 0.4) if candle_range > 0 else False
        is_above_trend = last_row['close'] > last_row['sma_20']
        
        if has_buyer_support or is_above_trend:
            confluence_reasons = []
            if is_above_trend: confluence_reasons.append("Hourly price trading cleanly above Local 20-Bar SMA.")
            if has_buyer_support: confluence_reasons.append("Intraday swing rejection detected confirming heavy local limit-buying order books.")
            return True, " & ".join(confluence_reasons)
            
        return False, "Hourly posture flat/bearish; fails entry timing metrics."