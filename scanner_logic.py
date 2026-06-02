import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger("PEAD_System.ScannerLogic")

class EarningsMomentumScanner:
    def __init__(self, config: dict, db_name: str):
        self.config = config
        self.db_name = db_name
        self.alpha_settings = self.config.get("alpha_ranking_weights", {})
        self.base_weights = self.alpha_settings.get("base_weights", {
            "strategy_confluence_pct": 45.0,
            "momentum_velocity_pct": 35.0,
            "news_sentiment_pct": 20.0
        })
        self.pead_settings = self.alpha_settings.get("earnings_surprise", {
            "max_initial_weight_pct": 30.0,
            "half_life_days": 15.0
        })

    def _get_days_since_earnings(self, ticker: str, current_scan_date: str) -> float:
        """
        Queries the database to find the number of days elapsed between 
        the target scan date and the most recent earnings announcement.
        """
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT MAX(announcement_date) 
                FROM ticker_fundamentals 
                WHERE ticker = ? AND announcement_date <= ?
            """, (ticker.upper(), current_scan_date))
            
            res = cursor.fetchone()
            if res and res[0]:
                last_earnings_dt = datetime.strptime(res[0], "%Y-%m-%d")
                scan_dt = datetime.strptime(current_scan_date, "%Y-%m-%d")
                delta_days = (scan_dt - last_earnings_dt).days
                return float(max(0, delta_days))
        except Exception as e:
            logger.debug(f"Earnings tracking lookup exception for {ticker}: {e}")
        finally:
            conn.close()
        return -1.0

    def compute_decayed_weights(self, ticker: str, current_scan_date: str) -> Dict[str, float]:
        """
        Implements the Continuous Exponential Time-Decay Alpha Model.
        Dynamically shifts factor weights based on the freshness of earnings shocks.
        """
        days_elapsed = self._get_days_since_earnings(ticker, current_scan_date)
        
        # Extract operational limits from config profile
        max_initial = float(self.pead_settings.get("max_initial_weight_pct", 30.0))
        half_life = float(self.pead_settings.get("half_life_days", 15.0))
        
        # 1. Compute Earnings Shock Factor Weight via Exponential Decay
        if days_elapsed >= 0:
            surprise_weight = max_initial * (0.5 ** (days_elapsed / half_life))
        else:
            surprise_weight = 0.0  # Safe resting baseline if no record exists
            
        # 2. Re-scale remaining capacity across baseline parameters
        remaining_capacity = 100.0 - surprise_weight
        
        b_strat = float(self.base_weights.get("strategy_confluence_pct", 45.0))
        b_mom = float(self.base_weights.get("momentum_velocity_pct", 35.0))
        b_news = float(self.base_weights.get("news_sentiment_pct", 20.0))
        total_base_pool = b_strat + b_mom + b_news
        
        if total_base_pool == 0:
            total_base_pool = 100.0

        # Apply proportional decompression matrices
        adjusted_weights = {
            "earnings_surprise_pct": surprise_weight,
            "strategy_confluence_pct": b_strat * (remaining_capacity / total_base_pool),
            "momentum_velocity_pct": b_mom * (remaining_capacity / total_base_pool),
            "news_sentiment_pct": b_news * (remaining_capacity / total_base_pool),
            "days_since_earnings": days_elapsed
        }
        return adjusted_weights


class VectorizedScannerEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scanner_settings = config.get("scanner_settings", {})
        self.sys_settings = config.get("system_settings", {})
        self.db_name = self.sys_settings.get("db_name", "trading_vault.db")
        
    def compute_daily_matrices(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Executes Pass 1 Macro Analysis across all tickers simultaneously 
        using high-performance array broadcasting vectors.
        """
        if df.empty:
            return df, {}

        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Technical parameters pulled dynamically from JSON specifications
        rsi_period = int(self.scanner_settings.get("rsi_period", 14))
        bb_period = int(self.scanner_settings.get("bb_period", 20))
        bb_std = float(self.scanner_settings.get("bb_std", 2.0))
        
        # Vectorized Math Calculations Grouped by Ticker
        grouped = df.groupby('ticker')
        
        df['ma20'] = grouped['close'].transform(lambda x: x.rolling(bb_period).mean())
        df['ma200'] = grouped['close'].transform(lambda x: x.rolling(200).mean())
        df['std20'] = grouped['close'].transform(lambda x: x.rolling(bb_period).std())
        df['upper_bb'] = df['ma20'] + (df['std20'] * bb_std)
        df['lower_bb'] = df['ma20'] - (df['std20'] * bb_std)
        df['volume_ma20'] = grouped['volume'].transform(lambda x: x.rolling(20).mean())
        df['roll_high20'] = grouped['high'].transform(lambda x: x.rolling(20).max().shift(1))
        df['roll_low20'] = grouped['low'].transform(lambda x: x.rolling(20).min().shift(1))
        
        # True Range & ATR Matrix Formula
        df['prev_close'] = grouped['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['prev_close']).abs()
        df['tr3'] = (df['low'] - df['prev_close']).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df.groupby('ticker')['tr'].transform(lambda x: x.rolling(14).mean())

        # Vectorized Relative Strength Index Matrix Engine
        delta = df.groupby('ticker')['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = df.groupby('ticker')['close'].apply(lambda x: x.diff().clip(lower=0).rolling(rsi_period).mean()).reset_index(level=0, drop=True)
        avg_loss = df.groupby('ticker')['close'].apply(lambda x: -x.diff().clip(upper=0).rolling(rsi_period).mean()).reset_index(level=0, drop=True)
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        # ---------------------------------------------------------------------
        # 9 DISTINCT TECHNICAL MATRIX CONDITIONS
        # ---------------------------------------------------------------------
        df['setup_1'] = df['close'] > df['roll_high20']
        df['setup_2'] = df['ma20'] > df['ma200']
        df['setup_3'] = df['rsi'] < 35
        df['setup_4'] = df['close'] < df['lower_bb']
        df['setup_5'] = df['volume'] > (df['volume_ma20'] * 2.0)
        df['setup_6'] = (df['close'] >= df['roll_low20']) & (df['close'] <= df['roll_low20'] * 1.02)
        df['setup_7'] = (df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)
        df['bb_width'] = (df['upper_bb'] - df['lower_bb']) / (df['ma20'] + 1e-9)
        df['setup_8'] = df['bb_width'] < df.groupby('ticker')['bb_width'].transform(lambda x: x.rolling(40).mean())
        df['setup_9'] = df['close'] > df['ma200']

        setup_cols = [f'setup_{i}' for i in range(1, 10)]
        df['active_setups_count'] = df[setup_cols].sum(axis=1)
        
        # Filter for the most recent day of calculation activity
        latest_rows = df.groupby('ticker').last().reset_index()
        
        shortlist_signals = {}
        setup_explanations = {
            'setup_1': "Channel Breakout (New 20-Day High Close)",
            'setup_2': "Golden Cross Trend Model Alignment",
            'setup_3': "Oversold Mean Reversion Inversion",
            'setup_4': "Volatility Band Bounce (Lower Bollinger Touch)",
            'setup_5': "Institutional Volume Spike Detection",
            'setup_6': "Structural Support Deflection Floor",
            'setup_7': "Velocity Breakthrough (RSI Crosses Over 50)",
            'setup_8': "Volatility Squeeze Compression Window",
            'setup_9': "Structural Bullish Armor Over MA 200"
        }

        for _, row in latest_rows.iterrows():
            ticker = row['ticker']
            cnt = int(row['active_setups_count'])
            
            if cnt > 0:
                justifications = []
                for sc in setup_cols:
                    if row[sc]:
                        justifications.append(setup_explanations[sc])
                
                # Standardize data payload interface mapping
                shortlist_signals[ticker] = {
                    "ticker": ticker,
                    "date": row['date'],
                    "close": float(row['close']),
                    "volume": float(row['volume']),
                    "rsi": float(row['rsi'] if not pd.isna(row['rsi']) else 50.0),
                    "atr": float(row['atr'] if not pd.isna(row['atr']) else 1.0),
                    "active_setups_count": cnt,
                    "total_setups": 9.0,
                    "score_multiplier": float(cnt / 9.0),  # Normalize to 0.0 - 1.0 range
                    "justifications": justifications
                }

        return df, shortlist_signals

    def verify_micro_confluence(self, df_hourly: pd.DataFrame) -> Tuple[bool, str]:
        """
        Pass 2: Hourly Micro-Structure Verification.
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
                confluence_reasons.append("Intraday wick profile signals dynamic order book support.")
            return True, " & ".join(confluence_reasons)
            
        return False, "Hourly momentum trading underneath localized moving average baseline floor."