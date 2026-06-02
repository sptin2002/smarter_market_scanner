import os
import sys
import json
import asyncio
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dotenv import load_dotenv
from ib_insync import IB

# Modular analytical frameworks imports
from init_universe import DataInitializationEngine
from scanner_logic import VectorizedScannerEngine, EarningsMomentumScanner
from data_factory import HistoricalDataFactory

load_dotenv()

def load_system_config() -> Dict[str, Any]:
    with open('config.json', 'r') as f:
        return json.load(f)

def load_macro_daily_data(db_name: str, tickers: List[str]) -> pd.DataFrame:
    conn = sqlite3.connect(db_name)
    ticker_placeholders = ",".join([f"'{t}'" for t in tickers])
    query = f"""
        SELECT date, ticker, open, high, low, close, volume, avg_sentiment, news_volume
        FROM ticker_data 
        WHERE ticker IN ({ticker_placeholders}) AND open IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def fetch_alpha_metadata(db_name: str, ticker: str) -> Dict[str, Any]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    meta = {"revenue": "N/A Surprise Data", "eps": "N/A Surprise Data", "surprise_pct": 0.0}
    try:
        cursor.execute("""
            SELECT revenue_surprise_pct, eps_surprise_pct 
            FROM ticker_fundamentals 
            WHERE ticker = ? 
            ORDER BY announcement_date DESC LIMIT 1
        """, (ticker.upper(),))
        row = cursor.fetchone()
        if row:
            rev_s = row[0] if row[0] is not None else 0.0
            eps_s = row[1] if row[1] is not None else 0.0
            meta["revenue"] = f"Rev Surprise: {rev_s:+.2f}%"
            meta["eps"] = f"EPS Surprise: {eps_s:+.2f}%"
            meta["surprise_pct"] = float((abs(rev_s) + abs(eps_s)) / 2.0)
    except Exception:
        pass
    finally:
        conn.close()
    return meta

async def main_async():
    print("="*90)
    print(" INTERCHANGEABLE DATA FACTORY PIPELINE RUNNER")
    print("="*90)
    
    config = load_system_config()
    sys_settings = config.get("system_settings", {})
    db_name = sys_settings.get("db_name", "trading_vault.db")
    tickers = config.get("trading_universe", {}).get("tickers", [])
    
    if not tickers:
        print("[!] Execution universe array is empty inside config.json.")
        return

    print(f"[*] Executing Pass 1 Macro Analysis across [{len(tickers)}] securities...")
    df_daily = load_macro_daily_data(db_name, tickers)
    
    if df_daily.empty:
        print("[!] No clean baseline macro daily historical data sequences found inside target SQLite vault.")
        return
        
    # Phase 1: Initialize the engine with the full configuration profile
    scanner = VectorizedScannerEngine(config)
    df_analyzed, shortlist_signals = scanner.compute_daily_matrices(df_daily)
    
    shortlist_keys = list(shortlist_signals.keys())
    print(f"[+] Pass 1 Complete. Matrix tracking flagged [{len(shortlist_keys)}] candidates.")
    print("[*] Computing Dynamic Multi-Factor Alpha Ranking scores for Pass 1 Shortlist...")
    
    # Initialize the time-decay weight calculation routine
    decay_engine = EarningsMomentumScanner(config, db_name)
    current_scan_date = datetime.today().strftime('%Y-%m-%d')
    
    alpha_ranked_pool = []
    data_factory = HistoricalDataFactory(config)

    for ticker in shortlist_keys:
        sig_info = shortlist_signals[ticker]
        
        # Calculate dynamic time-decay weights per asset
        adj_weights = decay_engine.compute_decayed_weights(ticker, current_scan_date)
        
        # Technical Score Component: Resolve key drift safely via standard fallbacks
        multiplier_key = next((k for k in ["score_multiplier", "confluence_multiplier", "confluence_score", "active_setups_count"] if k in sig_info), None)
        if multiplier_key:
            raw_mult = float(sig_info[multiplier_key])
            if multiplier_key == "active_setups_count" and raw_mult > 1.0:
                raw_mult = raw_mult / float(sig_info.get("total_setups", 9.0))
            score_multiplier = raw_mult
        else:
            score_multiplier = 0.5
            
        tech_score = score_multiplier * float(adj_weights["strategy_confluence_pct"])

        # Momentum Score Component: Distance from average baseline velocity
        rsi_val = float(sig_info.get("rsi", 50.0))
        momentum_score = (rsi_val / 100.0) * float(adj_weights["momentum_velocity_pct"])
        
        # Alternative Sentiment Component Engine
        ticker_rows = df_daily[df_daily['ticker'] == ticker]
        if not ticker_rows.empty:
            last_row = ticker_rows.iloc[-1]
            raw_sent = float(last_row.get('avg_sentiment', 0.0))
            raw_vol = float(last_row.get('news_volume', 0.0))
            confidence_multiplier = min(raw_vol / 5.0, 1.0)
            sentiment_score = (raw_sent * confidence_multiplier * float(adj_weights["news_sentiment_pct"]))
        else:
            raw_sent, raw_vol, sentiment_score = 0.0, 0.0, 0.0
            
        # Fundamental PEAD Shock Component Engine
        fund_meta = fetch_alpha_metadata(db_name, ticker)
        fundamental_shock_score = (min(fund_meta["surprise_pct"] / 100.0, 1.0) * float(adj_weights["earnings_surprise_pct"]))

        # Composite Multi-Factor Rank Mapping
        composite_alpha_score = tech_score + momentum_score + sentiment_score + fundamental_shock_score
        
        # Form structural tracking metrics payloads
        alpha_ranked_pool.append({
            "ticker": ticker,
            "score": composite_alpha_score,
            "weights_used": adj_weights,
            "info": sig_info,
            "meta": {
                "revenue": fund_meta["revenue"],
                "eps": fund_meta["eps"],
                "sentiment": raw_sent,
                "news_volume": raw_vol
            }
        })

    # Sort candidates by descending alpha score profiles
    alpha_ranked_pool = sorted(alpha_ranked_pool, key=lambda x: x["score"], reverse=True)
    target_shortlist = alpha_ranked_pool[:int(config.get("scanner_settings", {}).get("max_pass2_candidates", 15))]
    
    print(f"[+] Ranking Complete. Formed top allocation shortlist of [{len(target_shortlist)}] targets.")
    print("[*] Initiating Pass 2 Micro-Timeframe Intraday Validation...")

    final_selected_targets = []
    end_date_str = datetime.today().strftime('%Y-%m-%d')
    start_date_str = (datetime.today() - timedelta(days=int(config.get("scanner_settings", {}).get("pass2_lookback_days", 30)))).strftime('%Y-%m-%d')

    for target in target_shortlist:
        ticker = target["ticker"]
        try:
            # Dynamically fetch lower-timeframe intraday profiles
            df_hourly = await data_factory.fetch_hourly_bars(ticker, start_date=start_date_str, end_date=end_date_str)
            passed_micro, confluence_msg = scanner.verify_micro_confluence(df_hourly)
            
            if passed_micro:
                target["confluence_msg"] = confluence_msg
                final_selected_targets.append(target)
                print(f"  -> [✓] {ticker} verified successfully on intraday timeframes.")
            else:
                print(f"  -> [❌] {ticker} filtered out: {confluence_msg}")
        except Exception as e:
            print(f"  -> [!] Bypassed {ticker} due to data fetching exception: {e}")

    # Output executive summary metrics reports
    print("\n" + "="*90)
    print(" EXECUTIVE QUANT ALPHA SCANNER SEARCH SEED TERMINAL REPORT")
    print("="*90)
    
    for rank, target in enumerate(final_selected_targets, 1):
        ticker = target["ticker"]
        info = target["info"]
        meta = target["meta"]
        w = target["weights_used"]
        sentiment_status = "BULLISH" if meta["sentiment"] > 0.10 else "BEARISH" if meta["sentiment"] < -0.10 else "NEUTRAL"
        
        print(f"\n[RANK #{rank} TARGET ACQUISITION LOCK: {ticker.upper()}]")
        print(f"  System Combined Alpha Score: {target['score']:.2f} | Days Since Earnings: {w['days_since_earnings']}")
        print(f"  Dynamic Allocation Profiles: PEAD Shock: {w['earnings_surprise_pct']:.1f}% | Tech: {w['strategy_confluence_pct']:.1f}% | Mom: {w['momentum_velocity_pct']:.1f}%")
        print(f"  Daily Close Price: ${info['close']:.2f} | Dynamic Stop Tracking Range (2x ATR): ${info['atr']*2:.2f}")
        print("-" * 80)
        print("  Active Technical Setup Multipliers Found:")
        for justification in info["justifications"]:
            print(f"   -> [✓] {justification}")
            
        print("\n  Lower Timeframe Confirmation Confluence Metrics:")
        print(f"   [Micro Footprint]: {target['confluence_msg']}")
        print(f"   - NLP Alternative News Score: {meta['sentiment']:.3f} [{sentiment_status} over {int(meta['news_volume'])} headlines]")
        print("="*90)

    if not final_selected_targets:
        print("\n[!] Scanning loop complete: Zero structural targets satisfied both Pass 1 & Pass 2 thresholds.")
        print("="*90)
        
    data_factory.close()

if __name__ == "__main__":
    asyncio.run(main_async())