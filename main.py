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
from scanner_logic import VectorizedScannerEngine
from data_factory import HistoricalDataFactory

# FIXED: Ensure environment configurations load before initializing factories
load_dotenv()

def load_system_config() -> Dict[str, Any]:
    with open('config.json', 'r') as f:
        return json.load(f)

def load_macro_daily_data(db_name: str, tickers: List[str]) -> pd.DataFrame:
    conn = sqlite3.connect(db_name)
    ticker_placeholders = ",".join([f"'{t}'" for t in tickers])
    # FIXED: Select avg_sentiment and news_volume parameters during Pass 1 sweep 
    # to allow multi-factor alpha scoring immediately after Pass 1.
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
    cursor.execute("""
        SELECT reported_eps, expected_eps, reported_rev, expected_rev 
        FROM ticker_fundamentals WHERE ticker=? ORDER BY announcement_date DESC LIMIT 1
    """, (ticker,))
    fund_row = cursor.fetchone()
    
    cursor.execute("""
        SELECT avg_sentiment, news_volume FROM ticker_data 
        WHERE ticker=? AND avg_sentiment IS NOT NULL ORDER BY date DESC LIMIT 1
    """, (ticker,))
    sent_row = cursor.fetchone()
    conn.close()
    
    eps_str = f"EPS: Rep={fund_row[0]} Exp={fund_row[1]}" if fund_row else "EPS: N/A"
    rev_str = f"Rev: Rep={fund_row[2]} Exp={fund_row[3]}" if fund_row else "Rev: N/A"
    sentiment = float(sent_row[0]) if sent_row else 0.0
    volume = int(sent_row[1]) if sent_row else 0
    
    return {
        "eps": eps_str,
        "revenue": rev_str,
        "sentiment": sentiment,
        "news_volume": volume
    }

async def main_async():
    print("="*90)
    print(" INTERCHANGEABLE DATA FACTORY PIPELINE RUNNER")
    print("="*90)
    
    config = load_system_config()
    sys_settings = config.get("system_settings", {})
    scanner_settings = config.get("scanner_settings", {})
    weights = config.get("alpha_ranking_weights", {})
    
    db_name = sys_settings.get("db_name", "trading_vault.db")
    tickers = config.get("trading_universe", {}).get("tickers", [])
    max_shortlist = scanner_settings.get("max_shortlist_size", 25)
    lookback_days = scanner_settings.get("pass2_lookback_days", 30)
    
    if not tickers:
        print("[!] Execution universe is empty. Verify config.json architecture configurations.")
        return
        
    print(f"[*] Executing Pass 1 Macro Analysis across [{len(tickers)}] securities...")
    df_daily = load_macro_daily_data(db_name, tickers)
    
    if df_daily.empty:
        print("[!] No clean baseline macro daily historical data sequences found inside target SQLite vault.")
        return
        
    scanner = VectorizedScannerEngine(atr_lookback=config.get("risk_management", {}).get("atr_lookback", 14))
    df_analyzed, shortlist_signals = scanner.compute_daily_matrices(df_daily)
    
    shortlist_keys = list(shortlist_signals.keys())
    print(f"[+] Pass 1 Complete. Matrix tracking flagged [{len(shortlist_keys)}] candidates.")
    
    if not shortlist_keys:
        print("[-] Pipeline process terminated: No tickers qualified during primary vector tracking passes.")
        return
        
    print("[*] Computing Multi-Factor Alpha Ranking scores for Pass 1 Shortlist...")
    ranked_candidates = []
    for ticker in shortlist_keys:
        sig_info = shortlist_signals[ticker]
        
        # Pull last state rows to extract technical alpha weights
        t_rows = df_analyzed[df_analyzed['ticker'] == ticker]
        if t_rows.empty:
            continue
        last_row = t_rows.iloc[-1]
        
        # Calculate Technical Core Base Component
        tech_score = float(sig_info["score_multiplier"]) * float(weights.get("strategy_confluence_pct", 40.0))
        
        # Calculate Momentum Component (e.g. tracking close relative to structural moving averages)
        mom_score = 0.0
        if last_row.get('sma_20') and last_row['sma_20'] > 0:
            mom_score = ((last_row['close'] - last_row['sma_20']) / last_row['sma_20']) * float(weights.get("momentum_velocity_pct", 40.0))
            
        # Calculate Alternative Fundamental Sentiment Component
        sent_score = 0.0
        avg_sent = last_row.get('avg_sentiment', 0.0)
        news_vol = last_row.get('news_volume', 0)
        if avg_sent and news_vol > 0:
            confidence = min(float(news_vol) / 5.0, 1.0)
            sent_score = (float(avg_sent) * confidence) * float(weights.get("sentiment_tailwind_pct", 20.0))
            
        composite_alpha = tech_score + mom_score + sent_score
        ranked_candidates.append({
            "ticker": ticker,
            "score": composite_alpha,
            "tech_base": tech_score,
            "tailwind": sent_score,
            "info": sig_info
        })
        
    ranked_candidates = sorted(ranked_candidates, key=lambda x: x["score"], reverse=True)
    shortlist_keys = [c["ticker"] for c in ranked_candidates[:max_shortlist]]
    
    print(f"[+] Multi-Factor filtering compressed execution scope down to the top [{len(shortlist_keys)}] assets.")
    
    data_factory = HistoricalDataFactory(config)
    ib_client = None
    if data_factory.hourly_source == "IBKR":
        try:
            ib_client = IB()
            await ib_client.connectAsync('127.0.0.1', data_factory.ib_port, clientId=99)
        except Exception as e:
            print(f"[!] Warning: Failed to boot secondary async IBKR connection state link: {e}")
            print("    Reverting fallback systems to stateless connections where applicable.")
            ib_client = None

    print(f"[*] Dispatching Pass 2 lower timeframe confirmation via configured channel: {data_factory.hourly_source}")
    
    selected_targets = []
    total_candidates = len(shortlist_keys)
    
    # UPGRADED: Added counter loop with percentage math to prevent frozen terminal perception
    for idx, ticker in enumerate(shortlist_keys, 1):
        pct = (idx / total_candidates) * 100
        print(f"  [{idx}/{total_candidates}] ({pct:.1f}%) Extracting intraday hourly structures for: {ticker}")
        
        try:
            df_hourly = await data_factory.fetch_hourly_bars(ticker, lookback_days, ib_client)
            
            if df_hourly.empty:
                print(f"      [!] Empty data payload returned for {ticker}. Skipping validation.")
                continue
                
            is_confirmed, confluence_msg = scanner.verify_micro_confluence(df_hourly)
            if is_confirmed:
                print(f"      [✓] PASS 2 CONFIRMED: {ticker} matched lower timeframe confluence rules.")
                # Match meta records back out of original alpha sorting array
                orig_meta = next(item for item in ranked_candidates if item["ticker"] == ticker)
                meta_ext = fetch_alpha_metadata(db_name, ticker)
                
                selected_targets.append({
                    "ticker": ticker,
                    "score": orig_meta["score"],
                    "tailwind": orig_meta["tailwind"],
                    "info": orig_meta["info"],
                    "meta": meta_ext,
                    "confluence_msg": confluence_msg
                })
            else:
                print(f"      [x] Pass 2 Rejected: {ticker} failed intraday confirmation.")
                
        except Exception as e:
            print(f"      [ERROR] Fatal structural error handling pass 2 validation for {ticker}: {e}")

    if ib_client and ib_client.isConnected():
        ib_client.disconnect()
    data_factory.close()

    print("\n" + "="*90)
    print(" HIGH CONVICTION SYSTEM SIGNAL TARGET MATRIX ENGINE REPORT")
    print("="*90)
    
    # Sort remaining assets by score final output checks
    final_selected_targets = sorted(selected_targets, key=lambda x: x["score"], reverse=True)
    
    for rank, target in enumerate(final_selected_targets, 1):
        ticker = target["ticker"]
        info = target["info"]
        meta = target["meta"]
        sentiment_status = "BULLISH" if meta["sentiment"] > 0.10 else "BEARISH" if meta["sentiment"] < -0.10 else "NEUTRAL"
        
        print(f"\n[RANK #{rank} TARGET ACQUISITION LOCK: {ticker.upper()}]")
        print(f"  System Alpha Ranking Score: {target['score']:.2f} | DEFENSIVE COMPOSITE TAILWIND MATRIX SCORE: {target['tailwind']:.4f}")
        print(f"  Daily Close Price: ${info['close']:.2f} | Dynamic Stop Range (2x ATR): ${info['atr']*2:.2f}")
        print("-" * 75)
        print("  Precise Mathematical & Geometric Trade Logic Justification:")
        for justification in info["justifications"]:
            print(f"   -> {justification}")
            
        print("\n  Lower Timeframe Execution Confluence Metrics:")
        print(f"   [Micro Confirmation]: {target['confluence_msg']}")
        
        print("\n  Fundamental & Narrative Alpha Overlay Metrics:")
        print(f"   - Corporate Financial Status: {meta['revenue']} | {meta['eps']}")
        print(f"   - Alternative NLP News Ingestion: Media Tone: {meta['sentiment']:.3f} [{sentiment_status} via {meta['news_volume']} Headlines]")
        print("="*90)

    if not final_selected_targets:
        print("\n[!] Scanning loop complete: Zero structural targets satisfied both Pass 1 & Pass 2 thresholds.")
        print("="*90)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_async())