import os
import sys
import json
import asyncio
import sqlite3
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dotenv import load_dotenv

# Dynamic direct import from your project infrastructure files
from init_universe import DataInitializationEngine
from scanner_logic import VectorizedScannerEngine

def load_system_config() -> Dict[str, Any]:
    with open('config.json', 'r') as f:
        return json.load(f)

def load_macro_daily_data(db_name: str, tickers: List[str]) -> pd.DataFrame:
    """Extracts the entire daily trading history out of your database vault."""
    conn = sqlite3.connect(db_name)
    ticker_placeholders = ",".join([f"'{t}'" for t in tickers])
    query = f"""
        SELECT date, ticker, open, high, low, close, volume 
        FROM ticker_data 
        WHERE ticker IN ({ticker_placeholders}) AND open IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def fetch_on_demand_hourly_data(ticker: str, lookback_days: int, api_key: str) -> pd.DataFrame:
    """Pass 2 Data Ingestion: Requests localized 1-hour interval segments via Polygon API."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "apiKey": api_key}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "results" in data and data["results"]:
                df_h = pd.DataFrame(data["results"])
                df_h = df_h.rename(columns={
                    'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'
                })
                return df_h
    except Exception as e:
        print(f"  [!] Polygon Intraday API error for {ticker}: {e}")
    return pd.DataFrame()

def fetch_alpha_metadata(db_name: str, ticker: str) -> Dict[str, Any]:
    """
    Pass 3 Enrichment: Extracts the latest alternative corporate updates 
    and scalar Float texts scores directly from your schema boundaries.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Fundamentals query boundary tracking latest announcements
    cursor.execute("""
        SELECT reported_eps, expected_eps, reported_rev, expected_rev 
        FROM ticker_fundamentals 
        WHERE ticker=? ORDER BY announcement_date DESC LIMIT 1
    """, (ticker,))
    fund_row = cursor.fetchone()
    
    # Sentiment matrix record retrieval
    cursor.execute("""
        SELECT avg_sentiment, news_volume FROM ticker_data 
        WHERE ticker=? AND avg_sentiment IS NOT NULL AND news_volume > 0 
        ORDER BY date DESC LIMIT 1
    """, (ticker,))
    sent_row = cursor.fetchone()
    conn.close()
    
    meta = {
        "reported_eps": 0.0, "expected_eps": 0.0, "earnings_surprise": 0.0,
        "avg_sentiment": 0.0, "news_volume": 0, "media_buzz": 0.0,
        "revenue_status": "N/A", "eps_status": "N/A"
    }
    
    if fund_row:
        rep_eps, exp_eps, rep_rev, exp_rev = fund_row
        meta["reported_eps"] = rep_eps or 0.0
        meta["expected_eps"] = exp_eps or 0.0
        
        # Formulate explicit corporate performance surprise calculations
        if exp_eps and abs(exp_eps) > 0:
            raw_surprise = (rep_eps - exp_eps) / abs(exp_eps)
            meta["earnings_surprise"] = max(min(raw_surprise, 1.0), -1.0)
            
        meta["eps_status"] = f"Reported EPS ${rep_eps:.2f} vs Expected ${exp_eps:.2f}"
        meta["revenue_status"] = f"Reported Rev: {rep_rev} vs Expected: {exp_rev}"
        
    if sent_row:
        avg_sent, volume = sent_row
        meta["avg_sentiment"] = avg_sent or 0.0
        meta["news_volume"] = int(volume or 0)
        
        # Confidence Multiplier calculation to prevent isolated narrative skewing
        confidence_mult = min(meta["news_volume"] / 5.0, 1.0)
        meta["media_buzz"] = meta["avg_sentiment"] * confidence_mult
        
    return meta

def main():
    print("="*90)
    print(" SMARTER MARKET SCANNER: TWO-PASS MATRIX PROCESSING & JUSTIFICATION PIPELINE")
    print("="*90)
    
    # Load standardized configuration layout settings profile
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")

    config = load_system_config()
    sys_settings = config["system_settings"]
    scan_settings = config["scanner_settings"]
    weights = config["alpha_ranking_weights"]
    
    db_name = sys_settings.get("db_name", "trading_vault.db")
    tickers_universe = config.get("universe", {}).get("tickers", [])
    if not tickers_universe:
        tickers_universe = config.get("trading_universe", {}).get("tickers", [])
        
    # Instantiate modular core computing entities
    scanner = VectorizedScannerEngine(atr_lookback=config["risk_management"]["atr_lookback"])
    
    # ---------------------------------------------------------------------
    # PASS 1: RUN HIGH-PERFORMANCE VECTORIZED DAILY MATRIX SCANNER
    # ---------------------------------------------------------------------
    print(f"[*] Executing Pass 1 Macro Analysis over [{len(tickers_universe)}] watchlist records...")
    df_daily = load_macro_daily_data(db_name, tickers_universe)
    
    if df_daily.empty:
        print("[CRITICAL] Historical tables are unpopulated. Pipeline run terminated.")
        return
        
    _, shortlist_signals = scanner.compute_daily_matrices(df_daily)
    print(f"[+] Pass 1 Complete. Matrix tracking flagged [{len(shortlist_signals)}] candidates matching setups.")
    
    if not shortlist_signals:
        print("[+] Scan finished. Zero tickers triggered setups.")
        return

    # ---------------------------------------------------------------------
    # PASS 2 & 3: MICRO CONFLUENCE & SENTIMENT ENRICHMENT SCORING FRAMEWORK
    # ---------------------------------------------------------------------
    ranked_candidates = []
    print("\n[*] Processing Shortlist Confluence Routines & Fundamental Alpha Extraction...")
    
    for ticker, info in shortlist_signals.items():
        # Fetch related structural database context frames
        meta = fetch_alpha_metadata(db_name, ticker)
        
        # Formula Overlay: Composite Tailwind Score = (Earnings Surprise * 0.5) + (Media Buzz * 0.5)
        composite_tailwind_score = (meta["earnings_surprise"] * 0.5) + (meta["media_buzz"] * 0.5)
        
        # Request Polygon hourly bar structures on-demand safely
        df_hourly = fetch_on_demand_hourly_data(ticker, scan_settings["micro_lookback_days"], api_key)
        confluence_passed, confluence_msg = scanner.verify_micro_confluence(df_hourly)
        
        if not confluence_passed:
            continue  # Safe structural drop if lower timeframe timing is messy
            
        # Normalize structural matrix parameters into a scaled 0-100 system allocation metric
        tech_score = (info["total_setups"] / 9.0) * 100.0
        mom_score = max(min(info["roc_val"] * 5.0, 100.0), -100.0)  # Standardized clamp boundaries
        sent_score = ((composite_tailwind_score + 1.0) / 2.0) * 100.0  # Normalize -1 to +1 range onto 0-100 scale
        
        # Calculate final structured Alpha Score using weighted configurations profiles
        final_alpha_score = (
            (tech_score * (weights["strategy_confluence_pct"] / 100.0)) +
            (mom_score * (weights["momentum_velocity_pct"] / 100.0)) +
            (sent_score * (weights["sentiment_tailwind_pct"] / 100.0))
        )
        
        ranked_candidates.append({
            "ticker": ticker,
            "alpha_score": final_alpha_score,
            "tailwind_score": composite_tailwind_score,
            "info": info,
            "meta": meta,
            "confluence_msg": confluence_msg
        })
        
    # Sort entire active list descending using final computed Alpha Score matrix values
    ranked_candidates = sorted(ranked_candidates, key=lambda x: x["alpha_score"], reverse=True)
    final_limit = scan_settings.get("max_shortlist_size", 25)
    selected_targets = ranked_candidates[:final_limit]

    # ---------------------------------------------------------------------
    # HIGH-FIDELITY TRADE JUSTIFICATION ENGINE REPORT OUTPUT
    # ---------------------------------------------------------------------
    print("\n" + "="*90)
    print(" INSTITUTIONAL HIGH-FIDELITY TRADE JUSTIFICATION MATRIX ENGINE REPORT")
    print("="*90)
    
    for rank, target in enumerate(selected_targets, 1):
        t_ticker = target["ticker"]
        t_info = target["info"]
        t_meta = target["meta"]
        sentiment_status = "BULLISH" if t_meta["avg_sentiment"] > 0.10 else "BEARISH" if t_meta["avg_sentiment"] < -0.10 else "NEUTRAL"
        
        print(f"\n[RANK #{rank} TARGET ACQUISITION LOCK: {t_ticker.upper()}]")
        print(f"  System Alpha Ranking Score: {target['alpha_score']:.2f} | DEFENSIVE COMPOSITE TAILWIND MATRIX SCORE: {target['tailwind_score']:.4f}")
        print(f"  Daily Close Price: ${t_info['close']:.2f} | Dynamic Stop Range (2x ATR): ${t_info['atr']*2:.2f}")
        print("-" * 75)
        print("  Precise Mathematical & Geometric Trade Logic Justification:")
        
        for justification in t_info["justifications"]:
            print(f"   -> {justification}")
            
        print("\n  Lower Timeframe Execution Confluence Metrics:")
        print(f"   [Micro Confirmation]: {target['confluence_msg']}")
        
        print("\n  Fundamental & Narrative Alpha Overlay Metrics:")
        print(f"   - Corporate Earnings Matrix: {t_meta['eps_status']} (Surprise Ratio: {t_meta['earnings_surprise']*100:+.1f}%)")
        print(f"   - Corporate Revenue Status: {t_meta['revenue_status']}")
        print(f"   - Alternative NLP News Ingestion: Media Tone: {t_meta['avg_sentiment']:+.3f} [{sentiment_status} via {t_meta['news_volume']} Headlines]")
        print("="*90)

    if not selected_targets:
        print("\n[+] Scan finished. Zero tickers survived filters today.")
    else:
        print(f"\n[+] Pipeline run completed successfully. Identified [{len(selected_targets)}] institutional candidates.")

if __name__ == "__main__":
    main()