import os
import sys
import json
import sqlite3
import logging
import asyncio
import pandas as pd
import numpy as np
import torch
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

# Abstracted Data Pipeline Ingestion Isolation Factory
from data_factory import HistoricalDataFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PEAD_System.PEAD_Data_Initializer")
LOCAL_FINBERT_DIR = "finbert-local"

class HeadlineDataset(Dataset):
    """Efficient processing abstraction for deterministic sequence tokenization allocations."""
    def __init__(self, headlines: List[str], tokenizer: Any, max_length: int = 128):
        self.headlines = headlines
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int: 
        return len(self.headlines)
        
    def __getitem__(self, idx: int) -> str: 
        return self.headlines[idx]


class DataInitializationEngine:
    def __init__(self, config_path: str = "config.json"):
        # Root environmental initialization configuration parameters
        load_dotenv()
        self.config = self._load_config(config_path)
        
        sys_settings = self.config.get("system_settings", {})
        self.db_name = sys_settings.get("db_name", "trading_vault.db")
        self.ib_port = sys_settings.get("ibkr_port", 7497)
        
        # FIXED: Explicitly capture both keys into instance memory safely
        # self.poly_key = os.getenv("POLYGON_API_KEY")
        # self.av_key = os.getenv("ALPHA_VANTAGE_API_KEY") or self.poly_key

        # Core isolation injection initialization 
        self.data_factory = HistoricalDataFactory(self.config)
        self._initialize_vault_schema()

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return json.load(f)

    def _initialize_vault_schema(self):
        """Builds clean technical relational tables if missing from SQLite vault state."""
        conn = sqlite3.connect(self.db_name)
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticker_data (
                    date TEXT,
                    ticker TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    avg_sentiment REAL DEFAULT 0.0,
                    news_volume INTEGER DEFAULT 0,
                    PRIMARY KEY (date, ticker)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticker_fundamentals (
                    ticker TEXT,
                    announcement_date TEXT,
                    reported_eps REAL,
                    expected_eps REAL,
                    reported_rev REAL,
                    expected_rev REAL,
                    PRIMARY KEY (ticker, announcement_date)
                )
            """)
        conn.close()

    async def sync_ibkr_prices(self):
        """Fetches and caches macro daily candlestick data maps with active step logs."""
        logger.info("[*] Starting universe baseline price sync sequence...")
        tickers = self.config.get("trading_universe", {}).get("tickers", [])
        total = len(tickers)
        
        if not tickers:
            logger.warning("[!] Config file trading_universe array is empty.")
            return

        conn = sqlite3.connect(self.db_name)
        
        # UPGRADED: Added loop tracking metrics
        for idx, symbol in enumerate(tickers, 1):
            pct = (idx / total) * 100
            logger.info(f"[{idx}/{total}] ({pct:.1f}%) Synchronizing daily candlestick records for: {symbol}")
            
            # Check for existing records to leverage checkpoint safety
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ticker_data WHERE ticker = ?", (symbol,))
            if cursor.fetchone()[0] > 100:  # Skip if substantial baseline history is already loaded
                continue

            try:
                # Direct lookup using data factory interfaces
                df_daily = await self.data_factory.fetch_daily_bars(symbol, lookback_days=365)
                if not df_daily.empty:
                    payload = []
                    for _, row in df_daily.iterrows():
                        # Standardize datetime sorting parameters
                        date_str = row['timestamp'].strftime('%Y-%m-%d') if isinstance(row['timestamp'], datetime) else str(row['timestamp'])[:10]
                        payload.append((
                            date_str, symbol, float(row['open']), float(row['high']),
                            float(row['low']), float(row['close']), int(row['volume'])
                        ))
                    
                    conn.executemany("""
                        INSERT OR REPLACE INTO ticker_data (date, ticker, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, payload)
                    conn.commit()
            except Exception as e:
                logger.error(f"     [ERROR] Failed to ingest price profile rows for asset {symbol}: {e}")
                
        conn.close()

    def sync_company_fundamentals(self):
        """Extracts corporate accounting data with structural variable safeguards and progress logs."""
        logger.info("[*] Initializing fundamental balance sheet matrix sync...")
        tickers = self.config.get("trading_universe", {}).get("tickers", [])
        total = len(tickers)
        
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # UPGRADED: Added loop tracking metrics
        for idx, symbol in enumerate(tickers, 1):
            pct = (idx / total) * 100
            
            # Checkpoint lookup to see if data already exists
            cursor.execute("SELECT COUNT(*) FROM ticker_fundamentals WHERE ticker = ?", (symbol,))
            if cursor.fetchone()[0] > 0:
                logger.info(f"[{idx}/{total}] ({pct:.1f}%) -> {symbol} fundamentals already cached in local vault.")
                continue
                
            logger.info(f"[{idx}/{total}] ({pct:.1f}%) -> Fetching fresh fundamentals matrix for: {symbol}")
            
            try:
                df_fund = self.data_factory.fetch_ticker_fundamentals(symbol)
                
                if df_fund is not None and not df_fund.empty:
                    payload = []
                    for _, row in df_fund.iterrows():
                        payload.append((
                            symbol,
                            str(row['announcement_date']),
                            float(row['reported_eps']) if pd.notna(row['reported_eps']) else None,
                            float(row['expected_eps']) if pd.notna(row['expected_eps']) else None,
                            float(row['reported_rev']) if pd.notna(row['reported_rev']) else None,
                            float(row['expected_rev']) if pd.notna(row['expected_rev']) else None
                        ))
                    
                    cursor.executemany("""
                        INSERT OR REPLACE INTO ticker_fundamentals 
                        (ticker, announcement_date, reported_eps, expected_eps, reported_rev, expected_rev)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, payload)
                    conn.commit()
                    
                    # Mandatory pacing interval to comply with standard free-tier API rate metrics
                    time.sleep(12.0)
                else:
                    logger.warning(f"     [!] Empty financial framework parsed from provider for asset {symbol}")
                    
            except Exception as e:
                logger.error(f"     [ERROR] Fundamental data collection aborted for asset {symbol}: {e}")
                
        conn.close()

    def compute_vector_sentiment_batch(self, headlines: List[str], batch_size: int = 128) -> List[float]:
        """Runs parallel GPU/CPU pipeline batches using optimized Dataset tensors."""
        if not headlines:
            return []
            
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_FINBERT_DIR, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_FINBERT_DIR, local_files_only=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        dataset = HeadlineDataset(headlines, tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        all_scores = []
        with torch.no_grad():
            for batch in loader:
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Formula metrics map: Positive Probability minus Negative Probability
                scores = (probs[:, 0] - probs[:, 1]).cpu().numpy().tolist()
                all_scores.extend(scores)
                
        return all_scores

    def enrich_sentiment_pipeline(self):
        """Processes historical news metadata loops with step status counters."""
        logger.info("[*] Initializing FinBERT NLP sentiment classification engine...")
        tickers = self.config.get("trading_universe", {}).get("tickers", [])
        total = len(tickers)
        
        conn = sqlite3.connect(self.db_name)
        
        # UPGRADED: Added loop tracking metrics
        for idx, symbol in enumerate(tickers, 1):
            pct = (idx / total) * 100
            
            work_df = pd.read_sql(
                f"SELECT date FROM ticker_data WHERE ticker='{symbol}' AND news_volume=0", conn
            )
            
            if work_df.empty:
                logger.info(f"[{idx}/{total}] ({pct:.1f}%) -> Sentiment matrices for {symbol} are already up to date.")
                continue
                
            logger.info(f"[{idx}/{total}] ({pct:.1f}%) -> Tokenizing and vectorizing text fields for: {symbol}")
            
            flattened_headlines = []
            date_indices = []
            
            # Request endpoints directly to structural dictionary files
            for date_str in work_df['date']:
                # Pull raw headlines using unified factory connection points
                headlines = self.data_factory.fetch_raw_news_titles(symbol, date_str, date_str)
                for t in headlines:
                    flattened_headlines.append(t)
                    date_indices.append(date_str)
            
            mapped_results: Dict[str, List[float]] = {}
            if flattened_headlines:
                scores = self.compute_vector_sentiment_batch(flattened_headlines, batch_size=128)
                for d_idx, score in zip(date_indices, scores):
                    mapped_results.setdefault(d_idx, []).append(score)
                    
            update_payload = []
            for date_str in work_df['date']:
                day_scores = mapped_results.get(date_str, [])
                avg_sentiment = float(np.mean(day_scores)) if day_scores else 0.0
                update_payload.append((avg_sentiment, len(day_scores), symbol, date_str))
                
            if update_payload:
                conn.executemany(
                    "UPDATE ticker_data SET avg_sentiment=?, news_volume=? WHERE ticker=? AND date=?", 
                    update_payload
                )
                conn.commit()
                
        conn.close()
        self.data_factory.close()
        logger.info("[+] Sentiment Enrichment Sequence Finished Cleanly.")


if __name__ == "__main__":
    load_dotenv()
    
    initializer = DataInitializationEngine(config_path="config.json")
    # asyncio.run(initializer.sync_ibkr_prices())
    initializer.sync_company_fundamentals()
    # initializer.enrich_sentiment_pipeline()
    logger.info("Initialization Sequence Execution Cycle Complete.")