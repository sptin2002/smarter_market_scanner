import os
import sys
import json
import sqlite3
import logging
import asyncio
import pandas as pd
import numpy as np
import time
from transformers import BertTokenizer
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from transformers import AutoTokenizer
import onnxruntime as ort
from dotenv import load_dotenv

# Import tqdm for clean, single-line terminal progress animations
from tqdm import tqdm

# Abstracted Data Pipeline Ingestion Isolation Factory
from data_factory import HistoricalDataFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PEAD_System.PEAD_Data_Initializer")
LOCAL_FINBERT_DIR = "finbert-local"

class DataInitializationEngine:
    def __init__(self, config_path: str = "config.json"):
        # Root environmental initialization configuration parameters
        load_dotenv()
        self.config = self._load_config(config_path)
        
        sys_settings = self.config.get("system_settings", {})
        self.db_name = sys_settings.get("db_name", "trading_vault.db")
        self.ib_port = sys_settings.get("ibkr_port", 7497)

        # Core isolation injection initialization 
        self.data_factory = HistoricalDataFactory(self.config)
        self._initialize_vault_schema()
        
        # Initialize the Tokenizer and ONNX Session once at startup for optimal performance
        # self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_FINBERT_DIR, local_files_only=True, use_fast=True)
        self.tokenizer = BertTokenizer.from_pretrained(LOCAL_FINBERT_DIR, local_files_only=True)
        
        onnx_model_path = os.path.join(LOCAL_FINBERT_DIR, "finbert.onnx")
        if not os.path.exists(onnx_model_path):
            logger.error(f"[CRITICAL] ONNX model file not found at {onnx_model_path}. Please export the model first.")
            sys.exit(1)
            
        # Determine execution providers (Use GPU via CUDA if available, fallback to CPU)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
        logger.info(f"[+] ONNX Runtime session initialized successfully using providers: {self.ort_session.get_providers()}")

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
                    news_volume INTEGER DEFAULT -1,
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
        """Fetches and caches macro daily candlestick data maps using an active single-line progress bar."""
        logger.info("[*] Starting universe baseline price sync sequence...")
        tickers = self.config.get("trading_universe", {}).get("tickers", [])
        
        if not tickers:
            logger.warning("[!] Config file trading_universe array is empty.")
            return

        conn = sqlite3.connect(self.db_name)
        
        # Wrap the ticker sequence with tqdm to render a beautiful real-time progress layout
        with tqdm(tickers, desc="Price Sync Progress", unit="ticker") as pbar:
            for symbol in pbar:
                # Dynamically update the left-hand text label on the progress bar
                pbar.set_description(f"Syncing Prices -> {symbol}")
                
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(date) FROM ticker_data WHERE ticker = ?", (symbol,))
                res = cursor.fetchone()
                
                end_date_str = datetime.today().strftime('%Y-%m-%d')
                
                if res and res[0]:
                    latest_date = datetime.strptime(res[0], '%Y-%m-%d')
                    start_date_str = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
                    
                    if start_date_str > end_date_str:
                        continue  # Skipped silently to keep your terminal perfectly organized
                else:
                    start_date_str = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

                try:
                    df_daily = await self.data_factory.fetch_daily_bars(symbol, start_date=start_date_str, end_date=end_date_str)
                    if not df_daily.empty:
                        payload = []
                        for _, row in df_daily.iterrows():
                            date_str = str(row['date'])[:10]
                            payload.append((
                                date_str,
                                symbol,
                                float(row['open']),
                                float(row['high']),
                                float(row['low']),
                                float(row['close']),
                                int(row['volume'])
                            ))
                        
                        conn.executemany("""
                            INSERT OR REPLACE INTO ticker_data (date, ticker, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, payload)
                        conn.commit()
                except Exception as e:
                    # pbar.write prevents text collision or graphical glitched duplicates in the terminal
                    pbar.write(f" [ERROR] Failed to ingest price profile rows for asset {symbol}: {e}")
                    
        conn.close()

    def sync_company_fundamentals(self):
        """Extracts corporate accounting data with structural variable safeguards and progress bars."""
        logger.info("[*] Initializing fundamental balance sheet matrix sync...")
        tickers = self.config.get("trading_universe", {}).get("tickers", [])
        
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        try:
            with tqdm(tickers, desc="Fundamentals Sync Progress", unit="ticker") as pbar:
                for symbol in pbar:
                    pbar.set_description(f"Syncing Fundamentals -> {symbol}")
                    
                    cursor.execute("SELECT COUNT(*) FROM ticker_fundamentals WHERE ticker = ?", (symbol,))
                    if cursor.fetchone()[0] > 0:
                        continue
                        
                    try:
                        df_fund = self.data_factory.fetch_company_fundamentals(symbol)
                        
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
                            
                            # Mandatory pacing interval to comply with standard rate metrics
                            time.sleep(12.0)
                        else:
                            pbar.write(f"     [!] Empty financial framework parsed from provider for asset {symbol}")
                            
                    except Exception as e:
                        pbar.write(f"     [ERROR] Fundamental data collection aborted for asset {symbol}: {e}")

        except KeyboardInterrupt:
            logger.warning("\n[!!!] Control-C Detected: Aborting synchronization loop cleanly...")
            try:
                conn.rollback() 
                conn.close()
                self.data_factory.close()
            except Exception:
                pass
            logger.info("[+] Active databases and network connections terminated safely. System exiting.")
            sys.exit(1)  
        conn.close()

    def compute_vector_sentiment_batch(self, headlines: List[str], batch_size: int = 128) -> List[float]:
        """Runs accelerated parallel pipeline batches using native Python chunks and ONNX Runtime."""
        if not headlines:
            return []
            
        all_scores = []
        
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch, 
                padding="max_length", 
                truncation=True, 
                max_length=128, 
                return_tensors="np"
            )

            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            
            if "token_type_ids" in inputs:
                ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

            ort_outputs = self.ort_session.run(["logits"], ort_inputs)
            logits = ort_outputs[0]
            
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            scores = (probs[:, 0] - probs[:, 1]).tolist()
            all_scores.extend(scores)
                
        return all_scores

    def enrich_sentiment_pipeline(self):
        """Processes historical news metadata loops with a compact real-time progress bar."""
        logger.info("[*] Initializing FinBERT NLP sentiment classification engine...")
        tickers = self.config.get("trading_universe", {}).get("tickers", [])
        
        conn = sqlite3.connect(self.db_name)
        
        with tqdm(tickers, desc="Sentiment Ingestion Progress", unit="ticker") as pbar:
            for symbol in pbar:
                pbar.set_description(f"Running NLP Engine -> {symbol}")
                
                work_df = pd.read_sql(
                    f"SELECT date FROM ticker_data WHERE ticker='{symbol}' AND news_volume=-1", conn
                )
                
                if work_df.empty:
                    continue
                    
                flattened_headlines = []
                date_indices = []
                
                for date_str in work_df['date']:
                    news_items = self.data_factory.fetch_news_headlines(symbol, date_str, date_str)
                    for item in news_items:
                        flattened_headlines.append(item["title"])
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
    asyncio.run(initializer.sync_ibkr_prices())
    initializer.sync_company_fundamentals()
    initializer.enrich_sentiment_pipeline()
    logger.info("Initialization Sequence Execution Cycle Complete.")