import os
import sys
import json
import time
import sqlite3
import logging
import asyncio
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
from ib_insync import IB, util, Contract
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dotenv import load_dotenv

# Global Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PEAD_System.PEAD_Data_Initializer")
LOCAL_FINBERT_DIR = "finbert-local"

class HeadlineDataset(Dataset):
    """Efficient PyTorch Dataset wrapper for batch tokenization."""
    def __init__(self, headlines: List[str], tokenizer: Any, max_length: int = 128):
        self.headlines = headlines
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int: return len(self.headlines)
    def __getitem__(self, idx: int) -> str: return self.headlines[idx]


class DataInitializationEngine:
    def __init__(self, config_path: str = "config.json"):
        load_dotenv()
        self.poly_key = os.getenv("POLYGON_API_KEY")
        self.config = self._load_config(config_path)
        
        sys_settings = self.config.get("settings", self.config.get("system_settings", {}))
        self.db_name = sys_settings.get("db_name", "trading_vault.db")
        self.ib_port = sys_settings.get("ibkr_port", 7497)
        
        if not self.poly_key:
            raise KeyError("Critical Configuration Error: Polygon API Key is missing under 'api_keys.polygon'.")
        
        # Unify watchlist and universe array maps
        watchlist_tickers = self.config.get("watchlist", self.config.get("trading_watchlist", {}).get("tickers", []))
        universe_tickers = self.config.get("universe", self.config.get("trading_universe", {}).get("tickers", []))
        equity_tickers = list(set(watchlist_tickers + universe_tickers))
        
        macro_tickers = list(set(self.config.get("macro_symbols", self.config.get("macro_regime_symbols", []))))
        
        self.universe_targets: List[Tuple[str, str, str]] = []
        for ticker in equity_tickers: 
            self.universe_targets.append((ticker, "STK", "SMART"))
        for ticker in macro_tickers: 
            # Differentiate Index contracts if necessary, default to IND for macro listings like SPX/VIX
            sec_type = "IND" if ticker in ["SPX", "VIX", "COMP"] else "STK"
            exchange = "CBOE" if sec_type == "IND" else "SMART"
            self.universe_targets.append((ticker, sec_type, exchange))

        self.lookback_years = 5
        self.today_str = datetime.now().strftime('%Y-%m-%d')
        self.default_start_date = (datetime.now() - timedelta(days=self.lookback_years * 365.25)).strftime('%Y-%m-%d')

        self._init_db_schema()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Sentiment Inference Hardware Target Lock: [{self.device.upper()}]")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_FINBERT_DIR, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(LOCAL_FINBERT_DIR, local_files_only=True)
            self.model.to(self.device).eval()
        except Exception as e:
            logger.error(f"Failed to load local FinBERT models from '{LOCAL_FINBERT_DIR}': {e}")
            raise

    def _load_config(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration profile missing from execution route: {path}")
        with open(path, 'r') as f: 
            return json.load(f)

    def _init_db_schema(self):
        """Initializes schema layout and enforces column migrations safely."""
        with sqlite3.connect(self.db_name) as conn:
            # Table A: Pure Market & Sentiment Time Series Data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticker_data (
                    date TEXT, ticker TEXT, open REAL, high REAL, low REAL, close REAL,
                    volume INTEGER, avg_sentiment REAL, news_volume INTEGER,
                    PRIMARY KEY (date, ticker)
                ) WITHOUT ROWID;
            """)
            
            # Defensive Migration Handling for Table B (Fundamentals)
            # Check if table exists and inspect its columns to avoid "no such column" crash
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(ticker_fundamentals);")
            columns = [col[1] for col in cursor.fetchall()]
            
            if columns and "date" in columns:
                logger.warning("Legacy schema 'date' found in ticker_fundamentals. Executing database migration...")
                # Safest approach for SQLite WITHOUT ROWID tables: drop and recreate to avoid structural fragmentation
                conn.execute("DROP TABLE ticker_fundamentals;")
                columns = []
            
            # Table B: Dedicated Normalized Corporate Financial Announcements
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticker_fundamentals (
                    announcement_date TEXT, ticker TEXT, reported_eps REAL,
                    expected_eps REAL, reported_rev REAL, expected_rev REAL,
                    PRIMARY KEY (announcement_date, ticker)
                ) WITHOUT ROWID;
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_td_ticker_date ON ticker_data (ticker, date ASC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tf_ticker_date ON ticker_fundamentals (ticker, announcement_date ASC);")
            conn.commit()
            logger.info("Relational database schemas validated and locked down successfully.")

    async def sync_ibkr_prices(self):
        """Phase 1: Incremental Price Syncing via IBKR with explicit progress metrics."""
        print("\n" + "="*80)
        print(" [PHASE 1/3] SYNCHRONIZING HISTORICAL PRICE DATA VIA IBKR")
        print("="*80)
        
        ib = IB()
        try:
            await asyncio.wait_for(ib.connectAsync('127.0.0.1', self.ib_port, clientId=77), timeout=15)
        except Exception as e:
            logger.critical(f"IBKR API Socket connection dropped or unavailable on port {self.ib_port}: {e}")
            sys.exit(1)

        conn = sqlite3.connect(self.db_name)
        
        for symbol, sec_type, exchange in tqdm(self.universe_targets, desc="Syncing OHLCV Market Data", unit="ticker"):
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(date) FROM ticker_data WHERE ticker=? AND close IS NOT NULL", (symbol,))
                max_date_row = cursor.fetchone()
                
                if max_date_row and max_date_row[0] is not None:
                    start_date_str = (datetime.strptime(max_date_row[0], '%Y-%m-%d') - timedelta(days=2)).strftime('%Y-%m-%d')
                    is_incremental = True
                else:
                    start_date_str = self.default_start_date
                    is_incremental = False

                if is_incremental and start_date_str >= self.today_str:
                    continue

                all_dates = pd.date_range(start=start_date_str, end=self.today_str).strftime('%Y-%m-%d').tolist()
                payload = [(d, symbol, -1, 0.0) for d in all_dates]
                conn.executemany(
                    "INSERT OR IGNORE INTO ticker_data (date, ticker, news_volume, avg_sentiment) VALUES (?, ?, ?, ?)", 
                    payload
                )
                conn.commit()

                days_needed = (datetime.now() - datetime.strptime(start_date_str, '%Y-%m-%d')).days + 1
                duration_str = f"{int(np.ceil(days_needed / 365.25))} Y" if days_needed > 365 else f"{max(days_needed, 1)} D"

                contract = Contract(symbol=symbol, secType=sec_type, exchange=exchange, currency='USD')
                bars = await ib.reqHistoricalDataAsync(contract, '', duration_str, '1 day', 'TRADES', True)
                if not bars: 
                    continue

                df_prices = util.df(bars)
                df_prices['date'] = pd.to_datetime(df_prices['date']).dt.strftime('%Y-%m-%d')

                update_payload = []
                for _, row in df_prices.iterrows():
                    update_payload.append((row['open'], row['high'], row['low'], row['close'], int(row['volume']), row['date'], symbol))

                conn.executemany("UPDATE ticker_data SET open=?, high=?, low=?, close=?, volume=? WHERE date=? AND ticker=?", update_payload)
                conn.commit()

                # Continuous Calendar Forward-Fill Logic for Non-Trading Days
                conn.execute("""
                    UPDATE ticker_data SET 
                    open = (SELECT close FROM ticker_data b2 WHERE b2.ticker=ticker_data.ticker AND b2.date < ticker_data.date AND b2.close IS NOT NULL ORDER BY b2.date DESC LIMIT 1),
                    high = (SELECT close FROM ticker_data b2 WHERE b2.ticker=ticker_data.ticker AND b2.date < ticker_data.date AND b2.close IS NOT NULL ORDER BY b2.date DESC LIMIT 1),
                    low = (SELECT close FROM ticker_data b2 WHERE b2.ticker=ticker_data.ticker AND b2.date < ticker_data.date AND b2.close IS NOT NULL ORDER BY b2.date DESC LIMIT 1),
                    close = (SELECT close FROM ticker_data b2 WHERE b2.ticker=ticker_data.ticker AND b2.date < ticker_data.date AND b2.close IS NOT NULL ORDER BY b2.date DESC LIMIT 1),
                    volume = 0 WHERE ticker=? AND date >= ? AND open IS NULL
                """, (symbol, start_date_str))
                conn.commit()
                await asyncio.sleep(0.02)
            except Exception as ex:
                logger.error(f"Error processing price update for {symbol}: {ex}")
                continue

        ib.disconnect()
        conn.close()

    def sync_polygon_fundamentals(self):
        """Phase 2: Syncs corporate earnings parameters with uniform column tracking."""
        print("\n" + "="*80)
        print(" [PHASE 2/3] HARVESTING FUNDAMENTAL ANNOUNCEMENT CALENDARS")
        print("="*80)
        
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        # Isolate equity listings needing earnings tracking
        target_stocks = [t for t in self.universe_targets if t[1] == "STK"]
        
        for symbol, _, _ in tqdm(target_stocks, desc="Querying Fundamental Filings", unit="ticker"):
            url = f"https://api.polygon.io/vX/reference/financials?ticker={symbol}&timeframe=quarterly&limit=100&apiKey={self.poly_key}"
            raw_records = []
            try:
                res = requests.get(url, timeout=15)
                if res.status_code == 429:
                    time.sleep(60)
                    res = requests.get(url, timeout=15)
                if res.status_code != 200: 
                    continue

                results = res.json().get("results", [])
                for report in results:
                    ann_date = report.get("filing_date")
                    if not ann_date: 
                        continue
                    ann_date = ann_date[:10]  # Standardize to YYYY-MM-DD
                    
                    financials = report.get("financials", {})
                    income_stmt = financials.get("income_statement", {})
                    
                    reported_eps = np.nan
                    reported_rev = np.nan
                    
                    # Unified key resolution
                    eps_keys = ["basic_earnings_per_share", "diluted_earnings_per_share", "earnings_per_share"]
                    for k in eps_keys:
                        node = income_stmt.get(k, {})
                        if node and node.get("value") is not None:
                            reported_eps = float(node.get("value"))
                            break

                    rev_keys = ["revenues", "revenue", "sales_revenue_net", "operating_revenues"]
                    for k in rev_keys:
                        node = income_stmt.get(k, {})
                        if node and node.get("value") is not None:
                            reported_rev = float(node.get("value"))
                            break

                    raw_records.append({
                        "announcement_date": ann_date, "reported_eps": reported_eps, "reported_rev": reported_rev,
                        "expected_eps": np.nan, "expected_rev": np.nan
                    })
                
                if not raw_records: 
                    continue

                # Align DataFrames and calculate proxy consensus baseline metrics
                df_calc = pd.DataFrame(raw_records).sort_values("announcement_date").reset_index(drop=True)
                df_calc['reported_eps'] = pd.to_numeric(df_calc['reported_eps']).bfill().fillna(0.0)
                df_calc['reported_rev'] = pd.to_numeric(df_calc['reported_rev']).bfill().fillna(0.0)
                
                eps_proxy = df_calc['reported_eps'].shift(1).rolling(window=4, min_periods=1).median()
                rev_proxy = df_calc['reported_rev'].shift(1).rolling(window=4, min_periods=1).median()
                
                df_calc['expected_eps'] = df_calc['expected_eps'].fillna(eps_proxy).fillna(df_calc['reported_eps'])
                df_calc['expected_rev'] = df_calc['expected_rev'].fillna(rev_proxy).fillna(df_calc['reported_rev'])

                db_payload = []
                for _, row in df_calc.iterrows():
                    db_payload.append((
                        row['announcement_date'], symbol,
                        float(row['reported_eps']), float(row['expected_eps']),
                        float(row['reported_rev']), float(row['expected_rev'])
                    ))

                # Safe execution against unified column names
                cursor.executemany("""
                    INSERT OR REPLACE INTO ticker_fundamentals 
                    (announcement_date, ticker, reported_eps, expected_eps, reported_rev, expected_rev)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, db_payload)
                conn.commit()
                time.sleep(0.02)

            except Exception as e:
                logger.error(f"Error parsing corporate records for {symbol}: {e}")
                continue

        conn.close()

    def fetch_news_bulk_chunks(self, ticker: str, start_date_str: str) -> Dict[str, List[str]]:
        daily_titles: Dict[str, List[str]] = {}
        adj_start = (datetime.strptime(start_date_str, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%dT16:00:00Z')
        url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&published_utc.gte={adj_start}&published_utc.lte={self.today_str}T23:59:59Z&limit=1000&apiKey={self.poly_key}"
        try:
            while url:
                response = requests.get(url, timeout=15)
                if response.status_code == 429:
                    time.sleep(60); continue
                if response.status_code != 200: break
                data = response.json()
                results = data.get('results', [])
                if not results: break

                for article in results:
                    if not article.get('title'): continue
                    dt_utc = datetime.fromisoformat(article['published_utc'].replace('Z', '+00:00'))
                    dt_est = dt_utc.astimezone(ZoneInfo("America/New_York"))
                    cutoff_time = dt_est.replace(hour=15, minute=45, second=0, microsecond=0)
                    date_str = (dt_est + timedelta(days=1)).strftime('%Y-%m-%d') if dt_est >= cutoff_time else dt_est.strftime('%Y-%m-%d')
                    daily_titles.setdefault(date_str, []).append(article['title'])

                next_url = data.get('next_url')
                url = f"{next_url}&apiKey={self.poly_key}" if next_url else None
                time.sleep(0.01)
            return daily_titles
        except Exception as e:
            logger.error(f"News failure for {ticker}: {e}"); return daily_titles

    def compute_vector_sentiment_batch(self, headlines: List[str], batch_size: int = 128) -> List[float]:
        if not headlines: return []
        dataset = HeadlineDataset(headlines, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_sentiments = []
        
        for batch in dataloader:
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiments = (probs[:, 0] - probs[:, 1]).cpu().numpy().tolist()
            all_sentiments.extend(sentiments)
        return all_sentiments

    def enrich_sentiment_pipeline(self):
        """Phase 3: Deep NLP parsing of alternative text sentiment arrays using FinBERT."""
        print("\n" + "="*80)
        print(" [PHASE 3/3] EXECUTING NATURAL LANGUAGE SENTIMENT PROCESSING")
        print("="*80)
        
        conn = sqlite3.connect(self.db_name)
        target_stocks = [t for t in self.universe_targets if t[1] == "STK"]
        
        for symbol, _, _ in tqdm(target_stocks, desc="Processing Sentiment Matrices", unit="ticker"):
            work_df = pd.read_sql_query("SELECT date FROM ticker_data WHERE ticker=? AND news_volume=-1 ORDER BY date ASC", conn, params=(symbol,))
            if work_df.empty: 
                continue
            
            min_needed_date = work_df['date'].min()
            raw_news_map = self.fetch_news_bulk_chunks(symbol, min_needed_date)
            
            flattened_headlines, date_indices = [], []
            for date_str in work_df['date']:
                for t in raw_news_map.get(date_str, []):
                    flattened_headlines.append(t)
                    date_indices.append(date_str)
            
            if flattened_headlines:
                computed_scores = self.compute_vector_sentiment_batch(flattened_headlines, batch_size=128)
                mapped_results: Dict[str, List[float]] = {}
                for d_idx, score in zip(date_indices, computed_scores):
                    mapped_results.setdefault(d_idx, []).append(score)
            else:
                mapped_results = {}
                
            update_payload = []
            for date_str in work_df['date']:
                scores = mapped_results.get(date_str, [])
                avg_sentiment = float(np.mean(scores)) if scores else 0.0
                update_payload.append((avg_sentiment, len(scores), symbol, date_str))
                
            try:
                conn.executemany("UPDATE ticker_data SET avg_sentiment=?, news_volume=? WHERE ticker=? AND date=?", update_payload)
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                logger.error(f"SQL update failure for {symbol}: {e}")
                
        conn.close()


if __name__ == "__main__":
    initializer = DataInitializationEngine(config_path="config.json")
    asyncio.run(initializer.sync_ibkr_prices())
    initializer.sync_polygon_fundamentals()
    initializer.enrich_sentiment_pipeline()
    logger.info("Initialization Sequence Finished Without Errors.")