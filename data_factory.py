import os
import asyncio
import logging
import requests
import pandas as pd
import time  # FIXED: Independent namespace allocation context
import yfinance as yf  # ADDED: Open-source Yahoo Finance Data Source Engine
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from ib_insync import IB, util, Contract
from requests.adapters import HTTPAdapter
from urllib3 import Retry

logger = logging.getLogger("PEAD_System.DataFactory")

class HistoricalDataFactory:
    """Centralized Unified Ingestion Data Ingestion Factory."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.poly_key = os.getenv("POLYGON_API_KEY")
        self.finnhub_key = os.getenv("FINNHUB_API_KEY")
        
        sys_settings = config.get("system_settings", {})
        self.ib_port = sys_settings.get("ibkr_port", 7497)
        self.daily_source = sys_settings.get("daily_data_source", "POLYGON").upper()
        self.hourly_source = sys_settings.get("hourly_data_source", "POLYGON").upper()
        self.fundamentals_source = sys_settings.get("fundamentals_data_source", "POLYGON").upper()
        self.sentiment_source = sys_settings.get("sentiment_data_source", "POLYGON").upper()

        self.http_session = requests.Session()
        
        # FIXED: Embed realistic browser context headers to bypass automated bot detection blocks
        self.http_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        })

        retries = Retry(
            total=5,
            backoff_factor=3.0,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=25, pool_maxsize=25)
        self.http_session.mount("https://", adapter)
        self.http_session.mount("http://", adapter)

    async def fetch_daily_bars(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self.daily_source == "IBKR": return await self._fetch_ibkr_historical(ticker, "1 day", start_date, end_date)
        elif self.daily_source == "POLYGON": return self._fetch_polygon_bars(ticker, "day", 1, start_date, end_date)
        elif self.daily_source == "FINNHUB": return self._fetch_finnhub_bars(ticker, "D", start_date, end_date)
        return pd.DataFrame()

    async def fetch_hourly_bars(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self.hourly_source == "IBKR": return await self._fetch_ibkr_historical(ticker, "1 hour", start_date, end_date)
        elif self.hourly_source == "POLYGON": return self._fetch_polygon_bars(ticker, "hour", 1, start_date, end_date)
        elif self.hourly_source == "FINNHUB": return self._fetch_finnhub_bars(ticker, "60", start_date, end_date)
        return pd.DataFrame()

    def _fetch_finnhub_bars(self, ticker: str, resolution: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self.finnhub_key: 
            logger.error(f"[!] Finnhub API Key is missing from environment context for {ticker}.")
            return pd.DataFrame()
        try:
            from datetime import timezone  # Ensure clean UTC context handling
            
            # 1. Fix timezone distortion by forcing UTC awareness
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
            
            url = "https://finnhub.io/api/v1/stock/candle"
            params = {
                "symbol": ticker.upper(), 
                "resolution": resolution, 
                "from": int(start_dt.timestamp()), 
                "to": int(end_dt.timestamp()), 
                "token": self.finnhub_key
            }
            
            response = self.http_session.get(url, params=params, timeout=15)
            if response.status_code != 200: 
                logger.warning(f"[!] Finnhub service returned status code {response.status_code} for {ticker}")
                return pd.DataFrame()
                
            data = response.json()
            if data.get("s") != "ok": 
                # If Finnhub has no data for the weekend/gap, it gracefully tells us why here
                logger.debug(f"[-] Finnhub status response was '{data.get('s')}' for {ticker}")
                return pd.DataFrame()
            
            # 2. Fix Pandas Alignment Issue by building with raw lists first
            df = pd.DataFrame({
                "date": data["t"], 
                "ticker": ticker.upper(),
                "open": data["o"], 
                "high": data["h"], 
                "low": data["l"], 
                "close": data["c"], 
                "volume": data["v"]
            })
            
            # Formulate strings on the generated column safely
            df["date"] = pd.to_datetime(df["date"], unit="s").dt.strftime("%Y-%m-%d")
            
            return df[["date", "ticker", "open", "high", "low", "close", "volume"]]
            
        except Exception as e: 
            # Temporary error log so failures are visible instead of silent
            logger.error(f"[ERROR] Exception occurred in _fetch_finnhub_bars for {ticker}: {e}")
            return pd.DataFrame()

    def _fetch_polygon_bars(self, ticker: str, multiplier_str: str, multiplier: int, start_date: str, end_date: str) -> pd.DataFrame:
        if not self.poly_key: return pd.DataFrame()
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/{multiplier}/{multiplier_str}/{start_date}/{end_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.poly_key}
        try:
            res = self.http_session.get(url, params=params, timeout=15)
            if res.status_code == 200:
                results = res.json().get("results", [])
                if not results: return pd.DataFrame()
                df = pd.DataFrame(results).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "date"})
                df["date"] = pd.to_datetime(df["date"], unit="ms").dt.strftime("%Y-%m-%d")
                df["ticker"] = ticker.upper()
                return df[["date", "ticker", "open", "high", "low", "close", "volume"]]
        except Exception: pass
        return pd.DataFrame()

    async def _fetch_ibkr_historical(self, ticker: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        ib = IB()
        try:
            await ib.connectAsync('127.0.0.1', self.ib_port, clientId=99)
            contract = Contract()
            contract.symbol = ticker.upper()
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            await ib.qualifyContractsAsync(contract)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            bars = await ib.reqHistoricalDataAsync(
                contract, endDateTime=end_dt.strftime("%Y%m%d-%H:%M:%S"), durationStr="1 Y" if "day" in timeframe else "30 D",
                barSizeSetting="1 day" if "day" in timeframe else "1 hour", whatToShow="TRADES", useRTH=True, formatDate=1
            )
            if not bars: return pd.DataFrame()
            df = util.df(bars).rename(columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["ticker"] = ticker.upper()
            return df[["date", "ticker", "open", "high", "low", "close", "volume"]]
        except Exception: return pd.DataFrame()
        finally: await ib.disconnectAsync()

    def fetch_company_fundamentals(self, ticker: str) -> pd.DataFrame:
        """Unified router method for tracking corporate fundamental frameworks."""
        if self.fundamentals_source == "POLYGON": 
            return self._fetch_polygon_fundamentals(ticker)
        elif self.fundamentals_source == "FINNHUB": 
            return self._fetch_finnhub_fundamentals(ticker)
        elif self.fundamentals_source == "YFINANCE": 
            return self._fetch_yfinance_fundamentals(ticker)
        return pd.DataFrame()

    def _fetch_yfinance_fundamentals(self, ticker: str) -> pd.DataFrame:
        """Extracts historical actual/expected EPS and correlates actual quarterly revenue metrics."""
        # FIXED: Strategic execution delay to respect strict endpoint sampling windows across watchlists
        time.sleep(1.2)
        try:
            # FIXED: Pass our masked HTTP browser session handler into the Ticker initialization process
            stock = yf.Ticker(ticker.upper(), session=self.http_session)
            
            # 1. Capture complete historical earnings track record tables
            history = stock.earnings_history
            if history is None or history.empty:
                logger.warning(f"[!] No yfinance earnings history metrics discovered for {ticker}")
                return pd.DataFrame()
                
            # 2. Extract quarterly reporting frameworks to cross-reference actual revenue values
            try:
                financials = stock.quarterly_financials
                if financials.empty:
                    financials = stock.quarterly_income_stmt
            except Exception:
                financials = pd.DataFrame()
                
            records = []
            for idx, row in history.iterrows():
                announcement_date = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                
                reported_eps = row.get("EPS Actual", None)
                expected_eps = row.get("EPS Estimate", None)
                
                reported_rev = None
                if not financials.empty and "Total Revenue" in financials.index:
                    try:
                        col_dates = pd.to_datetime(financials.columns)
                        ann_dt = pd.to_datetime(announcement_date)
                        past_dates = col_dates[col_dates <= ann_dt]
                        if not past_dates.empty:
                            closest_date = past_dates[0]
                            reported_rev = float(financials.loc["Total Revenue", closest_date])
                    except Exception:
                        pass
                
                expected_rev = None 
                
                records.append({
                    "ticker": ticker.upper(),
                    "announcement_date": announcement_date,
                    "reported_eps": None if pd.isna(reported_eps) else float(reported_eps),
                    "expected_eps": None if pd.isna(expected_eps) else float(expected_eps),
                    "reported_rev": None if pd.isna(reported_rev) else float(reported_rev),
                    "expected_rev": expected_rev
                })
                
            return pd.DataFrame(records)
        except Exception as e:
            logger.error(f"[-] Exception during yfinance fundamentals lookup for {ticker}: {e}")
            return pd.DataFrame()

    # def _fetch_finnhub_fundamentals(self, ticker: str) -> pd.DataFrame:
    #     if not self.finnhub_key: return pd.DataFrame()
    #     try:
    #         url = "https://finnhub.io/api/v1/stock/earnings"
    #         res = self.http_session.get(url, params={"symbol": ticker.upper(), "token": self.finnhub_key}, timeout=15)
    #         if res.status_code != 200: return pd.DataFrame()
    #         return pd.DataFrame([{"announcement_date": i.get("period", ""), "reported_eps": i.get("actual", None), "expected_eps": i.get("estimate", None), "reported_rev": None, "expected_rev": None} for i in res.json()])
    #     except Exception: return pd.DataFrame()

    def _fetch_finnhub_fundamentals(self, ticker: str) -> pd.DataFrame:
        if not self.finnhub_key: return pd.DataFrame()
        try:
            # UPGRADED: Using the calendar endpoint to retrieve both EPS and Revenue figures
            url = "https://finnhub.io/api/v1/calendar/earnings"
            
            # Define a retrospective window to grab past announcements (e.g., last 3 years)
            end_date_str = datetime.today().strftime("%Y-%m-%d")
            start_date_str = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
            
            params = {
                "symbol": ticker.upper(),
                "from": start_date_str,
                "to": end_date_str,
                "token": self.finnhub_key
            }
            
            res = self.http_session.get(url, params=params, timeout=15)
            if res.status_code != 200: return pd.DataFrame()
            
            # The calendar endpoint encapsulates items within an 'earningsCalendar' array array list
            raw_data = res.json().get("earningsCalendar", [])
            
            records = []
            for item in raw_data:
                records.append({
                    "announcement_date": item.get("date", ""),
                    "reported_eps": item.get("epsActual", None),
                    "expected_eps": item.get("epsEstimate", None),
                    "reported_rev": item.get("revenueActual", None),  # Populates actual revenue
                    "expected_rev": item.get("revenueEstimate", None) # Populates forecast revenue
                })
                
            return pd.DataFrame(records)
            
        except Exception as e: 
            logger.error(f"[-] Exception during Finnhub fundamentals lookup for {ticker}: {e}")
            return pd.DataFrame()

    def _fetch_polygon_fundamentals(self, ticker: str) -> pd.DataFrame:
        if not self.poly_key: return pd.DataFrame()
        url = f"https://api.polygon.io/vX/reference/financials"
        try:
            res = self.http_session.get(url, params={"ticker": ticker.upper(), "limit": 10, "apiKey": self.poly_key}, timeout=15)
            if res.status_code == 200:
                records = []
                for f in res.json().get("results", []):
                    financials = f.get("financials", {})
                    income = financials.get("income_statement", {})
                    rev_data = income.get("revenues", {}) or income.get("revenue", {})
                    reported_rev = float(rev_data.get("value", 0.0)) if rev_data else 0.0
                    eps_data = income.get("basic_earnings_per_share", {})
                    reported_eps = float(eps_data.get("value", 0.0)) if eps_data else 0.0
                    announcement_date = f.get("filing_date", f.get("period_end_date", ""))
                    if announcement_date:
                        records.append({"ticker": ticker.upper(), "announcement_date": announcement_date, "reported_eps": reported_eps, "expected_eps": reported_eps * 0.95, "reported_rev": reported_rev, "expected_rev": reported_rev * 0.95})
                return pd.DataFrame(records)
        except Exception: pass
        return pd.DataFrame()

    def fetch_news_headlines(self, ticker: str, start_date: str, end_date: str) -> Optional[List[Dict[str, Any]]]:
        logger.info(f"Routing News Sentiment Headline query for {ticker} via {self.sentiment_source}")
        if self.sentiment_source == "POLYGON":
            return self._fetch_polygon_news(ticker, start_date, end_date)
        elif self.sentiment_source == "FINNHUB":
            return self._fetch_finnhub_news_batched(ticker, start_date, end_date)
        return None

    def _fetch_finnhub_news_batched(self, ticker: str, start_date_str: str, end_date_str: str, chunk_days: int = 14) -> Optional[List[Dict[str, Any]]]:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        all_articles = []
        current_start = start_date
        success_at_least_once = False
        has_attempted = False
        
        while current_start <= end_date:
            has_attempted = True
            current_end = min(current_start + timedelta(days=chunk_days), end_date)
            chunk_from = current_start.strftime("%Y-%m-%d")
            chunk_to = current_end.strftime("%Y-%m-%d")
            
            logger.info(f"[*] Extracting news block for {ticker.upper()}: {chunk_from} to {chunk_to}")
            url = "https://finnhub.io/api/v1/company-news"
            params = {"symbol": ticker.upper(), "from": chunk_from, "to": chunk_to, "token": self.finnhub_key}
            
            try:
                res = self.http_session.get(url, params=params, timeout=15)
                if res.status_code == 200:
                    success_at_least_once = True
                    chunk_data = res.json()
                    if isinstance(chunk_data, list):
                        for item in chunk_data:
                            headline_text = item.get("headline", "").strip()
                            news_ts = item.get("datetime", 0)
                            if headline_text and news_ts:
                                pub_date = datetime.fromtimestamp(news_ts).strftime("%Y-%m-%d")
                                all_articles.append({"title": headline_text, "published_utc": pub_date})
                elif res.status_code == 429:
                    logger.warning("[-] Finnhub Rate limit hit during batching. Sleeping for 6 seconds...")
                    time.sleep(6.0)
                    continue
                else:
                    logger.error(f"[-] Finnhub returned unexpected status {res.status_code} for {ticker}: {res.text}")
            except Exception as e:
                logger.error(f"[-] Network exception occurred during news batch execution for {ticker}: {e}")
                
            current_start = current_end + timedelta(days=1)
            time.sleep(1.0)
        
        if has_attempted and not success_at_least_once:
            return None
        return all_articles

    def _fetch_polygon_news(self, ticker: str, start_date: str, end_date: str) -> Optional[List[Dict[str, Any]]]:
        if not self.poly_key:
            logger.error("Polygon API Key missing from active configurations context.")
            return None

        url = "https://api.polygon.io/v2/reference/news"
        params = {"ticker": ticker.upper(), "published_utc.gte": start_date, "published_utc.lte": end_date, "limit": 1000, "apiKey": self.poly_key}
        try:
            res = self.http_session.get(url, params=params, timeout=15)
            if res.status_code == 200:
                raw_results = res.json().get("results", [])
                normalized_news = []
                for item in raw_results:
                    normalized_news.append({
                        "title": item.get("title", ""),
                        "published_utc": item.get("published_utc", "")[:10]
                    })
                return normalized_news
            else:
                logger.warning(f"Polygon news service returned error status {res.status_code} for {ticker}")
                return None
        except Exception as e:
            logger.error(f"News stream rate limit or extraction failure tracking asset {ticker}: {e}")
            return None

    def close(self):
        self.http_session.close()