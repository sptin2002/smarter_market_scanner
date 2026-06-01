import os
import asyncio
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from ib_insync import IB, util, Contract
from requests.adapters import HTTPAdapter
from urllib3 import Retry

logger = logging.getLogger("PEAD_System.DataFactory")

class HistoricalDataFactory:
    """
    Centralized Unified Data Ingestion Factory.
    Isolates extraction networks from system analytics logic. Handles both 
    Asynchronous stateful connections (IBKR) and stateless REST operations (Polygon).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.poly_key = os.getenv("POLYGON_API_KEY")
        sys_settings = config.get("system_settings", {})
        self.ib_port = sys_settings.get("ibkr_port", 7497)
        self.daily_source = sys_settings.get("daily_data_source", "IBKR").upper()
        self.hourly_source = sys_settings.get("hourly_data_source", "POLYGON").upper()

        # CONFIGURE A ROBUST, REUSABLE HTTP CONNECTION POOL WITH EXTENDED BACKOFF RETRIES
        self.http_session = requests.Session()
        retries = Retry(
            total=5,                # Up to 5 consecutive backoff recovery attempts
            backoff_factor=3.0,     # Curve: 3s, 6s, 12s, 24s, 48s. Perfectly clears 60s sliding rate windows!
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=True    # Forces transparent exceptions on total exhaustion for cleaner logging
        )
        self.http_session.mount("https://", HTTPAdapter(max_retries=retries))
        self.http_session.mount("http://", HTTPAdapter(max_retries=retries))

    async def fetch_daily_bars(self, ticker: str, lookback_days: int = 365) -> pd.DataFrame:
        """Unified routed method for daily historical candles with proactive safety pacing built-in."""
        if self.daily_source == "POLYGON":
            return await asyncio.to_thread(self._fetch_polygon_daily, ticker, lookback_days)
        else:
            # IBKR standard historical request pacing guard to prevent Pacing Violations (Error 162)
            await asyncio.sleep(1.0)
            return await self._fetch_ibkr_daily(ticker, lookback_days)

    async def fetch_hourly_bars(self, ticker: str, lookback_days: int = 30, ib_client: Optional[IB] = None) -> pd.DataFrame:
        """
        Unified hourly bar extraction matrix. Automatically routes traffic between 
        stateless REST queries (Polygon) or live stateful memory arrays (IBKR) 
        based on the system configuration rules.
        """
        # ROUTE 1: STATELESS REST INGESTION VIA POLYGON
        if self.hourly_source == "POLYGON":
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=lookback_days)
            start_str = start_dt.strftime("%Y-%m-%d")
            end_str = end_dt.strftime("%Y-%m-%d")
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{start_str}/{end_str}"
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 5000,
                "apiKey": self.poly_key
            }
            
            try:
                # Leverage our resilient connection pool with built-in retries
                res = self.http_session.get(url, params=params, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    results = data.get("results", [])
                    if results:
                        df = pd.DataFrame(results)
                        # Standardize column naming to align with downstream technical scanner matrices
                        df = df.rename(columns={
                            'c': 'close', 
                            'o': 'open', 
                            'h': 'high', 
                            'l': 'low', 
                            'v': 'volume', 
                            't': 'timestamp'
                        })
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        return df
                else:
                    logger.warning(f"Polygon Hourly API returned status code {res.status_code} for {ticker}")
            except Exception as e:
                logger.error(f"Polygon hourly aggregation sequence failed for asset {ticker}: {e}")
            return pd.DataFrame()

        # ROUTE 2: ASYNCHRONOUS TWS INGESTION VIA IBKR
        else:
            local_client = ib_client if ib_client else IB()
            disconnect_needed = False
            
            try:
                if not local_client.isConnected():
                    await local_client.connectAsync('127.0.0.1', self.ib_port, clientId=151)
                    disconnect_needed = True
                    
                contract = Contract(symbol=ticker, secType='STK', exchange='SMART', currency='USD')
                qualified = await local_client.qualifyContractsAsync(contract)
                if not qualified:
                    logger.warning(f"IBKR contract qualification rejected for ticker symbol: {ticker}")
                    return pd.DataFrame()
                    
                duration_str = f"{lookback_days} D"
                bars = await local_client.reqHistoricalDataAsync(
                    contract, endDateTime='', durationStr=duration_str,
                    barSizeSetting='1 hour', whatToShow='TRADES', useRTH=True
                )
                
                if bars:
                    df = util.df(bars)
                    # Harmonize schemas across platforms by mapping the datetime 'date' to 'timestamp'
                    df = df.rename(columns={'date': 'timestamp'})
                    return df  
            except Exception as e:
                logger.error(f"IBKR Hourly query breakdown routing asset {ticker}: {e}")
            finally:
                if disconnect_needed and local_client.isConnected():
                    local_client.disconnect()
            return pd.DataFrame()

    def _fetch_polygon_daily(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        """Ingests daily timeframes via Polygon REST architecture with automatic rate limit resolution."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": self.poly_key}
        
        try:
            res = self.http_session.get(url, params=params, timeout=15)
            if res.status_code == 200:
                results = res.json().get("results", [])
                if results:
                    df = pd.DataFrame(results)
                    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.strftime('%Y-%m-%d')
                    df['ticker'] = ticker
                    return df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.error(f"Polygon Daily ingestion failed for {ticker} (Rate limit triggered or network timeout): {e}")
        return pd.DataFrame()

    async def _fetch_ibkr_daily(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        """Ingests daily bars via a live Interactive Brokers socket connection."""
        ib = IB()
        disconnect_needed = False
        try:
            await ib.connectAsync('127.0.0.1', self.ib_port, clientId=150)
            disconnect_needed = True
            contract = Contract(symbol=ticker, secType='STK', exchange='SMART', currency='USD')
            qualified = await ib.qualifyContractsAsync(contract)
            if not qualified:
                logger.warning(f"IBKR Contract qualification failed for symbol: {ticker}")
                return pd.DataFrame()
            
            duration_str = f"{lookback_days} D"
            bars = await ib.reqHistoricalDataAsync(
                contract, endDateTime='', durationStr=duration_str,
                barSizeSetting='1 day', whatToShow='TRADES', useRTH=True
            )
            if bars:
                df = util.df(bars)
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                df['ticker'] = ticker
                return df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.error(f"IBKR Daily network socket error for {ticker}: {e}")
        finally:
            if disconnect_needed and ib.isConnected():
                ib.disconnect()
        return pd.DataFrame()

    def _fetch_polygon_hourly(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        """Ingests lower timeframe profile trends out of Polygon."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{start_date}/{end_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": self.poly_key}
        
        try:
            res = self.http_session.get(url, params=params, timeout=15)
            if res.status_code == 200:
                results = res.json().get("results", [])
                if results:
                    df = pd.DataFrame(results)
                    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.error(f"Polygon Hourly fetch exception error for {ticker}: {e}")
        return pd.DataFrame()

    async def _fetch_ibkr_hourly(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        """Extracts hourly bar sequences from live TWS socket connections."""
        ib = IB()
        disconnect_needed = False
        try:
            await ib.connectAsync('127.0.0.1', self.ib_port, clientId=151)
            disconnect_needed = True
            contract = Contract(symbol=ticker, secType='STK', exchange='SMART', currency='USD')
            qualified = await ib.qualifyContractsAsync(contract)
            if not qualified:
                return pd.DataFrame()
                
            duration_str = f"{lookback_days} D"
            bars = await ib.reqHistoricalDataAsync(
                contract, endDateTime='', durationStr=duration_str,
                barSizeSetting='1 hour', whatToShow='TRADES', useRTH=True
            )
            if bars:
                df = util.df(bars)
                df = df.rename(columns={'date': 'timestamp'})
                return df
        except Exception as e:
            logger.error(f"IBKR Hourly connection breakdown asset {ticker}: {e}")
        finally:
            if disconnect_needed and ib.isConnected():
                ib.disconnect()
        return pd.DataFrame()

    def fetch_ticker_fundamentals(self, ticker: str) -> pd.DataFrame:
        av_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        parsed_records = []
        consensus_lookup = {}

        # PHASE 1: EXTRACT TRUE WALL STREET CONSENSUS FROM ALPHA VANTAGE
        if av_key:
            av_url = "https://www.alphavantage.co/query"
            av_params = {"function": "EARNINGS", "symbol": ticker, "apikey": av_key}
            try:
                av_res = self.http_session.get(av_url, params=av_params, timeout=15)
                if av_res.status_code == 200:
                    av_data = av_res.json()

                    if isinstance(av_data, dict):
                        # Intercept Alpha Vantage's specific text-based rate-limit response structure
                        if "Note" in av_data or "Information" in av_data:
                            msg = av_data.get("Note") or av_data.get("Information")
                            logger.warning(f"Alpha Vantage limit note triggered for {ticker}: {msg}")
                        else:
                            quarterly_earnings = av_data.get("quarterlyEarnings", [])
                            if isinstance(quarterly_earnings, list):
                                for q in quarterly_earnings:
                                    fiscal_end = q.get("fiscalDateEnding")
                                    est_eps = q.get("estimatedEPS")
                                    if fiscal_end and est_eps not in [None, "None", ""]:
                                        consensus_lookup[fiscal_end] = float(est_eps)
                            else:
                                logger.warning(f"Alpha Vantage earnings payload for {ticker} is not a list.")
                    else:
                        logger.warning(f"Alpha Vantage payload for {ticker} is not a dictionary (type={type(av_data).__name__}).")
                else:
                    logger.warning(f"Alpha Vantage returned status {av_res.status_code} for {ticker}: {av_res.text}")
            except Exception as e:
                logger.error(f"Alpha Vantage consensus ingestion failed or throttled for {ticker}: {e}")
        else:
            logger.warning("ALPHA_VANTAGE_API_KEY not found in environment. Falling back to safe None values.")

        """
        Multi-Source Fundamental Ingestion Pipeline.
        Queries Alpha Vantage for true forward-looking analyst consensus expectations, 
        then overlays those expectations onto Polygon's raw SEC filing metrics.
        """
        


        # PHASE 1: EXTRACT TRUE WALL STREET CONSENSUS FROM ALPHA VANTAGE
        # if av_key:
        #     av_url = "https://www.alphavantage.co/query"
        #     av_params = {"function": "EARNINGS", "symbol": ticker, "apikey": av_key}
        #     try:
        #         av_res = self.http_session.get(av_url, params=av_params, timeout=15)
        #         if av_res.status_code == 200:
        #             av_data = av_res.json()
        #             quarterly_earnings = av_data.get("quarterlyEarnings", [])
                    
        #             # Index expectations by fiscal period end-date for O(1) alignment matrix mapping
        #             for q in quarterly_earnings:
        #                 fiscal_end = q.get("fiscalDateEnding")
        #                 est_eps = q.get("estimatedEPS")
        #                 if fiscal_end and est_eps not in [None, "None", ""]:
        #                     consensus_lookup[fiscal_end] = float(est_eps)
        #     except Exception as e:
        #         logger.error(f"Alpha Vantage consensus ingestion failed or throttled for {ticker}: {e}")
        # else:
        #     logger.warning("ALPHA_VANTAGE_API_KEY not found in environment. Falling back to safe None values.")

        # PHASE 2: EXTRACT FACTUAL REPORTED METRICS FROM POLYGON SEC FILINGS
        poly_url = "https://api.polygon.io/vX/reference/financials"
        poly_params = {"ticker": ticker, "limit": 5, "apiKey": self.poly_key}

        try:
            res = self.http_session.get(poly_url, params=poly_params, timeout=15)
            if res.status_code == 200:
                poly_data = res.json()
                if isinstance(poly_data, dict):
                    results = poly_data.get("results", [])
                else:
                    results = []
                    logger.warning(f"Polygon fundamentals response for {ticker} is not a dictionary (type={type(poly_data).__name__}).")

                if not isinstance(results, list):
                    logger.warning(f"Polygon fundamentals results for {ticker} is not a list (type={type(results).__name__}).")
                    results = []

                for item in results:
                    ann_date = item.get("start_date") or item.get("filing_date")
                    fiscal_end_date = item.get("end_date") # Used to cross-reference alignment matrices
                    if not ann_date:
                        continue

                    financials = item.get("financials", {}) or {}
                    income_stmt = financials.get("income_statement", {}) or {}

                    eps_val = income_stmt.get("basic_earnings_per_share", {}).get("value")
                    rev_val = (income_stmt.get("revenues", {}) or income_stmt.get("total_revenue", {})).get("value")

                    if eps_val is not None or rev_val is not None:
                        reported_eps = float(eps_val) if eps_val is not None else None
                        reported_rev = float(rev_val) if rev_val is not None else None

                        # ALIGNMENT MATRIX: Pull true consensus EPS if fiscal periods line up perfectly
                        expected_eps = consensus_lookup.get(fiscal_end_date)

                        # Note: Alpha Vantage focuses primarily on consensus EPS. If consensus revenue 
                        # is unavailable, we leave it as None to prevent introducing mathematical bias.
                        expected_rev = None 

                        parsed_records.append({
                            "ticker": ticker,
                            "announcement_date": ann_date,
                            "reported_eps": reported_eps,
                            "expected_eps": expected_eps, 
                            "reported_rev": reported_rev,
                            "expected_rev": expected_rev
                        })
            else:
                logger.warning(f"Polygon fundamentals returned status {res.status_code} for {ticker}: {res.text}")
        except Exception as e:
            logger.error(f"Polygon Fundamentals retrieval execution failure for asset {ticker}: {e}")

        return pd.DataFrame(parsed_records)

    def fetch_news_headlines(self, ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Isolates headline requests inside the factory retry architecture."""
        url = "https://api.polygon.io/v2/reference/news"
        params = {
            "ticker": ticker,
            "published_utc.gte": start_date,
            "published_utc.lte": end_date,
            "limit": 1000,
            "apiKey": self.poly_key
        }
        try:
            res = self.http_session.get(url, params=params, timeout=15)
            if res.status_code == 200:
                return res.json().get("results", [])
        except Exception as e:
            logger.error(f"News stream rate limit or extraction failure tracking asset {ticker}: {e}")
        return []

    def close(self):
        """Cleanly releases underlying HTTP pool handlers on application context termination."""
        self.http_session.close()