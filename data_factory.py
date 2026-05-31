# Prototype Data Factory Interface
class HistoricalDataFactory:
    def __init__(self, config: dict):
        self.config = config
        self.polygon_key = config["api_keys"]["polygon"]
        self.ib_port = config["system_settings"]["ibkr_port"]

    async def fetch_daily_bars(self, ticker: str, start_date: str) -> pd.DataFrame:
        source = self.config["system_settings"].get("daily_data_source", "IBKR").upper()
        if source == "POLYGON":
            return await self._fetch_polygon_daily(ticker, start_date)
        else:
            return await self._fetch_ibkr_daily(ticker, start_date)

    async def fetch_hourly_bars(self, ticker: str, lookback_days: int, ib_client=None) -> pd.DataFrame:
        source = self.config["system_settings"].get("hourly_data_source", "POLYGON").upper()
        if source == "POLYGON":
            return await self._fetch_polygon_hourly(ticker, lookback_days)
        else:
            # Reuses an existing open IB connection passed from main execution block
            return await self._fetch_ibkr_hourly(ticker, lookback_days, ib_client)