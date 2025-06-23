# analysis/data_loader.py
from pathlib import Path
import pandas as pd
from typing import Tuple, List, Dict
import SSSv095b2 as SSS

def load_data(ticker: str,
              start_date: str = "2000-01-01",
              end_date: str | None = None,
              smaa_source: str = "Self") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    直接調用 SSSv095b2.py 中的 load_data 函數，以確保數據載入邏輯完全一致。
    放棄本地的快取機制，因為它破壞了因子和價格數據的時間軸同步。
    """
    return SSS.load_data(ticker, start_date, end_date, smaa_source)

def filter_periods_by_data(df_price: pd.DataFrame, periods: List[Dict]) -> List[Dict]:
    """
    根據數據的實際日期範圍過濾分析期間。
    """
    if df_price.empty:
        return []
    first, last = df_price.index.min(), df_price.index.max()
    return [p for p in periods if pd.to_datetime(p["start"]) >= first and pd.to_datetime(p["end"]) <= last]