# analysis/data_loader.py
from pathlib import Path
import pandas as pd
import pyarrow.feather as feather
from typing import Tuple, List, Dict
from analysis import config as cfg
import SSSv095a1 as SSS

# ---------- 路徑 ----------
CACHE_DIR       = cfg.CACHE_DIR
PRICE_DIR       = CACHE_DIR / "price"
FACTOR_DIR      = CACHE_DIR / "factor"
PRICE_DIR.mkdir(parents=True, exist_ok=True)
FACTOR_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 基本 I/O ----------
def save_price_feather(ticker: str, df: pd.DataFrame) -> Path:
    f = PRICE_DIR / f"{ticker}.feather"
    feather.write_feather(df.reset_index(), f)
    return f

def load_price_feather(ticker: str) -> pd.DataFrame:
    f = PRICE_DIR / f"{ticker}.feather"
    df = feather.read_feather(f)
    # 如果原來 SSS.load_data 輸出時索引被 reset 成了 date 欄，優先用 date
    if 'date' in df.columns:
        df = df.set_index('date')
    # 否則就用 reset_index 時自動產生的 index 欄
    elif 'index' in df.columns:
        df = df.set_index('index')
    # 最後確保索引是 DatetimeIndex
    df.index = pd.to_datetime(df.index)
    return df


def save_factor_feather(name: str, df: pd.DataFrame) -> Path:
    f = FACTOR_DIR / f"{name}.feather"
    feather.write_feather(df.reset_index(), f)
    return f

def load_factor_feather(name: str) -> pd.DataFrame:
    f = FACTOR_DIR / f"{name}.feather"
    if f.exists():
        return feather.read_feather(f).set_index("index")
    return pd.DataFrame()

# ---------- 高階 ── 將 SSS 原始載入包一層 ----------
@cfg.MEMORY.cache
def load_data(ticker: str,
              start_date: str = cfg.START_DATE,
              end_date: str | None = None,
              smaa_source: str = "Self") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    根據指定的 smaa_source 載入數據，並快取到 feather 檔案。
    """
    # 為不同數據源生成唯一的快取檔案名稱
    cache_suffix = smaa_source.replace(" ", "_").replace("/", "_").replace("^", "")
    price_file = PRICE_DIR / f"{ticker}_{cache_suffix}.feather"
    factor_file = FACTOR_DIR / f"{ticker}_{cache_suffix}_factor.feather"

    # 檢查快取是否存在
    if price_file.exists() and factor_file.exists():
        df_price = load_price_feather(f"{ticker}_{cache_suffix}")
        df_factor = load_factor_feather(f"{ticker}_{cache_suffix}_factor")
        return df_price, df_factor

    # 若無快取，則從 SSS.load_data 載入
    df_price, df_factor = SSS.load_data(ticker, start_date, end_date, smaa_source)
    save_price_feather(f"{ticker}_{cache_suffix}", df_price)
    if not df_factor.empty:
        save_factor_feather(f"{ticker}_{cache_suffix}_factor", df_factor)
    return df_price, df_factor

def filter_periods_by_data(df_price: pd.DataFrame, periods: List[Dict]) -> List[Dict]:
    first, last = df_price.index.min(), df_price.index.max()
    return [p for p in periods if p["start"] >= str(first) and p["end"] <= str(last)]