__all__ = [
    "load_data", "compute_single", "compute_dual", "compute_RMA",
    "compute_ssma_turn_combined", "backtest_unified","build_smaa_path",
    "compute_backtest_for_periods", "calculate_metrics",
]
VERSION = "094a1"

# === 系統與標準函式庫 ===
import os
import json
import random
import hashlib
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from filelock import FileLock
# === 資料處理與運算 ===
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from joblib import Parallel, delayed
import textwrap

# === 可視化資料與前端介面 ===
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
try:
    import streamlit as st
except ModuleNotFoundError:
    class _Dummy:
        def __getattr__(self, k): return self
        def __call__(self, *a, **k): pass
    st = _Dummy()


# === 類型註解與工具 ===
from typing import Dict, List, Tuple, Optional
import itertools
from dataclasses import dataclass, field

# --- 專案結構與日誌設定 ---
from analysis import config as cfg
from analysis.logging_config import setup_logging
import logging
DATA_DIR = cfg.DATA_DIR
LOG_DIR = cfg.LOG_DIR
CACHE_DIR = cfg.CACHE_DIR
# 全局費率常數
BASE_FEE_RATE = 0.001425 # 基礎手續費 = 0.1425%
TAX_RATE = 0.003 # 賣出交易稅率 = 0.3%


# 預計算 SMAA
param_presets = {

"Single 2": {"linlen": 90, "factor": 40, "smaalen": 30, "devwin": 30, "buy_mult": 1.45, "sell_mult": 1.25,"stop_loss":0.2, "strategy_type": "single", "smaa_source": "Self"},
"Single 3": {"linlen": 80, "factor": 10, "smaalen": 60, "devwin": 20, "buy_mult": 0.4, "sell_mult": 1.5, "strategy_type": "single", "smaa_source": "Self"},
"SSMA_turn 1": {"linlen": 15,"smaalen": 40,"factor": 40.0,"prom_factor": 70,"min_dist": 10,"buy_shift": 6,"exit_shift": 4,"vol_window": 40,"quantile_win": 65,
 "signal_cooldown_days": 10,"buy_mult": 1.55,"sell_mult": 2.1,"stop_loss": 0.15,"strategy_type": "ssma_turn","smaa_source": "Self"},
"SSMA_turn 2": {"linlen": 10,"smaalen": 35,"factor": 50.0,"prom_factor": 70,"min_dist": 8,"buy_shift": 6,"exit_shift": 0,"vol_window": 40,"quantile_win": 85,
    "signal_cooldown_days": 10,"buy_mult": 1.6,"sell_mult": 2.2,"stop_loss": 0.2,"strategy_type": "ssma_turn","smaa_source": "Factor (^TWII / 2414.TW)"},

"SSMA_turn 4": {"linlen": 10,"smaalen": 35,"factor": 40.0,"prom_factor": 68,"min_dist": 8,"buy_shift": 6,"exit_shift": 0,"vol_window": 40,"quantile_win": 65,
    "signal_cooldown_days": 10,"buy_mult": 1.6,"sell_mult": 2.2,"stop_loss": 0.15,"strategy_type": "ssma_turn","smaa_source": "Factor (^TWII / 2414.TW)"},



"SSMA_turn 3": {"linlen": 20,"smaalen": 40,"factor": 40.0,"prom_factor": 69,"min_dist": 10,"buy_shift": 6,"exit_shift": 4,"vol_window": 45,"quantile_win": 55,
    "signal_cooldown_days": 10,"buy_mult": 1.65,"sell_mult": 2.1,"stop_loss": 0.2,"strategy_type": "ssma_turn","smaa_source": "Self"},
 
"SSMA_turn 0": {"linlen": 25, "smaalen": 85, "factor": 80.0, "prom_factor": 9, "min_dist": 8, "buy_shift": 0, "exit_shift": 6, "vol_window": 90, "quantile_win": 65, "signal_cooldown_days": 7, "buy_mult": 0.15, "sell_mult": 0.1, "stop_loss": 0.13, 
                "strategy_type": "ssma_turn", "smaa_source": "Factor (^TWII / 2414.TW)"},
#"ssma_turn_1308": {'linlen': 20, 'smaalen': 240, 'factor': 40.0, 'prom_factor': 47, 'min_dist': 16, 'buy_shift': 2, 'exit_shift': 1, 'vol_window': 90, 'quantile_win': 175, 'signal_cooldown_days': 4, 'buy_mult': 1.45, 'sell_mult': 2.1, 'stop_loss': 0.0,
#                "strategy_type": "ssma_turn", "smaa_source": "Self"},        
"ssma_turn_1939":{'linlen': 20, 'smaalen': 240, 'factor': 40.0, 'prom_factor': 48, 'min_dist': 14, 'buy_shift': 1, 'exit_shift': 1, 'vol_window': 80, 'quantile_win': 175, 'signal_cooldown_days': 4, 'buy_mult': 1.45, 'sell_mult': 2.6, 'stop_loss': 0.2,
                "strategy_type": "ssma_turn", "smaa_source": "Self"},                   
"RMA_1921": {'linlen': 167, 'smaalen': 107, 'rma_len': 35, 'dev_len': 80, 'factor': 40, 'buy_mult': 1.15, 'sell_mult': 0.95, 'stop_loss': 0.3, 'prom_factor': 0.5, 'min_dist': 5,  "strategy_type": "RMA", "smaa_source": "Factor (^TWII / 2412.TW)"},
#"RMA_1615":{'linlen': 75, 'smaalen': 232, 'rma_len': 60, 'dev_len': 45, 'factor': 40, 'buy_mult': 0.9, 'sell_mult': 2.15, 'stop_loss': 0.3, 'prom_factor': 0.5, 'min_dist': 5,  "strategy_type": "RMA", "smaa_source": "Factor (^TWII / 2414.TW)"},
        

"single_1939": {'linlen': 64, 'smaalen': 132, 'devwin': 20, 'factor': 40, 'buy_mult': 0.1, 'sell_mult': 1.75, 'stop_loss': 0.4, 'prom_factor': 0.5, 'min_dist': 5, "strategy_type": "single", "smaa_source": "Self"},
"ssma_turn_1994": {'linlen': 145, 'smaalen': 25, 'factor': 40.0, 'prom_factor': 55, 'min_dist': 6, 'buy_shift': 0, 'exit_shift': 1, 'vol_window': 70, 'quantile_win': 15, 'signal_cooldown_days': 5, 'buy_mult': 1.2, 'sell_mult': 1.1, 'stop_loss': 0.4,
                "strategy_type": "ssma_turn", "smaa_source": "Factor (^TWII / 2412.TW)"},

"ssma_turn_1679": {'linlen': 60, 'smaalen': 225, 'factor': 40.0, 'prom_factor': 27, 'min_dist': 5, 'buy_shift': 4, 'exit_shift': 7, 'vol_window': 65, 'quantile_win': 165, 'signal_cooldown_days': 7, 'buy_mult': 0.6, 'sell_mult': 1.7, 'stop_loss': 0.1,
                "strategy_type": "ssma_turn", "smaa_source": "Factor (^TWII / 2414.TW)"},


}
setup_logging()  # 初始化統一日誌設定
logger = logging.getLogger("SSSv095b2")  # 使用專屬 logger
# 新增：設置 log 檔案 handler（如尚未設置）
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    file_handler = logging.FileHandler("backtest_trade_records.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
from functools import wraps
import pickle
@dataclass
class TradeSignal:
    ts: pd.Timestamp
    side: str  # "BUY", "SELL", "FORCE_SELL", "STOP_LOSS"
    reason: str

# --- 快取路徑生成 ---
def build_smaa_path(ticker: str, source_key: str, linlen: int, factor: float, smaalen: int, data_hash: str, cache_dir: str = cfg.SMAA_CACHE_DIR) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"smaa_{source_key}_{ticker}_{linlen}_{factor}_{smaalen}_{data_hash}.npy"
# --- 參數驗證 ---
def validate_params(params: Dict, required_keys: set, positive_ints: set = None, positive_floats: set = None) -> bool:
    """
    驗證參數是否符合要求.

    Args:
        params (Dict): 待驗證的參數字典.
        required_keys (set): 必要的參數鍵.
        positive_ints (set, optional): 必須為正整數的參數鍵.
        positive_floats (set, optional): 必須為正浮點數的參數鍵.

    Returns:
        bool: 驗證通過返回 True,否則返回 False.
    """
    if not all(k in params for k in required_keys):
        logger.error(f"缺少必要參數: {required_keys - set(params.keys())}")
        st.error(f"缺少必要參數: {required_keys - set(params.keys())}")
        return False
    if positive_ints:
        for k in positive_ints:
            if k in params and (not isinstance(params[k], int) or params[k] <= 0):
                logger.error(f"參數 {k} 必須為正整數")
                st.error(f"參數 {k} 必須為正整數")
                return False
    if positive_floats:
        for k in positive_floats:
            if k in params and (not isinstance(params[k], (int, float)) or params[k] <= 0):
                logger.error(f"參數 {k} 必須為正數")
                st.error(f"參數 {k} 必須為正數")
                return False
    return True

def compute_cache_key(params, start_date, end_date, discount=0.30):
    safe_params = {}
    for k, v in params.items():
        if isinstance(v, tuple):
            safe_params[k] = list(v)
        else:
            safe_params[k] = v
    param_str = json.dumps(safe_params, sort_keys=True)
    return hashlib.md5(f"{param_str}_{start_date}_{end_date}_{discount}_{VERSION}".encode()).hexdigest()

def save_to_cache(cache_dir, key, data):
    """儲存計算結果到磁碟快取"""
    cache_dir.mkdir(exist_ok=True)
    with open(cache_dir / f"{key}.pkl", 'wb') as f:
        pickle.dump(data, f)

def load_from_cache(cache_dir, key):
    """從磁碟快取載入結果"""
    cache_file = cache_dir / f"{key}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

# --- 資料獲取與加載 ---
def fetch_yf_data(ticker: str, filename: Path, start_date: str = "2000-01-01", end_date: Optional[str] = None) -> None:
    """
    下載並儲存 Yahoo Finance 數據,檢查是否為當日最新數據.
    """
    now_taipei = pd.Timestamp.now(tz='Asia/Taipei')
    update_midnight_taipei = now_taipei.normalize() + pd.Timedelta(days=1)
    file_exists = filename.exists()
    proceed_with_fetch = True

    try:
        pd.to_datetime(start_date, format='%Y-%m-%d')
        if end_date:
            pd.to_datetime(end_date, format='%Y-%m-%d')
    except ValueError as e:
        logger.error(f"日期格式錯誤: {e}")
        st.sidebar.error(f"日期格式錯誤: {e}")
        return

    if file_exists:
        file_mod_time_taipei = pd.to_datetime(os.path.getmtime(filename), unit='s', utc=True).tz_convert('Asia/Taipei')
        logger.info(f"本地數據 '{filename}' 最後更新: {file_mod_time_taipei.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        if (file_mod_time_taipei.date() == now_taipei.date() and file_mod_time_taipei >= update_midnight_taipei) or \
           (now_taipei < update_midnight_taipei and file_mod_time_taipei >= (update_midnight_taipei - pd.Timedelta(days=1))):
            logger.info(f"數據 '{ticker}' 今日已是最新 (12:00 PM 後更新).")
            st.sidebar.success(f"數據 '{ticker}' 今日已是最新 (12:00 PM 後更新).")
            proceed_with_fetch = False
        else:
            logger.warning(f"數據 '{ticker}' 將嘗試更新 (上次更新早於今日12:00 PM).")
            st.sidebar.warning(f"數據 '{ticker}' 將嘗試更新 (上次更新早於今日12:00 PM).")
    else:
        logger.warning(f"本地數據 '{filename}' 未找到,將嘗試下載.")
        st.sidebar.warning(f"本地數據 '{filename}' 未找到,將嘗試下載.")

    if not proceed_with_fetch:
        return

    try:
        df = yf.download(ticker, period='max', auto_adjust=True)
        if df.empty:
            raise ValueError("下載的數據為空")
        df.to_csv(filename)
        logger.info(f"成功下載 '{ticker}' 數據到 '{filename}'.")
        st.sidebar.success(f"成功下載 '{ticker}' 數據到 '{filename}'.")
    except Exception as e:
        logger.error(f"警告: '{ticker}' 初次下載失敗: {e}")
        st.sidebar.error(f"警告: '{ticker}' 初次下載失敗: {e}")
        if file_exists:
            logger.info("使用現有舊數據檔案.")
            st.sidebar.info("使用現有舊數據檔案.")
            return
        try:
            logger.info(f"嘗試下載 '{ticker}' 的完整歷史數據...")
            df = yf.download(ticker, period='max', auto_adjust=True)
            if df.empty:
                raise ValueError("Fallback fetch empty")
            df.to_csv(filename)
            logger.info(f"成功下載 '{ticker}' 的完整歷史數據到 '{filename}'.")
            st.sidebar.success(f"成功下載 '{ticker}' 的完整歷史數據到 '{filename}'.")
        except Exception as e2:
            logger.error(f"無法下載 '{ticker}' 的數據: {e2}")
            st.sidebar.error(f"無法下載 '{ticker}' 的數據: {e2}")
            if not file_exists:
                raise RuntimeError(f"徹底無法獲取 {ticker} 的數據")

# --- 指標計算與回測函數 ---

def load_data(ticker: str, start_date: str = "2000-01-01", end_date: Optional[str] = None, smaa_source: str = "Self") -> Tuple[pd.DataFrame, pd.DataFrame]:
    filename = DATA_DIR / f"{ticker.replace(':','_')}_data_raw.csv"
    fetch_yf_data(ticker, filename, start_date, end_date)
    
    if not filename.exists():
        logger.error(f"數據文件 '{filename}' 不存在且無法下載.請檢查股票代號或網絡連接.")
        st.error(f"數據文件 '{filename}' 不存在且無法下載.請檢查股票代號或網絡連接.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        df = pd.read_csv(filename, parse_dates=[0], index_col=0, date_format='%Y-%m-%d')
        df.name = ticker.replace(':', '_')
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
        df = df[~df.index.isna()]

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                logger.warning(f"警告:數據中缺少 '{col}' 欄位,將以 NaN 填充.")
                st.warning(f"警告:數據中缺少 '{col}' 欄位,將以 NaN 填充.")
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['close'])
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        df_factor = pd.DataFrame()  # 預設空因子數據
        if smaa_source in ["Factor (^TWII / 2412.TW)", "Factor (^TWII / 2414.TW)"]:
            twii_file = DATA_DIR / "^TWII_data_raw.csv"
            factor_ticker = "2412.TW" if smaa_source == "Factor (^TWII / 2412.TW)" else "2414.TW"
            factor_file = DATA_DIR / f"{factor_ticker.replace(':','_')}_data_raw.csv"
            fetch_yf_data("^TWII", twii_file, start_date, end_date)
            fetch_yf_data(factor_ticker, factor_file, start_date, end_date)
            
            if not twii_file.exists() or not factor_file.exists():
                logger.warning(f"無法載入因子數據 (^TWII 或 {factor_ticker}),回退到 Self 模式.")
                st.warning(f"無法載入因子數據 (^TWII 或 {factor_ticker}),回退到 Self 模式.")
                return df, pd.DataFrame()
            
            try:
                df_twii = pd.read_csv(twii_file, parse_dates=[0], index_col=0, date_format='%Y-%m-%d')
                df_factor_ticker = pd.read_csv(factor_file, parse_dates=[0], index_col=0, date_format='%Y-%m-%d')
                df_twii.columns = [c.lower().replace(' ', '_') for c in df_twii.columns]
                df_factor_ticker.columns = [c.lower().replace(' ', '_') for c in df_factor_ticker.columns]
                df_twii.index = pd.to_datetime(df_twii.index, format='%Y-%m-%d', errors='coerce')
                df_factor_ticker.index = pd.to_datetime(df_factor_ticker.index, format='%Y-%m-%d', errors='coerce')
                df_twii = df_twii[~df_twii.index.isna()]
                df_factor_ticker = df_factor_ticker[~df_factor_ticker.index.isna()]
                for col in ['close', 'volume']:
                    df_twii[col] = pd.to_numeric(df_twii[col], errors='coerce')
                    df_factor_ticker[col] = pd.to_numeric(df_factor_ticker[col], errors='coerce')
                
                common_index = df_twii.index.intersection(df_factor_ticker.index).intersection(df.index)
                if len(common_index) < 100:
                    logger.warning(f"因子數據與價格數據的共同交易日不足 ({len(common_index)} 天),回退到 Self 模式.")
                    st.warning(f"因子數據與價格數據的共同交易日不足 ({len(common_index)} 天),回退到 Self 模式.")
                    return df, pd.DataFrame()
                factor_price = (df_twii['close'].loc[common_index] / df_factor_ticker['close'].loc[common_index]).rename('close')
                factor_volume = df_factor_ticker['volume'].loc[common_index].rename('volume')
                df_factor = pd.DataFrame({'close': factor_price, 'volume': factor_volume})
                df_factor = df_factor.reindex(df.index).dropna()
                if end_date:
                    df_factor = df_factor[df_factor.index <= pd.to_datetime(end_date)]
            except Exception as e:
                logger.warning(f"處理因子數據時出錯: {e},回退到 Self 模式.")
                st.warning(f"處理錯誤: {e},回退到 Self 模式.")
                return df, pd.DataFrame()
        
        df_factor.name = f"{ticker}_factor" if not df_factor.empty else None
        return df, df_factor
    except Exception as e:
        logger.error(f"讀取或處理數據文件 '{filename}' 時出錯: {e}")
        st.error(f"讀取或處理數據文件 '{filename}' 時出錯: {e}")
        return pd.DataFrame(), pd.DataFrame()


def load_data_wrapper(ticker: str, start_date: str = "2000-01-01",
                      end_date: str | None = None,
                      smaa_source: str = "Self"):
    """
    v096 相容接口:回傳 (df_price, df_factor)
    """
    df_price, df_factor = load_data(ticker, start_date, end_date, smaa_source)
    return df_price, df_factor



# --- 指標計算與回測函數 ---
def linreg_last_original(series: pd.Series, length: int) -> pd.Series:
    """原始的 linreg_last 實現,僅用於對比測試."""
    if len(series) < length or series.isnull().sum() > len(series) - length:
        return pd.Series(np.nan, index=series.index)
    return series.rolling(length, min_periods=length).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (len(x)-1) + np.polyfit(np.arange(len(x)), x, 1)[1]
        if len(x[~np.isnan(x)]) == length else np.nan, raw=True)

def linreg_last_vectorized(series: np.ndarray, length: int) -> np.ndarray:
    """
    向量化計算滾動線性回歸的最後一點擬合值.

    Args:
        series (np.ndarray): 輸入價格序列.
        length (int): 滾動窗口長度.

    Returns:
        np.ndarray: 每個窗口最後一點的擬合值,前 length-1 點為 NaN.
    """
    # 輸入驗證
    series = np.asarray(series, dtype=float)
    if len(series) < length:
        return np.full(len(series), np.nan, dtype=float)
    
    # 生成滑動窗口
    windows = np.lib.stride_tricks.sliding_window_view(series, length)
    valid = ~np.any(np.isnan(windows), axis=1)  # 檢查窗口內是否有 NaN
    
    # 構建 X 矩陣(x 座標:0, 1, ..., length-1)
    X = np.vstack([np.arange(length), np.ones(length)]).T  # 形狀 (length, 2)
    
    # 計算 (X^T X)^(-1) X^T(僅一次)
    XtX_inv_Xt = np.linalg.inv(X.T @ X) @ X.T  # 形狀 (2, length)
    
    # 向量化計算回歸係數:coeffs = (X^T X)^(-1) X^T y
    coeffs = np.einsum('ij,kj->ki', XtX_inv_Xt, windows[valid])  # 形狀 (valid_count, 2)
    
    # 計算最後一點的擬合值:y = slope * (length-1) + intercept
    result = np.full(len(windows), np.nan, dtype=float)
    result[valid] = coeffs[:, 0] * (length - 1) + coeffs[:, 1]
    
    # 填充前 length-1 個 NaN
    output = np.full(len(series), np.nan, dtype=float)
    output[length-1:] = result
    return output

def calc_smaa(series: pd.Series, linlen: int, factor: float, smaalen: int) -> pd.Series:
    """
    計算 SMAA(去趨勢化後的簡單移動平均),僅對足夠長的子序列計算。
    
    Args:
        series (pd.Series): 輸入價格序列.
        linlen (int): 線性回歸窗口長度.
        factor (float): 去趨勢放大因子.
        smaalen (int): 簡單移動平均窗口長度.
    
    Returns:
        pd.Series: SMAA 序列,早期數據點可能為 NaN,與輸入索引對齊.
    """
    # 轉為 NumPy 陣列並初始化結果
    series_values = series.values
    result = np.full(len(series), np.nan, dtype=float)
    
    # 檢查最小數據需求
    min_required = max(linlen, smaalen)
    if len(series) < min_required:
        logger.warning(f"序列長度不足: len={len(series)}, required={min_required},回傳全 NaN")
        return pd.Series(result, index=series.index)
    
    # 計算線性回歸
    lr = linreg_last_vectorized(series_values, linlen)
    
    # 去趨勢化並應用放大因子
    detr = (series_values - lr) * factor
    
    # 計算簡單移動平均,僅對有效窗口計算
    if len(detr) >= smaalen:
        sma = np.convolve(detr, np.ones(smaalen)/smaalen, mode='valid')
        result[smaalen-1:] = sma  # 從第 smaalen-1 個數據點開始填入有效值
    else:
        logger.warning(f"去趨勢化後數據不足以計算 SMA: len={len(detr)}, required={smaalen}")
    
    return pd.Series(result, index=series.index)

def precompute_smaa(ticker: str, param_combinations: list, start_date: str, smaa_source: str = "Self", cache_dir: str = str(cfg.SMAA_CACHE_DIR)):
    df_price, df_factor = load_data(ticker, start_date=start_date, smaa_source=smaa_source)
    source_df = df_factor if smaa_source != "Self" else df_price
    source_key = smaa_source.replace(" ", "_").replace("/", "_").replace("^", "")
    df_cleaned = source_df.dropna(subset=['close'])
    data_hash = str(pd.util.hash_pandas_object(df_cleaned['close']).sum())

    for linlen, factor, smaalen in param_combinations:
        smaa_path = build_smaa_path(ticker, source_key, linlen, factor, smaalen, data_hash, cache_dir)
        lock = FileLock(str(smaa_path) + ".lock")
        with lock:
            if smaa_path.exists():
                logger.debug(f"SMAA 快取已存在: {smaa_path}")
                continue
            logger.info(f"SMAA 已快取 : {ticker} ({linlen}, {factor}, {smaalen})")
            smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)
            np.save(smaa_path, smaa.to_numpy())
            logger.info(f"SMAA 快取完成: {smaa_path}")

            
def compute_single(df: pd.DataFrame, smaa_source_df: pd.DataFrame, linlen: int, factor: float, smaalen: int, devwin: int,smaa_source: str = "Self" ,cache_dir: str = str(cfg.SMAA_CACHE_DIR)) -> pd.DataFrame:
    source_df = smaa_source_df if not smaa_source_df.empty else df
    df_cleaned = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)  # 0001FIX:標準化精度
    data_hash = str(pd.util.hash_pandas_object(df_cleaned['close']).sum())

    source_key = smaa_source.replace(" ", "_").replace("/", "_").replace("^", "")
    ticker = df.name if hasattr(df, 'name') else "unknown"
    smaa_path = build_smaa_path(ticker, source_key, linlen, factor, smaalen, data_hash, cache_dir)
    
    from filelock import FileLock
    lock = FileLock(str(smaa_path) + ".lock")
    with lock:
        if smaa_path.exists():
            smaa_data = np.load(smaa_path, mmap_mode="r")
            if len(smaa_data) == len(df_cleaned):
                logger.debug(f"快取已存在: {smaa_path}")
                smaa = pd.Series(smaa_data, index=df_cleaned.index)
            else:
                logger.warning(f"SMAA快取 {smaa_path} 的長度({len(smaa_data)})與 df_cleaned({len(df_cleaned)})不一致，重新計算…")
                smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)
                np.save(smaa_path, smaa.to_numpy())
                logger.info(f"SMAA快取完成: {smaa_path}")
        else:
            logger.info(f"未找到 SMAA 快取 {smaa_path}，正在計算...")
            smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)
            np.save(smaa_path, smaa.to_numpy())
            logger.info(f"SMAA快取完成: {smaa_path}")
    
    # 計算 base 和 sd,並返回完整 DataFrame
    base = smaa.ewm(alpha=1/devwin, adjust=False, min_periods=devwin).mean()
    sd = (smaa - base).abs().ewm(alpha=1/devwin, adjust=False, min_periods=devwin).mean()
    
    results_df = pd.DataFrame({
        'smaa': smaa,
        'base': base,
        'sd': sd
    }, index=df_cleaned.index)
    final_df = pd.concat([df[['open', 'high', 'low', 'close']], results_df], axis=1, join='inner')
    final_df = final_df.dropna()  # 移除 NaN 行,但保留有效數據
    if final_df.empty:
        logger.warning(f"最終 DataFrame 為空,可能是 SMAA 數據不足,策略: single, linlen={linlen}, smaalen={smaalen}, data_len={len(df_cleaned)}, valid_smaa={len(smaa.dropna())}")
    return final_df

def compute_dual(df: pd.DataFrame, smaa_source_df: pd.DataFrame, linlen: int, factor: float, smaalen: int, short_win: int, long_win: int, smaa_source: str = "Self", cache_dir: str = str(cfg.SMAA_CACHE_DIR)) -> pd.DataFrame:
    source_df = smaa_source_df if not smaa_source_df.empty else df
    df_cleaned = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)  # 0001FIX:標準化精度
    data_hash = str(pd.util.hash_pandas_object(df_cleaned['close']).sum())
    source_key = smaa_source.replace(" ", "_").replace("/", "_").replace("^", "")
    ticker = df.name if hasattr(df, 'name') else "unknown"
    smaa_path = build_smaa_path(ticker, source_key, linlen, factor, smaalen, data_hash, cache_dir)
    
    from filelock import FileLock
    lock = FileLock(str(smaa_path) + ".lock")
    with lock:
        if smaa_path.exists():
            smaa_data = np.load(smaa_path, mmap_mode="r")
            if len(smaa_data) != len(df_cleaned):
                logger.warning(f"SMAA快取 {smaa_path} 的長度({len(smaa_data)})与 df_cleaned({len(df_cleaned)})不一致,正在重新計算…")
            else:
                logger.debug(f"快取已存在: {smaa_path}")
                smaa = pd.Series(smaa_data, index=df_cleaned.index)
        
        if 'smaa' not in locals():  # 表示需要重新計算
            logger.warning(f"未找到 SMAA 快取 {smaa_path},正在計算...")
            smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)
            np.save(smaa_path, smaa.to_numpy())
            logger.info(f"SMAA快取完成: {smaa_path}")
    
    # 計算 short 和 long 週期的 base 和 sd
    base_s = smaa.ewm(alpha=1/short_win, adjust=False, min_periods=short_win).mean()
    sd_s = (smaa - base_s).abs().ewm(alpha=1/short_win, adjust=False, min_periods=short_win).mean()
    base_l = smaa.ewm(alpha=1/long_win, adjust=False, min_periods=long_win).mean()
    sd_l = (smaa - base_l).abs().ewm(alpha=1/long_win, adjust=False, min_periods=long_win).mean()
    
    results_df = pd.DataFrame({
        'smaa': smaa,
        'base': base_s,
        'sd': sd_s,
        'base_long': base_l,
        'sd_long': sd_l
    }, index=df_cleaned.index)
    final_df = pd.concat([df[['open', 'high', 'low', 'close']], results_df], axis=1, join='inner')
    final_df = final_df.dropna()  # 移除 NaN 行,但保留有效數據
    if final_df.empty:
        logger.warning(f"最終 DataFrame 為空,可能是 SMAA 數據不足,策略: dual, linlen={linlen}, smaalen={smaalen}, data_len={len(df_cleaned)}, valid_smaa={len(smaa.dropna())}")
    return final_df

def compute_RMA(
    df: pd.DataFrame,
    smaa_source_df: pd.DataFrame,
    linlen: int,
    factor: float,
    smaalen: int,
    rma_len: int,
    dev_len: int,
    smaa_source: str = "Self",
    cache_dir: str = str(cfg.SMAA_CACHE_DIR)
) -> pd.DataFrame:
    """
    rma_len: 用於 EMA(base) 的視窗
    dev_len: 用於 rolling std(sd) 的視窗
    smaa_source: 快取鍵,從呼叫端傳入
    """
    # 1) 取快取鍵
    source_key = smaa_source.replace(" ", "_").replace("/", "_").replace("^", "")

    # 2) 準備資料
    source_df   = smaa_source_df if not smaa_source_df.empty else df
    df_cleaned  = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)  # 0001FIX:標準化精度
    data_hash   = str(pd.util.hash_pandas_object(df_cleaned['close']).sum())
    ticker      = getattr(df, 'name', 'unknown')
    smaa_path   = build_smaa_path(ticker, source_key, linlen, factor, smaalen, data_hash, cache_dir)

    # 3) 讀/寫快取
    from filelock import FileLock
    lock = FileLock(str(smaa_path) + ".lock")
    with lock:
        if smaa_path.exists():
            arr = np.load(smaa_path, mmap_mode="r")
            if len(arr) != len(df_cleaned):
                smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)
                np.save(smaa_path, smaa.to_numpy())
            else:
                smaa = pd.Series(arr, index=df_cleaned.index)
        else:
            smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)
            np.save(smaa_path, smaa.to_numpy())

    # 4) 計算 base / sd
    base = smaa.ewm(alpha=1/rma_len, adjust=False, min_periods=rma_len).mean()
    sd   = smaa.rolling(window=dev_len, min_periods=dev_len).std()

    # 5) 組成最終 DataFrame
    results = pd.DataFrame({
        'smaa': smaa,
        'base': base,
        'sd':   sd
    }, index=df_cleaned.index)
    final = pd.concat([df[['open','high','low','close']], results], axis=1, join='inner')
    final = final.dropna()  # 移除 NaN 行,但保留有效數據
    if final.empty:
        logger.warning(f"最終 DataFrame 為空,可能是 SMAA 數據不足,策略: single, linlen={linlen}, smaalen={smaalen}, data_len={len(df_cleaned)}, valid_smaa={len(smaa.dropna())}")
    return final




def compute_ssma_turn_combined(
    df: pd.DataFrame, smaa_source_df: pd.DataFrame, linlen: int, factor: float,
    smaalen: int, prom_factor: float, min_dist: int, buy_shift: int = 0, exit_shift: int = 0, vol_window: int = 20,
    signal_cooldown_days: int = 10, quantile_win: int = 100,
    smaa_source: str = "Self", cache_dir: str = str(cfg.SMAA_CACHE_DIR)
) -> Tuple[pd.DataFrame, List[pd.Timestamp], List[pd.Timestamp]]:
    logger.info("開始計算策略: linlen=%d, factor=%.2f, smaalen=%d, prom_factor=%.2f, min_dist=%d, buy_shift=%d, exit_shift=%d, vol_window=%d, quantile_win=%d, signal_cooldown_days=%d",
                linlen, factor, smaalen, prom_factor, min_dist, buy_shift, exit_shift, vol_window, quantile_win, signal_cooldown_days)

    # 參數驗證
    try:
        linlen = int(linlen)
        smaalen = int(smaalen)
        min_dist = int(min_dist)
        vol_window = int(vol_window)
        quantile_win = max(int(quantile_win), vol_window)
        signal_cooldown_days = int(signal_cooldown_days)
        buy_shift = int(buy_shift)
        exit_shift = int(exit_shift)
        if min_dist < 1 or vol_window < 1 or quantile_win < 1 or signal_cooldown_days < 0:
            raise ValueError("參數必須為非負整數")
    except (ValueError, TypeError) as e:
        logger.error(f"參數類型無效: {e}")
        st.error(f"參數類型無效: {e}")
        return pd.DataFrame(), [], []

    # 數據驗證
    source_df = smaa_source_df if not smaa_source_df.empty else df
    if 'close' not in source_df.columns or 'volume' not in source_df.columns:
        logger.error("'close' 或 'volume' 欄位在數據中缺失,無法計算 SMAA.")
        st.error("'close' 或 'volume' 欄位在數據中缺失,無法計算 SMAA.")
        return pd.DataFrame(), [], []

    df_cleaned = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)  # 0001FIX:標準化精度
    if df_cleaned.empty:
        logger.warning(f"清洗後數據為空,無法計算 SMAA, data_len={len(df_cleaned)}")
        st.warning(f"清洗後數據為空,無法計算 SMAA, data_len={len(df_cleaned)}")
        return pd.DataFrame(), [], []

    # 生成 source_key 和快取路徑
    source_key = smaa_source.replace(" ", "_").replace("/", "_").replace("^", "")
    ticker = getattr(df, 'name', 'unknown')
    data_hash = str(pd.util.hash_pandas_object(df_cleaned['close']).sum())
    smaa_path = build_smaa_path(ticker, source_key, linlen, factor, smaalen, data_hash, cache_dir)

    # 計算 SMAA
    from filelock import FileLock
    lock = FileLock(str(smaa_path) + ".lock")
    with lock:
        try:
            if smaa_path.exists():
                smaa_data = np.load(smaa_path, mmap_mode="r")
                if len(smaa_data) != len(df_cleaned):
                    logger.warning(f"SMAA快取 {smaa_path} 長度 ({len(smaa_data)}) 與清洗後資料長度 ({len(df_cleaned)}) 不符,重新計算中…")
                    smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)
                    np.save(smaa_path, smaa.to_numpy())
                    logger.info(f"SMAA快取更新至 {smaa_path}")
                else:
                    logger.debug(f"載入SMAA快取 {smaa_path}")
                    smaa = pd.Series(smaa_data, index=df_cleaned.index)
            else:
                logger.info(f"SMAA快取 {smaa_path} 不存在,重新計算中…")
                smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)
                np.save(smaa_path, smaa.to_numpy())
                logger.info(f"SMAA快取更新至 {smaa_path}")
        except Exception as e:
            logger.warning(f"載入SMAA快取 {smaa_path} 失敗:{e},重新計算中…")
            smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)
            np.save(smaa_path, smaa.to_numpy())
            logger.info(f"SMAA快取更新至 {smaa_path}")

    series_clean = smaa.dropna()
    if series_clean.empty:
        logger.warning(f"SMAA 資料為空,無法進行峰谷檢測, valid_smaa={len(series_clean)}, linlen={linlen}, smaalen={smaalen}")
        st.warning(f"SMAA 資料為空,無法進行峰谷檢測, valid_smaa={len(series_clean)}, linlen={linlen}, smaalen={smaalen}")
        return pd.DataFrame(), [], []

 

    # 閾值計算(使用舊版 ptp 邏輯)
    prom = series_clean.rolling(window=min_dist+1, min_periods=min_dist+1).apply(lambda x: x.ptp(), raw=True)
    initial_threshold = prom.quantile(prom_factor / 100) if len(prom.dropna()) > 0 else prom.median()
    threshold_series = prom.rolling(window=quantile_win, min_periods=quantile_win).quantile(prom_factor / 100).shift(1).ffill().fillna(initial_threshold)
    
    # 峰谷檢測(滾動窗口,放寬條件)
    peaks = []
    valleys = []
    last_signal_idx = -1          # 儲存「上一個訊號」在 series_clean 中的索引
    last_signal_dt  = None        # 儲存上一個訊號的日期

    for i in range(quantile_win, len(series_clean)):
        if (last_signal_dt is not None and
            (series_clean.index[i] - last_signal_dt).days <= signal_cooldown_days):
            continue
        window_data = series_clean.iloc[max(0, i - quantile_win):i + 1].to_numpy()
        if len(window_data) < min_dist + 1:
            continue
        current_threshold = threshold_series.iloc[i]
        window_peaks, _ = find_peaks(window_data, distance=min_dist, prominence=current_threshold)
        window_valleys, _ = find_peaks(-window_data, distance=min_dist, prominence=current_threshold)
        window_start_idx = max(0, i - quantile_win)
        if window_peaks.size > 0:
            for p_idx in window_peaks:
                peak_date = series_clean.index[window_start_idx + p_idx]
                if peak_date not in peaks:  # 去重
                    peaks.append(peak_date)
        if window_valleys.size > 0:
            for v_idx in window_valleys:
                valley_date = series_clean.index[window_start_idx + v_idx]
                if valley_date not in valleys:  # 去重
                    valleys.append(valley_date)
    
    # 成交量過濾
    vol_ma = df['volume'].rolling(vol_window, min_periods=vol_window).mean().shift(1)
    valid_peaks = []
    valid_valleys = []
    for p in peaks:
        if p in vol_ma.index and p in df.index:
            v = df.loc[p, 'volume']
            vol_avg = vol_ma.loc[p]
            if pd.notna(v) and pd.notna(vol_avg) and v > vol_avg:
                valid_peaks.append(p)
    for v in valleys:
        if v in vol_ma.index and v in df.index:
            v_vol = df.loc[v, 'volume']
            vol_avg = vol_ma.loc[v]
            if pd.notna(v_vol) and pd.notna(vol_avg) and v_vol > vol_avg:
                valid_valleys.append(v)
    
    # 冷卻期
    def apply_cooldown(dates, cooldown_days):
        filtered = []
        last_date = pd.Timestamp('1900-01-01')
        for d in sorted(dates):
            if (d - last_date).days >= cooldown_days:
                filtered.append(d)
                last_date = d
        return filtered
    valid_peaks = apply_cooldown(valid_peaks, signal_cooldown_days)
    valid_valleys = apply_cooldown(valid_valleys, signal_cooldown_days)
    
    # 買賣信號
    buy_dates = []
    sell_dates = []
    for dt in valid_valleys:
        try:
            tgt_idx = df.index.get_loc(dt) + 1 + buy_shift
            if 0 <= tgt_idx < len(df):
                buy_dates.append(df.index[tgt_idx])
        except KeyError:
            continue
    for dt in valid_peaks:
        try:
            tgt_idx = df.index.get_loc(dt) + 1 + exit_shift
            if 0 <= tgt_idx < len(df):
                sell_dates.append(df.index[tgt_idx])
        except KeyError:
            continue
    
    df_ind = df[['open', 'close']].copy()
    df_ind['smaa'] = smaa.reindex(df.index)
    if df_ind.dropna().empty:
        logger.warning(f"最終 df_ind 為空,可能是 SMAA 數據不足,策略: ssma_turn, linlen={linlen}, smaalen={smaalen}, valid_smaa={len(smaa.dropna())}")
    return df_ind.dropna(), buy_dates, sell_dates

def calculate_trade_mmds(trades: List[Tuple[pd.Timestamp, float, pd.Timestamp]], equity_curve: pd.Series) -> List[float]:
    """
    計算每筆持有期間的最大回撤（MMD）。
    Args:
        trades: 交易記錄,包含 entry_date, return, exit_date。
        equity_curve: 全部回測期間的每日資產淨值。
    Returns:
        List[float]: 每筆持有期間的最大回撤。
    """
    mmds = []
    for entry_date, _, exit_date in trades:
        # 取出持有期間的 equity
        period_equity = equity_curve.loc[entry_date:exit_date]
        if len(period_equity) < 2:
            mmds.append(0.0)
            continue
        roll_max = period_equity.cummax()
        drawdown = period_equity / roll_max - 1
        mmds.append(drawdown.min())
    return mmds

def calculate_metrics(trades: List[Tuple[pd.Timestamp, float, pd.Timestamp]], df_ind: pd.DataFrame, equity_curve: pd.Series = None) -> Dict:
    """
    計算回測績效指標.
    Args:
        trades: 交易記錄,包含日期和報酬率.
        df_ind: 指標數據 DataFrame,包含交易日索引.
        equity_curve: 全部回測期間的每日資產淨值（可選, 若要計算持有期間MMD需提供）
    Returns:
        Dict: 包含總回報率、年化回報率、最大回撤等指標.
    """
    metrics = {
        'total_return': 0.0,
        'annual_return': 0.0,
        'max_drawdown': 0.0,  # 這裡之後會直接用 max_mmd 覆蓋
        'max_drawdown_duration': 0,
        'calmar_ratio': np.nan,
        'num_trades': 0,
        'win_rate': 0.0,
        'avg_win': np.nan,
        'avg_loss': np.nan,
        'payoff_ratio': np.nan,
        'sharpe_ratio': np.nan,
        'sortino_ratio': np.nan,
        'max_consecutive_wins': 0,# 新增
        'max_consecutive_losses': 0,# 新增
        'avg_holding_period': np.nan,# 新增
        'annualized_volatility': np.nan,# 新增
        'profit_factor': np.nan,# 新增 
        # 'avg_mmd': np.nan, # 不再需要
        # 'max_mmd': np.nan, # 不再需要單獨欄位
    }
    if not trades:
        return metrics
    
    trade_metrics = pd.DataFrame(trades, columns=['entry_date', 'return', 'exit_date']).set_index('exit_date')
    trade_metrics['equity'] = (1 + trade_metrics['return']).cumprod()
    roll_max = trade_metrics['equity'].cummax()
    daily_drawdown = trade_metrics['equity'] / roll_max - 1
    
    # 基本指標
    metrics['total_return'] = trade_metrics['equity'].iloc[-1] - 1
    years = max((trade_metrics.index[-1] - trade_metrics.index[0]).days / 365.25, 1)
    metrics['annual_return'] = (1 + metrics['total_return']) ** (1 / years) - 1
    # metrics['max_drawdown'] = daily_drawdown.min() # 不再用這個
    dd_series = daily_drawdown < 0
    dd_dur = max((dd_series.groupby((~dd_series).cumsum()).cumcount() + 1).max(), 0) if dd_series.any() else 0
    metrics['max_drawdown_duration'] = dd_dur
    # metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] < 0 else np.nan
    metrics['num_trades'] = len(trade_metrics)
    metrics['win_rate'] = (trade_metrics['return'] > 0).sum() / metrics['num_trades'] if metrics['num_trades'] > 0 else 0
    metrics['avg_win'] = trade_metrics[trade_metrics['return'] > 0]['return'].mean() if metrics['win_rate'] > 0 else np.nan
    metrics['avg_loss'] = trade_metrics[trade_metrics['return'] < 0]['return'].mean() if metrics['win_rate'] < 1 else np.nan
    metrics['payoff_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 and not np.isnan(metrics['avg_win']) else np.nan
    
    # 新增:計算每日報酬率與波動率
    daily_dates = df_ind.index.intersection(pd.date_range(start=trade_metrics.index.min(), end=trade_metrics.index.max(), freq='B'))
    daily_equity = pd.Series(index=daily_dates, dtype=float)
    for date, row in trade_metrics.iterrows():
        daily_equity.loc[date] = row['equity']
    daily_equity = daily_equity.ffill()
    daily_returns = daily_equity.pct_change().dropna()
    
    metrics['sharpe_ratio'] = (daily_returns.mean() * np.sqrt(252)) / daily_returns.std() if daily_returns.std() != 0 else np.nan
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else np.nan
    metrics['sortino_ratio'] = (daily_returns.mean() * np.sqrt(252)) / downside_std if downside_std != 0 else np.nan
    
    # 新增:計算最大連續盈利和最大連續虧損
    trade_metrics['win_flag'] = trade_metrics['return'] > 0
    trade_metrics['grp'] = (trade_metrics['win_flag'] != trade_metrics['win_flag'].shift(1)).cumsum()
    consec = trade_metrics.groupby(['grp', 'win_flag']).size()
    metrics['max_consecutive_wins'] = consec[consec.index.get_level_values('win_flag') == True].max() if True in consec.index.get_level_values('win_flag') else 0
    metrics['max_consecutive_losses'] = consec[consec.index.get_level_values('win_flag') == False].max() if False in consec.index.get_level_values('win_flag') else 0

    # 新增:計算年化波動率
    metrics['annualized_volatility'] = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else np.nan

    # 新增:計算盈虧因子 (Profit Factor)
    total_profits = trade_metrics[trade_metrics['return'] > 0]['return'].sum()
    total_losses = abs(trade_metrics[trade_metrics['return'] < 0]['return'].sum())
    metrics['profit_factor'] = total_profits / total_losses if total_losses != 0 else np.nan

    # 新增:計算每筆持有期間MMD，並直接用 max_mmd 覆蓋 max_drawdown
    if equity_curve is not None:
        mmds = calculate_trade_mmds(trades, equity_curve)
        if mmds:
            metrics['max_drawdown'] = float(np.min(mmds)) # drawdown為負值,最小值為最大跌幅
            metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] < 0 else np.nan

    return metrics

def calculate_holding_periods(trade_df: pd.DataFrame) -> float:
    """
    計算平均持倉天數.
    
    Args:
        trade_df: 交易記錄 DataFrame,包含 signal_date, trade_date, type.
    
    Returns:
        float: 平均持倉天數.
    """
    if trade_df.empty or 'trade_date' not in trade_df.columns or 'type' not in trade_df.columns:
        return np.nan

    holding_periods = []
    entry_date = None

    for _, row in trade_df.iterrows():
        if row['type'] == 'buy':
            entry_date = row['trade_date']
        elif row['type'] == 'sell' and entry_date is not None:
            exit_date = row['trade_date']
            holding_days = (exit_date - entry_date).days
            holding_periods.append(holding_days)
            entry_date = None  # 重置

    return np.mean(holding_periods) if holding_periods else np.nan


def backtest_unified(
    df_ind: pd.DataFrame,
    strategy_type: str,
    params: Dict,
    buy_dates: Optional[List[pd.Timestamp]] = None,
    sell_dates: Optional[List[pd.Timestamp]] = None,
    discount: float = 0.30,
    trade_cooldown_bars: int = 3,
    bad_holding: bool = False,
    use_leverage: bool = False,
    lev_params: Optional[Dict] = None
) -> Dict:
    if not isinstance(df_ind, pd.DataFrame):
        logger.error(f"df_ind 必須是一個 pandas.DataFrame，卻傳入 {type(df_ind)}")
        return {'trades': [], 'trade_df': pd.DataFrame(), 'trades_df': pd.DataFrame(), 'signals_df': pd.DataFrame(), 'metrics': [], 'equity_curve': pd.Series()}

    BUY_FEE_RATE = BASE_FEE_RATE * discount
    SELL_FEE_RATE = BASE_FEE_RATE * discount + TAX_RATE
    ROUND_TRIP_FEE = BUY_FEE_RATE + SELL_FEE_RATE

    if use_leverage:
        from leverage import LeverageEngine
        lev = LeverageEngine(**(lev_params or {}))
    else:
        lev = None

    required_cols = ['open', 'close'] if strategy_type == 'ssma_turn' else ['open', 'close', 'smaa', 'base', 'sd']
    if df_ind.empty or not all(col in df_ind.columns for col in required_cols):
        logger.warning(f"指標數據不完整，無法執行回測(缺少欄位: {set(required_cols) - set(df_ind.columns)})")
        return {'trades': [], 'trade_df': pd.DataFrame(), 'trades_df': pd.DataFrame(), 'signals_df': pd.DataFrame(), 'metrics': [], 'equity_curve': pd.Series()}

    try:
        trade_cooldown_bars = int(trade_cooldown_bars)
        if trade_cooldown_bars < 0:
            raise ValueError("trade_cooldown_bars 必須為非負整數")
        params['stop_loss'] = float(params.get('stop_loss', 0.0))
        if bad_holding and params['stop_loss'] <= 0:
            raise ValueError("當啟用 bad_holding 時，stop_loss 必須為正數")
        if strategy_type == 'ssma_turn':
            params['exit_shift'] = int(params.get('exit_shift', 0))
            if params['exit_shift'] < 0:
                raise ValueError("exit_shift 必須為非負整數")
        else:
            params['buy_mult'] = float(params.get('buy_mult', 0.5))
            params['sell_mult'] = float(params.get('sell_mult', 0.5))
            params['prom_factor'] = float(params.get('prom_factor', 0.5))
            params['min_dist'] = int(params.get('min_dist', 5))
            if params['buy_mult'] < 0 or params['sell_mult'] < 0:
                raise ValueError("buy_mult 和 sell_mult 必須為非負數")
            if params['min_dist'] < 1:
                raise ValueError("min_dist 必須為正整數")
    except (ValueError, TypeError) as e:
        logger.error(f"參數驗證失敗: {e}")
        return {'trades': [], 'trade_df': pd.DataFrame(), 'trades_df': pd.DataFrame(), 'signals_df': pd.DataFrame(), 'metrics': [], 'equity_curve': pd.Series()}

    initial_cash = 100000
    cash = initial_cash
    total_shares = 0
    trades = []
    trade_records = []
    signals = []
    in_pos = False
    entry_price = 0.0
    entry_date = None
    accum_interest = 0.0
    last_trade_idx = -trade_cooldown_bars - 1
    buy_idx = 0
    sell_idx = 0

    # 初始化 Equity Curve
    equity_curve = pd.Series(initial_cash, index=df_ind.index, dtype=float)
    cash_series = pd.Series(initial_cash, index=df_ind.index, dtype=float)
    shares_series = pd.Series(0, index=df_ind.index, dtype=float)

    signals_list = []
    if strategy_type == 'ssma_turn':
        buy_dates = sorted(buy_dates or [])
        sell_dates = sorted(sell_dates or [])
        for dt in buy_dates:
            signals_list.append(TradeSignal(ts=dt, side="BUY", reason="ssma_turn_valley"))
        for dt in sell_dates:
            signals_list.append(TradeSignal(ts=dt, side="SELL", reason="ssma_turn_peak"))
    else:
        for i in range(len(df_ind)):
            date = df_ind.index[i]
            if df_ind['smaa'].iloc[i] < df_ind['base'].iloc[i] + df_ind['sd'].iloc[i] * params['buy_mult']:
                signals_list.append(TradeSignal(ts=date, side="BUY", reason=f"{strategy_type}_buy"))
            elif df_ind['smaa'].iloc[i] > df_ind['base'].iloc[i] + df_ind['sd'].iloc[i] * params['sell_mult']:
                signals_list.append(TradeSignal(ts=date, side="SELL", reason=f"{strategy_type}_sell"))

    signals_list.sort(key=lambda x: x.ts)

    n = len(df_ind)
    scheduled_buy = np.zeros(n, dtype=bool)
    scheduled_sell = np.zeros(n, dtype=bool)
    scheduled_forced = np.zeros(n, dtype=bool)
    idx_by_date = {date: i for i, date in enumerate(df_ind.index)}

    for sig in signals_list:
        ts = pd.Timestamp(sig.ts).tz_localize(None) if sig.ts.tzinfo else sig.ts
        if ts in idx_by_date:
            i = idx_by_date[ts]
            if i + 1 < n:
                if sig.side == "BUY":
                    scheduled_buy[i + 1] = True
                elif sig.side in ["SELL", "STOP_LOSS", "FORCE_SELL"]:
                    scheduled_sell[i + 1] = True if sig.side == "SELL" else False
                    scheduled_forced[i + 1] = True if sig.side in ["STOP_LOSS", "FORCE_SELL"] else scheduled_forced[i + 1]

    for i in range(n):
        today = df_ind.index[i]
        today_open = df_ind['open'].iloc[i]
        today_close = df_ind['close'].iloc[i]
        mkt_val = total_shares * today_close

        if use_leverage and in_pos:
            interest = lev.accrue()
            cash -= interest
            accum_interest += interest
            forced = lev.margin_call(mkt_val=mkt_val)
            if forced > 0 and i + 1 < n:
                scheduled_forced[i + 1] = True

        # 更新現金與持股
        cash_series.iloc[i] = cash
        shares_series.iloc[i] = total_shares
        equity_curve.iloc[i] = cash + total_shares * today_close

        if (scheduled_sell[i] or scheduled_forced[i]) and in_pos and total_shares > 0:
            exit_price = today_open
            exit_date = today
            trade_ret = (exit_price / entry_price) - 1 - ROUND_TRIP_FEE - (accum_interest / (entry_price * total_shares)) if entry_price != 0 and total_shares > 0 else 0
            if bad_holding and trade_ret < -0.20 and not scheduled_forced[i]:
                continue
            cash += total_shares * exit_price
            sell_shares = total_shares
            total_shares = 0
            if use_leverage and lev.loan > 0:
                repay_amt = min(cash, lev.loan)
                lev.repay(repay_amt)
                cash -= repay_amt
                trade_records.append({
                    'signal_date': today,
                    'trade_date': exit_date,
                    'type': 'repay',
                    'price': 0.0,
                    'loan_amount': repay_amt
                })
                signals.append({'signal_date': today, 'type': 'repay', 'price': 0.0})
            trades.append((entry_date, trade_ret, exit_date))
            trade_records.append({
                'signal_date': today,
                'trade_date': exit_date,
                'type': 'sell' if scheduled_sell[i] else 'sell_forced',
                'price': exit_price,
                'shares': sell_shares,
                'return': trade_ret
            })
            #logger.info(f"TRADE | type={'sell' if scheduled_sell[i] else 'sell_forced'} | signal_date={today} | trade_date={exit_date} | price={exit_price} | shares={sell_shares} | return={trade_ret}")
            signals.append({'signal_date': today, 'type': 'sell' if scheduled_sell[i] else 'sell_forced', 'price': today_close})
            in_pos = False
            last_trade_idx = i
            accum_interest = 0.0
            if strategy_type == 'ssma_turn' and scheduled_sell[i]:
                sell_idx += 1
            continue

        if scheduled_buy[i] and not in_pos and i - last_trade_idx > trade_cooldown_bars:
            shares = int(cash // today_open)
            if shares > 0:
                need_cash = shares * today_open
                if use_leverage:
                    gap = need_cash - cash
                    if gap > 0:
                        borrowable = lev.avail(mkt_val=mkt_val)
                        draw = min(gap, borrowable)
                        if draw > 0:
                            lev.borrow(draw)
                            cash += draw
                cash -= need_cash
                total_shares = shares
                entry_price = today_open
                entry_date = today
                in_pos = True
                last_trade_idx = i
                accum_interest = 0.0
                trade_records.append({
                    'signal_date': today,
                    'trade_date': entry_date,
                    'type': 'buy',
                    'price': entry_price,
                    'shares': shares
                })
                #logger.info(f"TRADE | type=buy | signal_date={today} | trade_date={entry_date} | price={entry_price} | shares={shares}")
                signals.append({'signal_date': today, 'type': 'buy', 'price': today_close})
                if strategy_type == 'ssma_turn':
                    buy_idx += 1
            continue

        if bad_holding and in_pos and entry_price > 0 and today_close / entry_price - 1 <= -params['stop_loss'] and i + 1 < n:
            scheduled_forced[i + 1] = True

    if in_pos and total_shares > 0:
        exit_idx = n - 1
        exit_price = df_ind['open'].iloc[exit_idx]
        exit_date = df_ind.index[exit_idx]
        trade_ret = (exit_price / entry_price) - 1 - ROUND_TRIP_FEE - (accum_interest / (entry_price * total_shares)) if entry_price != 0 and total_shares > 0 else 0
        cash += total_shares * exit_price
        sell_shares = total_shares
        total_shares = 0
        if use_leverage and lev.loan > 0:
            repay_amt = min(cash, lev.loan)
            lev.repay(repay_amt)
            cash -= repay_amt
            trade_records.append({
                'signal_date': exit_date,
                'trade_date': exit_date,
                'type': 'repay',
                'price': 0.0,
                'loan_amount': repay_amt
            })
            signals.append({'signal_date': exit_date, 'type': 'repay', 'price': 0.0})
        trades.append((entry_date, trade_ret, exit_date))
        trade_records.append({
            'signal_date': exit_date,
            'trade_date': exit_date,
            'type': 'sell_forced',
            'price': exit_price,
            'shares': sell_shares,
            'return': trade_ret
        })
        #logger.info(f"TRADE | type=sell_forced | signal_date={exit_date} | trade_date={exit_date} | price={exit_price} | shares={sell_shares} | return={trade_ret}")
        signals.append({'signal_date': exit_date, 'type': 'sell_forced', 'price': exit_price})
        accum_interest = 0.0

    # 更新最後一天的 Equity Curve
    cash_series.iloc[-1] = cash
    shares_series.iloc[-1] = total_shares
    equity_curve.iloc[-1] = cash + total_shares * df_ind['close'].iloc[-1]

    trade_df = pd.DataFrame(trade_records)
    trades_df = pd.DataFrame(trades, columns=['entry_date', 'ret', 'exit_date'])
    signals_df = pd.DataFrame(signals)
    metrics = calculate_metrics(trades, df_ind, equity_curve)

    logger.info(f"{strategy_type} 回測結果: 總報酬率 = {metrics.get('total_return', 0):.2%}, 交易次數={metrics.get('num_trades', 0)}")
    return {'trades': trades, 'trade_df': trade_df, 'trades_df': trades_df, 'signals_df': signals_df, 'metrics': metrics, 'equity_curve': equity_curve}
def compute_backtest_for_periods(ticker: str,periods: List[Tuple[str, str]],strategy_type: str,params: Dict,
    smaa_source: str = "Self",trade_cooldown_bars: int = 3,discount: float = 0.30,
    bad_holding: bool = False,df_price: Optional[pd.DataFrame] = None,df_factor: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
    """
    對多個時段進行回測，確保返回標準化結果。
    """
    results = []
    for start_date, end_date in periods:
        logger.info(f"回測時段: {start_date} 至 {end_date}")
        # 若未提供預載數據，則內部載入
        if df_price is None or df_factor is None:
            df_raw, df_factor = load_data(ticker, start_date, end_date, smaa_source)
        else:
            df_raw = df_price.loc[start_date:end_date].copy() if start_date in df_price.index and end_date in df_price.index else pd.DataFrame()
            df_factor = df_factor.loc[start_date:end_date].copy() if not df_factor.empty and start_date in df_factor.index and end_date in df_factor.index else pd.DataFrame()
        
        if df_raw.empty:
            logger.warning(f"時段 {start_date} 至 {end_date} 數據為空，跳過。")
            results.append({
                'trades': [],
                'trade_df': pd.DataFrame(),
                'signals_df': pd.DataFrame(),
                'metrics': {'total_return': -np.inf, 'num_trades': 0},  # 修改預設值
                'period': {'start_date': start_date, 'end_date': end_date}
            })
            continue
        
        if strategy_type == 'ssma_turn':
            calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
            ssma_params = {k: v for k, v in params.items() if k in calc_keys}
            df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(df_raw, df_factor, **ssma_params)
            if df_ind.empty:
                logger.warning(f"{strategy_type} 策略計算失敗，可能是數據不足。")
                results.append({
                    'trades': [],
                    'trade_df': pd.DataFrame(),
                    'signals_df': pd.DataFrame(),
                    'metrics': {'total_return': -np.inf, 'num_trades': 0},  # 修改預設值
                    'period': {'start_date': start_date, 'end_date': end_date}
                })
                continue
            result = backtest_unified(df_ind, strategy_type, params, buy_dates, sell_dates, 
                                     discount=discount, trade_cooldown_bars=trade_cooldown_bars, bad_holding=bad_holding)
        else:
            if strategy_type == 'single':
                df_ind = compute_single(df_raw, df_factor, params['linlen'], params['factor'], params['smaalen'], params['devwin'])
            elif strategy_type == 'dual':
                df_ind = compute_dual(df_raw, df_factor, params['linlen'], params['factor'], params['smaalen'], params['short_win'], params['long_win'])
            elif strategy_type == 'RMA':
                df_ind = compute_RMA(df_raw, df_factor, params['linlen'], params['factor'], params['smaalen'], params['rma_len'], params['dev_len'])
            if df_ind.empty:
                logger.warning(f"{strategy_type} 策略計算失敗，可能是數據不足。")
                results.append({
                    'trades': [],
                    'trade_df': pd.DataFrame(),
                    'signals_df': pd.DataFrame(),
                    'metrics': {'total_return': -np.inf, 'num_trades': 0},  # 修改預設值
                    'period': {'start_date': start_date, 'end_date': end_date}
                })
                continue
            result = backtest_unified(df_ind, strategy_type, params, discount=discount, trade_cooldown_bars=trade_cooldown_bars, bad_holding=bad_holding)
        
        result['period'] = {'start_date': start_date, 'end_date': end_date}
        results.append(result)
    
    return results


# --- 可視化函數 ---
def plot_stock_price(df: pd.DataFrame, trades_df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    繪製股票價格與買賣信號.
    
    Args:
        df (pd.DataFrame): 價格數據,包含 close 欄位.
        trades_df (pd.DataFrame): 交易記錄,包含 trade_date, type, price.
        ticker (str): 股票代號.
    
    Returns:
        go.Figure: Plotly 圖表物件.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close Price', line=dict(color='dodgerblue')))
    if not trades_df.empty:
        buys = trades_df[trades_df['type'] == 'buy']
        adds = trades_df[trades_df['type'] == 'add'] if 'add' in trades_df['type'].values else pd.DataFrame()
        sells = trades_df[trades_df['type'] == 'sell']  # 修改:僅顯示 sell,忽略 sell_forced
        fig.add_trace(go.Scatter(x=buys['trade_date'], y=buys['price'], mode='markers', name='Buy',
                                 marker=dict(symbol='cross', size=10, color='green')))
        if not adds.empty:
            fig.add_trace(go.Scatter(x=adds['trade_date'], y=adds['price'], mode='markers', name='Add-On Buy',
                                     marker=dict(symbol='cross', size=10, color='limegreen')))
        fig.add_trace(go.Scatter(x=sells['trade_date'], y=sells['price'], mode='markers', name='Sell',
                                 marker=dict(symbol='x', size=10, color='red')))
    fig.update_layout(title=f'{ticker} 股價與交易信號',
                      xaxis_title='日期', yaxis_title='價格', template='plotly_white')
    return fig

def plot_indicators(df_ind: pd.DataFrame, strategy_type: str, trades_df: pd.DataFrame, params: Dict) -> go.Figure:
    fig = go.Figure()
    if strategy_type in ['single', 'dual', 'RMA']:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['smaa'], name='SMAA', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['base'], name='Base', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['base'] + df_ind['sd'] * params['buy_mult'], name='Buy Level',
                                 line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['base'] + df_ind['sd'] * params['sell_mult'], name='Sell Level',
                                 line=dict(color='red', dash='dash')))
        if 'base_long' in df_ind.columns:
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['base_long'], name='Base Long', line=dict(color='purple', dash='dot')))
    else:  # ssma_turn
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['smaa'], name='SMAA', line=dict(color='blue')))

    buys = trades_df[trades_df['type'] == 'buy']
    sells = trades_df[trades_df['type'] == 'sell']
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys['trade_date'], y=df_ind['smaa'].reindex(buys['trade_date']).values, mode='markers', name='Buy Signal',
                                 marker=dict(symbol='cross', size=10, color='green')))
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells['trade_date'], y=df_ind['smaa'].reindex(sells['trade_date']).values, mode='markers', name='Sell Signal',
                                 marker=dict(symbol='x', size=10, color='red')))

    fig.update_layout(title='SMAA指標',
                      xaxis_title='Date', yaxis_title='Value', template='plotly_white')
    return fig

def plot_equity_cash(trades_df: pd.DataFrame, price_df: pd.DataFrame, initial_cash: float = 100000) -> go.Figure:
    """
    繪製每日浮動權益和現金曲線,確保連續且反映收盤價變化.
    """
    if trades_df.empty or price_df.empty:
        logger.warning("交易或價格數據為空,無法繪製權益曲線.")
        return go.Figure()

    # 1. 建立每日時間軸
    dates = price_df.index
    cash_series = pd.Series(initial_cash, index=dates, dtype=float)
    shares_series = pd.Series(0, index=dates, dtype=float)

    # 型別自動對齊：將 trade_date 轉為 Timestamp
    if 'trade_date' in trades_df.columns:
        trades_df = trades_df.copy()
        trades_df['trade_date'] = pd.to_datetime(trades_df['trade_date'])

    # 2. 逐筆交易更新持股和現金
    for _, row in trades_df.iterrows():
        dt = row['trade_date']
        px = row['price']
        if dt not in dates:
            logger.debug(f"交易日期 {dt} 不在價格數據範圍內,跳過.")
            continue
        shares = row.get('shares', 0)
        if row['type'] in ('buy', 'add'):
            shares_series.loc[dt:] += shares
            cash_series.loc[dt:] -= shares * px
        elif row['type'] == 'sell':
            shares_series.loc[dt:] = 0
            cash_series.loc[dt:] += shares * px

    # 3. 計算每日浮動權益 = 現金 + 持股 x 收盤價
    equity_series = cash_series + shares_series * price_df['close']

    # 4. 繪製圖表
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=equity_series, name='淨值', line=dict(color='dodgerblue')))
    fig.add_trace(go.Scatter(x=dates, y=cash_series, name='現金', line=dict(color='limegreen')))
    fig.update_layout(
        title='每日淨值與現金曲線',
        xaxis_title='日期',
        yaxis_title='數值',
        template='plotly_white'
    )
    return fig
def display_metrics_flex(metrics: dict):
    """
    將 metrics 這個字典裡面的 (指標名 → 數值) 轉成 HTML Flexbox 格式自動換行的卡片顯示.
    """
    # 先把 metrics 的 key/value 轉成「顯示用的 label」和「格式化後的 value 字串」
    items = []
    for k, v in metrics.items():
        # 依據 k 決定要不要以百分比、或小數、或純文字來格式化
        if k in ["total_return", "annual_return", "win_rate", "max_drawdown", "annualized_volatility", "avg_win", "avg_loss"]:
            txt = f"{v:.2%}" if pd.notna(v) else ""
        elif k in ["calmar_ratio", "sharpe_ratio", "sortino_ratio", "payoff_ratio", "profit_factor"]:
            txt = f"{v:.2f}" if pd.notna(v) else ""
        elif k in ["max_drawdown_duration", "avg_holding_period"]:
            txt = f"{v:.1f} 天" if pd.notna(v) else ""
        elif k in ["num_trades", "max_consecutive_wins", "max_consecutive_losses"]:
            txt = str(int(v)) if pd.notna(v) else ""
        else:
            # 其他就先盡量當純文字顯示
            txt = f"{v}"
        # 把字典 key → 中文顯示 label
        label_map = {
            "total_return": "總回報率",
            "annual_return": "年化回報率",
            "win_rate": "勝率",
            "max_drawdown": "最大回撤",
            "max_drawdown_duration": "回撤持續",
            "calmar_ratio": "卡瑪比率",
            "sharpe_ratio": "夏普比率",
            "sortino_ratio": "索提諾比率",
            "payoff_ratio": "盈虧比",
            "profit_factor": "盈虧因子",
            "num_trades": "交易次數",
            "avg_holding_period": "平均持倉天數",
            "annualized_volatility": "年化波動率",
            "max_consecutive_wins": "最大連續盈利",
            "max_consecutive_losses": "最大連續虧損",
            "avg_win": "平均盈利",
            "avg_loss": "平均虧損",
        }
        label = label_map.get(k, k)
        items.append((label, txt))

    # 開始產生 HTML:外層一個 flex container,內層每組(指標+數值) 都是 flex item
    html = """
<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;">
"""
    for label, val in items:
        html += f"""
  <div style="flex:0 1 150px;border:1px solid #444;border-radius:4px;padding:8px 12px;background:#1a1a1a;">
    <div style="font-size:14px;color:#aaa;">{label}</div>
    <div style="font-size:20px;font-weight:bold;color:#fff;margin-top:4px;">{val}</div>
  </div>
"""
    html += "</div>"

    # 去掉多餘縮排,避免開頭空白被解讀成 code block
    html = textwrap.dedent(html)

    st.markdown(html, unsafe_allow_html=True)

def display_strategy_summary(strategy: str, params: Dict, metrics: Dict, smaa_source: str, trade_df: pd.DataFrame):
    """
    顯示策略參數與回測績效摘要,使用 HTML Flexbox 卡片展示.
    """
    # 參數展示
    param_display = {k: v for k, v in params.items() if k != "strategy_type"}
    st.write("**參數設定**: " + ", ".join(f"{k}: {v}" for k, v in param_display.items()))

    # 計算平均持倉天數
    avg_holding_period = calculate_holding_periods(trade_df)

    # 將平均持倉天數加入 metrics
    metrics['avg_holding_period'] = avg_holding_period

    # 績效指標展示
    if metrics:
        display_metrics_flex(metrics)
    else:
        st.warning("尚未執行回測,無法顯示績效指標.")
# --- 主應用程式 ---
def run_app():
    st.set_page_config(layout="wide")
    st.sidebar.title("00631L策略系統")    
    page = st.sidebar.selectbox("📑 頁面導航", ["策略回測", "各版本沿革紀錄"])

    try:
        from version_history import get_version_history_html
        if page == "各版本沿革紀錄":
            html = get_version_history_html()
            st.markdown(html, unsafe_allow_html=True)
            return
    except ImportError:
        st.warning("無法載入版本歷史記錄,請確保 version_history.py 存在.")
        if page == "各版本沿革紀錄":
            st.markdown("")
            return

    # 側邊欄
    st.sidebar.header("設定")
    default_tickers = ["00631L.TW", "2330.TW", "AAPL", "VOO"]
    ticker = st.sidebar.selectbox("股票代號:", default_tickers, index=0)
    start_date_input = st.sidebar.text_input("數據起始日期 (YYYY-MM-DD):", "2010-01-01")
    end_date_input = st.sidebar.text_input("數據結束日期 (YYYY-MM-DD):", "")
    trade_cooldown_bars = st.sidebar.number_input("冷卻期 (bars):", min_value=0, max_value=20, value=3, step=1, format="%d")
    discount = st.sidebar.slider("券商折數(0.7=7折, 0.1=1折)", min_value=0.1, max_value=0.70, value=0.30, step=0.01)
    st.sidebar.markdown("買進手續費 = 0.1425% x 折數,賣出成本 = 0.1425% x 折數 + 0.3%(交易稅).")
    bad_holding = st.sidebar.checkbox("賣出報酬率<-20%,等待下次賣點", value=False)
    run_backtests = st.sidebar.button("🚀 一鍵執行所有回測")

    # 載入數據
    df_raw, df_factor = load_data(ticker, start_date=start_date_input, 
                                  end_date=end_date_input if end_date_input else None, 
                                  smaa_source="Self")
    if df_raw.empty:
        st.error(f"無法載入 {ticker} 的數據.請檢查股票代碼或稍後重試.")
        return




    # 回測標籤頁
    strategy_names = list(param_presets.keys())
    tabs = st.tabs(strategy_names + ["買賣點比較"])
    results = {}

    if run_backtests:
        with st.spinner("正在計算所有回測..."):
            for strategy in strategy_names:
                params = param_presets[strategy]
                strategy_type = params["strategy_type"]
# 取得預設的 smaa_source 用於初始化 selectbox
                default_source = params.get("smaa_source", "Self")
                # 使用 selectbox 的值作為實際的 smaa_source
                smaa_source = default_source  # 在批量回測時使用預設值
                # 載入數據
                df_raw, df_factor = load_data(
ticker,
start_date=start_date_input,
end_date=end_date_input if end_date_input else None,
smaa_source=smaa_source
                )
                
                if strategy_type == 'ssma_turn':
                    calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'signal_cooldown_days', 'quantile_win']
                    ssma_params = {k: v for k, v in params.items() if k in calc_keys}
                    backtest_params = ssma_params.copy()
                    backtest_params['stop_loss'] = params.get('stop_loss', 0.0)
                    df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(
                        df_raw, df_factor, **ssma_params, smaa_source=smaa_source
                    )
                    if df_ind.empty:
                        st.warning(f"{strategy} 策略計算失敗,可能是數據不足.")
                        continue
                    result = backtest_unified(
                        df_ind, strategy_type, backtest_params, buy_dates, sell_dates,
                        discount=discount, trade_cooldown_bars=trade_cooldown_bars, bad_holding=bad_holding
                    )
                else:
                    if strategy_type == 'single':
                        df_ind = compute_single(
                            df_raw, df_factor, params["linlen"], params["factor"], params["smaalen"],
                            params["devwin"], smaa_source=smaa_source
                        )
                    elif strategy_type == 'dual':
                        df_ind = compute_dual(
                            df_raw, df_factor, params["linlen"], params["factor"], params["smaalen"],
                            params["short_win"], params["long_win"], smaa_source=smaa_source
                        )
                    elif strategy_type == 'RMA':
                        df_ind = compute_RMA(
                            df_raw, df_factor, params["linlen"], params["factor"], params["smaalen"],
                            params["rma_len"], params["dev_len"], smaa_source=smaa_source
                        )
                    if df_ind.empty:
                        st.warning(f"{strategy} 策略計算失敗,可能是數據不足.")
                        continue
                    result = backtest_unified(
                        df_ind, strategy_type, params, discount=discount,
                        trade_cooldown_bars=trade_cooldown_bars, bad_holding=bad_holding
                    )

                results[strategy] = (df_ind, result['trades'], result['trade_df'],
                                     result['signals_df'], result['metrics'])

    for tab, strategy in zip(tabs[:-1], strategy_names):
        with tab:
            col1, col2 = st.columns([3, 1])
            with col1:
                # 使用 Markdown 顯示策略說明
                if strategy in ["Single ★", "Single ●", "Single ▲"]:
                    tooltip = f"{strategy} 是 Single 策略的參數變體,展示不同參數表現."
                elif strategy == "Dual-Scale":
                    tooltip = f"{strategy} 是 Dual-Scale 策略,使用短期和長期 SMAA 進行交易."
                elif strategy in ["SSMA_turn_1", "SSMA_turn_2"]:
                    tooltip = f"{strategy} 是基於 SMAA 峰谷檢測的成交量過濾策略,專注於穩健的買賣時機."
                else:
                    tooltip = "策略說明"

                # 使用 HTML 和 CSS 實現氣泡提示
                html = f"""
                <h3>
                    <span title="{tooltip}" style="cursor: help; font-weight: bold;">
                        回測策略: {strategy}
                    </span>
                </h3>
                """
                st.markdown(html, unsafe_allow_html=True)

            with col2:
                # 取得預設的 smaa_source 用於初始化 selectbox
                params = param_presets[strategy]
                default_source = params.get("smaa_source", "Self")
                options = ["Self", "Factor (^TWII / 2412.TW)", "Factor (^TWII / 2414.TW)"]
                # 設置 selectbox，初始值為 default_source
                smaa_source = st.selectbox(
                    "SMAA 數據源:",
                    options,
                    index=options.index(default_source) if default_source in options else 0,
                    key=f"smaa_source_{strategy}"
                )

            strategy_type = params["strategy_type"]
            # 使用 selectbox 的 smaa_source 載入數據
            df_raw, df_factor = load_data(
                ticker,
                start_date=start_date_input,
                end_date=end_date_input if end_date_input else None,
                smaa_source=smaa_source
            )
            
            # 計算回測結果
            if strategy_type == 'ssma_turn':
                calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 
                             'buy_shift', 'exit_shift', 'vol_window', 'signal_cooldown_days', 
                             'quantile_win']
                ssma_params = {k: v for k, v in params.items() if k in calc_keys}
                # 2) 確保 stop_loss 傳遞給 backtest_unified
                backtest_params = ssma_params.copy()
                backtest_params['stop_loss'] = params.get('stop_loss', 0.0)
                df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(
                    df_raw, df_factor, **ssma_params, smaa_source=smaa_source
                )
                if df_ind.empty:
                    st.warning(f"{strategy} 策略計算失敗,可能是數據不足.")
                    continue
                result = backtest_unified(
                    df_ind, strategy_type, backtest_params, buy_dates, sell_dates,
                    discount=discount, trade_cooldown_bars=trade_cooldown_bars, bad_holding=bad_holding
                )
            else:
                if strategy_type == 'single':
                    df_ind = compute_single(
                        df_raw, df_factor, params["linlen"], params["factor"], params["smaalen"],
                        params["devwin"], smaa_source=smaa_source
                    )
                elif strategy_type == 'dual':
                    df_ind = compute_dual(
                        df_raw, df_factor, params["linlen"], params["factor"], params["smaalen"],
                        params["short_win"], params["long_win"], smaa_source=smaa_source
                    )
                elif strategy_type == 'RMA':
                    df_ind = compute_RMA(
                        df_raw, df_factor, params["linlen"], params["factor"], params["smaalen"],
                        params["rma_len"], params["dev_len"], smaa_source=smaa_source
                    )
                if df_ind.empty:
                    st.warning(f"{strategy} 策略計算失敗,可能是數據不足.")
                    continue
                result = backtest_unified(
                    df_ind, strategy_type, params, discount=discount,
                    trade_cooldown_bars=trade_cooldown_bars, bad_holding=bad_holding
                )
            
            results[strategy] = (df_ind, result['trades'], result['trade_df'], 
                                 result['signals_df'], result['metrics'])
            
            # 顯示策略摘要,傳入 trade_df 以計算平均持倉天數
            display_strategy_summary(strategy, params, result['metrics'], smaa_source, result['trade_df'])
            
            # 顯示圖表與交易明細
            if result['trades']:
                st.plotly_chart(plot_stock_price(df_raw, result['trade_df'], ticker), 
                                use_container_width=True, key=f"stock_price_{strategy}")

                st.plotly_chart(plot_equity_cash(result['trade_df'], df_raw), 
                                use_container_width=True, key=f"equity_cash_{strategy}")
                st.plotly_chart(plot_indicators(df_ind, strategy_type, result['trade_df'], params), 
                                use_container_width=True, key=f"indicators_{strategy}")                
                st.subheader("交易明細")
                trade_df = result['trade_df'].copy()
                # 只顯示日期
                for col in ['signal_date', 'trade_date']:
                    if col in trade_df.columns:
                        trade_df[col] = pd.to_datetime(trade_df[col]).dt.date
                # price, return 只顯示兩位小數
                if 'price' in trade_df.columns:
                    trade_df['price'] = trade_df['price'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
                if 'return' in trade_df.columns:
                    trade_df['return'] = trade_df['return'].apply(lambda x: "-" if pd.isna(x) else f"{x:.2%}")
                st.dataframe(trade_df)
                st.markdown("**所有下單日皆為信號日的隔天（T+1），本表 signal_date=trade_date 代表信號日即下單日**")
            else:
                st.warning(f"{strategy} 策略未產生任何交易,可能是參數設置或數據問題.")

    # 買賣點比較標籤頁
    with tabs[-1]:
        st.subheader("所有策略買賣點比較")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['close'], name='Close Price', line=dict(color='dodgerblue')))
        colors = ['green', 'limegreen', 'red', 'orange', 'purple', 'blue', 'pink', 'cyan']
        for i, strategy in enumerate(strategy_names):
            if strategy in results and not results[strategy][2].empty:
                trade_df = results[strategy][2]
                buys = trade_df[trade_df['type'] == 'buy']
                sells = trade_df[trade_df['type'] == 'sell']
                fig.add_trace(go.Scatter(x=buys['trade_date'], y=buys['price'], mode='markers', name=f'{strategy} Buy',
                                         marker=dict(symbol='cross', size=8, color=colors[i % len(colors)])))
                fig.add_trace(go.Scatter(x=sells['trade_date'], y=sells['price'], mode='markers', name=f'{strategy} Sell',
                                         marker=dict(symbol='x', size=8, color=colors[i % len(colors)])))
        fig.update_layout(title=f'{ticker} 所有策略買賣點比較',
                          xaxis_title='Date', yaxis_title='股價', template='plotly_white')
        fig.update_layout(
            legend=dict(
                x=1.05,
                y=1,
                xanchor='left',
                yanchor='top',
                bordercolor="Black",
                borderwidth=1,
                bgcolor="white",
                itemsizing='constant',
                orientation='v'
            )
        )
        st.plotly_chart(fig, use_container_width=True, key="buy_sell_comparison")
        
        st.subheader("績效比較表")
        comparison_data = []
        for strategy in strategy_names:
            if strategy in results and results[strategy][4]:
                metrics = results[strategy][4]
                comparison_data.append({
                    '策略': strategy,
                    '總回報率': f"{metrics.get('total_return', 0):.2%}",
                    '年化回報率': f"{metrics.get('annual_return', 0):.2%}",
                    '最大回撤': f"{metrics.get('max_drawdown', 0):.2%}",
                    '卡瑪比率': f"{metrics.get('calmar_ratio', 0):.2f}",
                    '交易次數': metrics.get('num_trades', 0),
                    '勝率': f"{metrics.get('win_rate', 0):.2%}",
                    '盈虧比': f"{metrics.get('payoff_ratio', 0):.2f}"
                })
        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data))
        else:
            st.warning("無可比較的回測結果,請先執行回測.")

if __name__ == "__main__":
    run_app()                    