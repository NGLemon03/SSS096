# -*- coding: utf-8 -*-
"""
SSSv096.py - 股票策略系統 v0.96

主要功能：
1. 單一策略回測（single, dual, RMA）
2. 轉向策略回測（ssma_turn）
3. 統一回測框架
4. 績效指標計算
5. 圖表繪製

作者：SSS Team
版本：v0.96
日期：2025-01-12
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import warnings
import logging

# 配置 logger
from analysis.logging_config import LOGGING_DICT
import logging.config
logging.config.dictConfig(LOGGING_DICT)
logger = logging.getLogger("SSS.Core")

# 忽略 pandas 的 PerformanceWarning
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

__all__ = [
    "load_data", "compute_single", "compute_dual", "compute_RMA",
    "compute_ssma_turn_combined", "backtest_unified",
    "compute_backtest_for_periods", "calculate_metrics",
]
VERSION = "096"

# === 系統與標準函式庫 ===
import os
import json
import random
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
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

def _coerce_trade_schema(df):
    """檢查是否需要下載股價數據的剎車機制"""
    import pandas as pd, numpy as np
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["trade_date","type","price"])

    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]

    # trade_date
    if "trade_date" not in out.columns:
        if "date" in out.columns:
            out["trade_date"] = pd.to_datetime(out["date"], errors="coerce")
        elif isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "trade_date"})
        else:
            out["trade_date"] = pd.NaT
    else:
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")

    # type
    if "type" not in out.columns:
        if "action" in out.columns:
            out["type"] = out["action"].astype(str).str.lower()
        elif "side" in out.columns:
            out["type"] = out["side"].astype(str).str.lower()
        elif "dw" in out.columns:
            out["type"] = np.where(out["dw"]>0, "buy", np.where(out["dw"]<0, "sell", "hold"))
        else:
            out["type"] = "hold"

    # price
    if "price" not in out.columns:
        for c in ["open","price_open","exec_price","px","close"]:
            if c in out.columns:
                out["price"] = out[c]
                break
        if "price" not in out.columns:
            out["price"] = np.nan

    return out.sort_values("trade_date")

# --- 專案結構與日誌設定 ---
from analysis import config as cfg
from analysis.logging_config import setup_logging
import logging
DATA_DIR = cfg.DATA_DIR
LOG_DIR = cfg.LOG_DIR
CACHE_DIR = cfg.CACHE_DIR
# 全局費率常數
BASE_FEE_RATE = 0.001425 # 基礎手續費 = 0.1425%
TAX_RATE = 0.003 # 賣出交易税率 = 0.3%


# 預計算 SMAA
param_presets = {

"Single 2": {"linlen": 90, "factor": 40, "smaalen": 30, "devwin": 30, "buy_mult": 1.45, "sell_mult": 1.25,"stop_loss":0.2, "strategy_type": "single", "smaa_source": "Self"},
"single_1887": {"linlen": 93, "smaalen": 27, "devwin": 30, "factor": 40, "buy_mult": 1.55, "sell_mult": 0.95, "stop_loss": 0.4, "prom_factor": 0.5, "min_dist": 5, "strategy_type": "single", "smaa_source": "Self"},

"Single 3": {"linlen": 80, "factor": 10, "smaalen": 60, "devwin": 20, "buy_mult": 0.4, "sell_mult": 1.5, "strategy_type": "single", "smaa_source": "Self"},
"RMA_69": {"linlen": 151, "smaalen": 162, "rma_len": 55, "dev_len": 40, "factor": 40, "buy_mult": 1.4, "sell_mult": 3.15, "stop_loss": 0.1, "prom_factor": 0.5, "min_dist": 5, "strategy_type": "RMA", "smaa_source": "Factor (^TWII / 2414.TW)"},
"RMA_669": {"linlen": 178, "smaalen": 112, "rma_len": 95, "dev_len": 95, "factor": 40, "buy_mult": 1.7, "sell_mult": 0.9, "stop_loss": 0.4, "prom_factor": 0.5, "min_dist": 5, "strategy_type": "RMA", "smaa_source": "Self"},
"STM0": {"linlen": 25, "smaalen": 85, "factor": 80.0, "prom_factor": 9, "min_dist": 8, "buy_shift": 0, "exit_shift": 6, "vol_window": 90, "quantile_win": 65, "signal_cooldown_days": 7, "buy_mult": 0.15, "sell_mult": 0.1, "stop_loss": 0.13, 
                "strategy_type": "ssma_turn", "smaa_source": "Factor (^TWII / 2414.TW)"},
"STM1": {"linlen": 15,"smaalen": 40,"factor": 40.0,"prom_factor": 70,"min_dist": 10,"buy_shift": 6,"exit_shift": 4,"vol_window": 40,"quantile_win": 65,
 "signal_cooldown_days": 10,"buy_mult": 1.55,"sell_mult": 2.1,"stop_loss": 0.15,"strategy_type": "ssma_turn","smaa_source": "Self"},
"STM3": {"linlen": 20,"smaalen": 40,"factor": 40.0,"prom_factor": 69,"min_dist": 10,"buy_shift": 6,"exit_shift": 4,"vol_window": 45,"quantile_win": 55,
    "signal_cooldown_days": 10,"buy_mult": 1.65,"sell_mult": 2.1,"stop_loss": 0.2,"strategy_type": "ssma_turn","smaa_source": "Self"},
"STM4": {"linlen": 10,"smaalen": 35,"factor": 40.0,"prom_factor": 68,"min_dist": 8,"buy_shift": 6,"exit_shift": 0,"vol_window": 40,"quantile_win": 65,
    "signal_cooldown_days": 10,"buy_mult": 1.6,"sell_mult": 2.2,"stop_loss": 0.15,"strategy_type": "ssma_turn","smaa_source": "Factor (^TWII / 2414.TW)"},
"STM_1939":{'linlen': 20, 'smaalen': 240, 'factor': 40.0, 'prom_factor': 48, 'min_dist': 14, 'buy_shift': 1, 'exit_shift': 1, 'vol_window': 80, 'quantile_win': 175, 'signal_cooldown_days': 4, 'buy_mult': 1.45, 'sell_mult': 2.6, 'stop_loss': 0.2,
                "strategy_type": "ssma_turn", "smaa_source": "Self"},

"STM_2414_273": {"linlen": 175, "smaalen": 10, "factor": 40.0, "prom_factor": 47, "min_dist": 5, "buy_shift": 0, "exit_shift": 0, "vol_window": 90, "quantile_win": 165, "signal_cooldown_days": 4, "buy_mult": 1.25, "sell_mult": 1.7, "stop_loss": 0.4, "strategy_type": "ssma_turn", "smaa_source": "Factor (^TWII / 2414.TW)"},

# 多策略組合策略（使用小型網格掃描優化後的參數）
"Ensemble_Majority": {
    "strategy_type": "ensemble",
    "method": "majority",
    "params": {
        "floor": 0.2,
        "ema_span": 3,
        "min_cooldown_days": 1,
        "delta_cap": 0.3,
        "min_trade_dw": 0.01,
        # "majority_k": 66,   # 120 檔時 ≈ 55% 門檻（你這次最佳）
        # （可選）若之後檔數會變動，想用比例門檻： "majority_k_pct": 0.55
    },
    "trade_cost": {        # 成本你已經在模組中支援，這裡啟用即可
        "discount_rate": 0.3,
        "buy_fee_bp": 4.27,
        "sell_fee_bp": 4.27,
        "sell_tax_bp": 30.0,
    },
    "ticker": "00631L.TW",
    # === 修復：使用比例門檻避免 N 變動時失真 ===
    "majority_k_pct": 0.55,  # 55% 門檻，會根據實際策略數量自動調整
},

"Ensemble_Proportional": {
    "strategy_type": "ensemble",
    "method": "proportional",
    "params": {
        "floor": 0.2,
        "ema_span": 3,
        "min_cooldown_days": 1,
        "delta_cap": 0.2,   # 你最後一輪（含成本、120檔）最佳
        "min_trade_dw": 0.03
    },
    "trade_cost": {
        "discount_rate": 0.3,
        "buy_fee_bp": 4.27,
        "sell_fee_bp": 4.27,
        "sell_tax_bp": 30.0,
    },
    "ticker": "00631L.TW",
}

}
setup_logging()  # 初始化統一日誌設定
logger = logging.getLogger("SSSv096")  # 使用專屬 logger
from functools import wraps

@dataclass
class TradeSignal:
    ts: pd.Timestamp
    side: str  # "BUY", "SELL", "FORCE_SELL", "STOP_LOSS"
    reason: str

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
        st.error(f"日期格式錯誤: {e}")
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

def is_price_data_up_to_date(csv_path):
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            last_date = pd.to_datetime(df['date'].iloc[-1])
        else:
            last_date = pd.to_datetime(df.iloc[-1, 0])
        today = pd.Timestamp.now(tz='Asia/Taipei').normalize()
        return last_date >= today
    except Exception:
        return False

def clear_all_caches():
    """清除所有快取"""
    try:
        # 清除 joblib 快取
        cfg.MEMORY.clear()
        logger.info("已清除 joblib 快取")
        
        # 清除 SMAA 快取目錄
        smaa_cache_dir = CACHE_DIR / "cache_smaa"
        if smaa_cache_dir.exists():
            import shutil
            shutil.rmtree(smaa_cache_dir)
            smaa_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("已清除 SMAA 快取目錄")
        
        # 清除 optuna 快取
        optuna_cache_dir = CACHE_DIR / "optuna16_equity"
        if optuna_cache_dir.exists():
            import shutil
            shutil.rmtree(optuna_cache_dir)
            optuna_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("已清除 Optuna 快取目錄")
            
        return True
    except Exception as e:
        logger.error(f"清除快取時發生錯誤: {e}")
        return False

def force_update_price_data(ticker: str = None):
    """強制更新股價數據"""
    try:
        if ticker:
            # 更新指定股票
            filename = DATA_DIR / f"{ticker.replace(':','_')}_data_raw.csv"
            fetch_yf_data(ticker, filename, "2000-01-01")
            logger.info(f"已強制更新 {ticker} 股價數據")
        else:
            # 更新所有常用股票
            common_tickers = ["00631L.TW", "^TWII", "2414.TW", "2412.TW"]
            for t in common_tickers:
                filename = DATA_DIR / f"{t.replace(':','_')}_data_raw.csv"
                fetch_yf_data(t, filename, "2000-01-01")
            logger.info("已強制更新所有常用股票數據")
        return True
    except Exception as e:
        logger.error(f"強制更新股價數據時發生錯誤: {e}")
        return False

@cfg.MEMORY.cache
def load_data(ticker: str, start_date: str = "2000-01-01", end_date: Optional[str] = None, smaa_source: str = "Self", force_update: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加載並清理數據,支持指定結束日期與因子數據.
    
    Args:
        ticker: 股票代號
        start_date: 起始日期
        end_date: 結束日期
        smaa_source: SMAA數據源
        force_update: 是否強制更新股價數據
    """
    filename = DATA_DIR / f"{ticker.replace(':','_')}_data_raw.csv"
    
    # 根據參數決定是否更新股價
    if force_update:
        fetch_yf_data(ticker, filename, start_date, end_date)
    else:
        # 啟動時自動檢查股價csv是否為今日，若不是則自動更新
        if not is_price_data_up_to_date(filename):
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
            # 檢查 twii/factor 是否為今日，若不是則自動更新
            if force_update or not is_price_data_up_to_date(twii_file):
                fetch_yf_data("^TWII", twii_file, start_date, end_date)
            if force_update or not is_price_data_up_to_date(factor_file):
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

@cfg.MEMORY.cache
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

@cfg.MEMORY.cache
def compute_single(df: pd.DataFrame, smaa_source_df: pd.DataFrame, linlen: int, factor: float, smaalen: int, devwin: int, smaa_source: str = "Self") -> pd.DataFrame:
    source_df = smaa_source_df if not smaa_source_df.empty else df
    df_cleaned = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)  # 0001FIX:標準化精度
    
    # 直接計算 SMAA，讓 joblib.Memory 處理快取
    smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)
    
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

@cfg.MEMORY.cache
def compute_dual(df: pd.DataFrame, smaa_source_df: pd.DataFrame, linlen: int, factor: float, smaalen: int, short_win: int, long_win: int, smaa_source: str = "Self") -> pd.DataFrame:
    source_df = smaa_source_df if not smaa_source_df.empty else df
    df_cleaned = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)  # 0001FIX:標準化精度
    
    # 直接計算 SMAA，讓 joblib.Memory 處理快取
    smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)
    
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

@cfg.MEMORY.cache
def compute_RMA(
    df: pd.DataFrame,
    smaa_source_df: pd.DataFrame,
    linlen: int,
    factor: float,
    smaalen: int,
    rma_len: int,
    dev_len: int,
    smaa_source: str = "Self"
) -> pd.DataFrame:
    """
    rma_len: 用於 EMA(base) 的視窗
    dev_len: 用於 rolling std(sd) 的視窗
    smaa_source: 快取鍵,從呼叫端傳入
    """
    # 準備資料
    source_df   = smaa_source_df if not smaa_source_df.empty else df
    df_cleaned  = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)  # 0001FIX:標準化精度
    
    # 直接計算 SMAA，讓 joblib.Memory 處理快取
    smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)

    # 計算 base / sd
    base = smaa.ewm(alpha=1/rma_len, adjust=False, min_periods=rma_len).mean()
    sd   = smaa.rolling(window=dev_len, min_periods=dev_len).std()

    # 組成最終 DataFrame
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

@cfg.MEMORY.cache
def compute_ssma_turn_combined(
    df: pd.DataFrame, smaa_source_df: pd.DataFrame, linlen: int, factor: float,
    smaalen: int, prom_factor: float, min_dist: int, buy_shift: int = 0, exit_shift: int = 0, vol_window: int = 20,
    signal_cooldown_days: int = 10, quantile_win: int = 100,
    smaa_source: str = "Self"
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

    # 直接計算 SMAA，讓 joblib.Memory 處理快取
    smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)

    series_clean = smaa.dropna()
    if series_clean.empty:
        logger.warning(f"SMAA 資料為空,無法進行峯谷檢測, valid_smaa={len(series_clean)}, linlen={linlen}, smaalen={smaalen}")
        st.warning(f"SMAA 資料為空,無法進行峯谷檢測, valid_smaa={len(series_clean)}, linlen={linlen}, smaalen={smaalen}")
        return pd.DataFrame(), [], []

 

    # 閾值計算(使用舊版 ptp 邏輯)
    prom = series_clean.rolling(window=min_dist+1, min_periods=min_dist+1).apply(lambda x: x.ptp(), raw=True)
    initial_threshold = prom.quantile(prom_factor / 100) if len(prom.dropna()) > 0 else prom.median()
    threshold_series = prom.rolling(window=quantile_win, min_periods=quantile_win).quantile(prom_factor / 100).shift(1).ffill().fillna(initial_threshold)
    
    # 峯谷檢測(滾動窗口,放寬條件)
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

    # 處理 ensemble 策略類型
    if strategy_type == "ensemble":
        try:
            from SSS_EnsembleTab import run_ensemble, EnsembleParams, CostParams, RunConfig
            # 創建配置
            ensemble_params = EnsembleParams(
                floor=params.get("floor", 0.2),
                ema_span=params.get("ema_span", 3),
                delta_cap=params.get("delta_cap", 0.3),
                majority_k=params.get("majority_k", 6),
                min_cooldown_days=params.get("min_cooldown_days", 1),  # 與param_presets一致
                min_trade_dw=params.get("min_trade_dw", 0.01)          # 與param_presets一致
            )
            
            # === 第3步：統一路徑與preset，確保app與SSS使用相同的參數 ===
            # 強制使用比例門檻，避免N變動時失真
            if params.get("method") == "majority":
                if params.get("majority_k_pct"):
                    logger.info(f"[Ensemble] 使用比例門檻 majority_k_pct={params.get('majority_k_pct')}")
                else:
                    # 如果沒有majority_k_pct，強制使用0.55
                    params["majority_k_pct"] = 0.55
                    logger.info(f"[Ensemble] 強制設定 majority_k_pct=0.55")
                # majority_k 會在 SSS_EnsembleTab 中根據實際策略數量動態調整
            
            cost_params = CostParams(
                buy_fee_bp=params.get("buy_fee_bp", 4.27),  # 與param_presets一致
                sell_fee_bp=params.get("sell_fee_bp", 4.27), # 與param_presets一致
                sell_tax_bp=params.get("sell_tax_bp", 30.0)  # 與param_presets一致
            )
            # 優先使用 params 中的 ticker，然後嘗試從 df_ind 推斷，最後使用默認值
            ticker_name = params.get("ticker")
            if not ticker_name:
                # 嘗試從 df_ind 的 name 屬性獲取
                ticker_name = getattr(df_ind, 'name', None)
                if not ticker_name:
                    # 最後使用默認值
                    ticker_name = "00631L.TW"
            
            cfg = RunConfig(
                ticker=ticker_name,
                method=params.get("method", "majority"),
                params=ensemble_params,
                cost=cost_params
            )
            
            # === 修復：傳遞比例門檻參數 ===
            if params.get("majority_k_pct"):
                cfg.majority_k_pct = params.get("majority_k_pct")
            
            logger.info(f"[Ensemble] 執行配置: ticker={ticker_name}, method={params.get('method')}, majority_k_pct={params.get('majority_k_pct', 'N/A')}")
            
            # 運行 ensemble 策略
            try:
                open_px, w, trades, stats, method_name, equity, daily_state, trade_ledger = run_ensemble(cfg)
                
                # 直接使用 run_ensemble 計算的權益曲線，避免重複計算
                equity_curve = equity
                
                # 構造交易記錄
                trade_records = []
                if not trades.empty:
                    for _, row in trades.iterrows():
                        trade_records.append({
                            'signal_date': row['trade_date'],
                            'trade_date': row['trade_date'],
                            'type': row['type'],
                            'price': row['price_open'],
                            'shares': 1.0,  # 簡化處理
                            'return': 0.0   # 簡化處理
                        })
                
                trade_df = pd.DataFrame(trade_records)
                trades_df = pd.DataFrame()  # 簡化處理
                signals_df = pd.DataFrame()  # 簡化處理
                
                # 構造指標
                metrics = {
                    'total_return': stats.get('total_return', 0.0),
                    'annual_return': stats.get('annual_return', 0.0),
                    'max_drawdown': stats.get('max_drawdown', 0.0),
                    'sharpe_ratio': stats.get('sharpe_ratio', 0.0),
                    'calmar_ratio': stats.get('calmar_ratio', 0.0),
                    'num_trades': len(trade_records)
                }
                
                logger.info(f"[Ensemble] 執行成功: {method_name}, 權益曲線長度={len(equity_curve)}, 交易數={len(trade_records)}")
                
            except FileNotFoundError as e:
                # 如果沒有找到 trades_*.csv 文件，創建一個模擬的 ensemble 結果
                logger.warning(f"Ensemble 策略找不到 trades_*.csv 文件，創建模擬結果: {e}")
                
                # 創建模擬的權益曲線
                equity_curve = pd.Series(1.0, index=df_ind.index)
                
                # 創建空的交易記錄
                trade_df = pd.DataFrame()
                trades_df = pd.DataFrame()
                signals_df = pd.DataFrame()
                
                # 創建模擬指標
                metrics = {
                    'total_return': 0.0,
                    'annual_return': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'calmar_ratio': 0.0,
                    'num_trades': 0
                }
            
            logger.info(f"Ensemble {method_name} 回測結果: 總報酬率 = {metrics.get('total_return', 0):.2%}, 交易次數={metrics.get('num_trades', 0)}")
            return {'trades': [], 'trade_df': trade_df, 'trades_df': trades_df, 'signals_df': signals_df, 'metrics': metrics, 'equity_curve': equity_curve}
            
        except ImportError as e:
            logger.error(f"無法導入 SSS_EnsembleTab 模塊: {e}")
            return {'trades': [], 'trade_df': pd.DataFrame(), 'trades_df': pd.DataFrame(), 'signals_df': pd.DataFrame(), 'metrics': {'total_return': 0.0, 'annual_return': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0, 'calmar_ratio': 0.0, 'num_trades': 0}, 'equity_curve': pd.Series()}
        except Exception as e:
            logger.error(f"Ensemble 策略執行失敗: {e}")
            return {'trades': [], 'trade_df': pd.DataFrame(), 'trades_df': pd.DataFrame(), 'signals_df': pd.DataFrame(), 'metrics': {'total_return': 0.0, 'annual_return': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0, 'calmar_ratio': 0.0, 'num_trades': 0}, 'equity_curve': pd.Series()}

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
        return {'trades': [], 'trade_df': pd.DataFrame(), 'trades_df': pd.DataFrame(), 'signals_df': pd.DataFrame(), 'metrics': {'total_return': 0.0, 'annual_return': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0, 'calmar_ratio': 0.0, 'num_trades': 0}, 'equity_curve': pd.Series()}

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
                                     discount=discount, trade_cooldown_bars=trade_cooldown_bars, bad_holding=bad_holding
                    )
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


# --- 新增：把各種交易格式轉成畫圖要的統一規格 ---
def normalize_trades_for_plots(trades_df: pd.DataFrame, price_series: pd.Series | None = None) -> pd.DataFrame:
    """把各種交易格式轉成畫圖要的統一規格：trade_date + type + price"""
    import pandas as pd
    import numpy as np
    
    if trades_df is None or len(trades_df) == 0:
        return pd.DataFrame(columns=["trade_date", "type", "price"])

    t = trades_df.copy()

    # 動作欄位：支援 type / side / action
    if "type" not in t.columns:
        for c in ("side", "action"):
            if c in t.columns:
                t["type"] = t[c].astype(str).str.lower()
                break
        else:
            # 如果都沒有找到動作欄位，返回空表
            return pd.DataFrame(columns=["trade_date", "type", "price"])

    # 日期欄位：支援 trade_date / date / index 是時間
    if "trade_date" not in t.columns:
        if "date" in t.columns:
            t["trade_date"] = pd.to_datetime(t["date"], errors="coerce")
        elif isinstance(t.index, pd.DatetimeIndex):
            t["trade_date"] = t.index
        else:
            # 如果都沒有找到日期欄位，返回空表
            return pd.DataFrame(columns=["trade_date", "type", "price"])
    
    t["trade_date"] = pd.to_datetime(t["trade_date"], errors="coerce")

    # 價格欄位：優先用 price；否則用 open；都沒有就從 price_series 對齊
    if "price" not in t.columns:
        if "open" in t.columns:
            t["price"] = pd.to_numeric(t["open"], errors="coerce")
        elif price_series is not None:
            # 從 price_series 對齊當日價格
            s = price_series.rename("open").reset_index()
            s.columns = ["trade_date", "open"]
            t = t.merge(s, on="trade_date", how="left")
            t["price"] = t["open"]
        else:
            # 如果都沒有價格資訊，返回空表
            return pd.DataFrame(columns=["trade_date", "type", "price"])

    # 只保留畫圖必要欄位（統一規格）
    keep = ["trade_date", "type", "price"]
    t = t[[c for c in keep if c in t.columns]]
    
    # 過濾掉無效資料
    t = t.dropna(subset=["trade_date", "type", "price"])
    
    return t

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
    # 先規格化交易資料，確保有統一的 trade_date + type + price 欄位
    trades_df = normalize_trades_for_plots(trades_df, price_series=df.get("open", df["close"]))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close Price', line=dict(color='dodgerblue')))
    
    if not trades_df.empty:
        # 現在 trades_df 已經有統一的 trade_date + type + price 欄位
        buys = trades_df[trades_df['type'] == 'buy']
        adds = trades_df[trades_df['type'] == 'add'] if 'add' in trades_df['type'].values else pd.DataFrame()
        sells = trades_df[trades_df['type'] == 'sell']
        
        # 繪製買入信號
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys['trade_date'], 
                y=buys['price'], 
                mode='markers', 
                name='Buy',
                marker=dict(symbol='cross', size=10, color='green')
            ))
        
        # 繪製加碼信號
        if not adds.empty:
            fig.add_trace(go.Scatter(
                x=adds['trade_date'], 
                y=adds['price'], 
                mode='markers', 
                name='Add-On Buy',
                marker=dict(symbol='cross', size=10, color='limegreen')
            ))
        
        # 繪製賣出信號
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells['trade_date'], 
                y=sells['price'], 
                mode='markers', 
                name='Sell',
                marker=dict(symbol='x', size=10, color='red')
            ))
    
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

    # 先規格化交易資料，確保有統一的 trade_date + type + price 欄位
    trades_df = normalize_trades_for_plots(trades_df, price_series=df_ind.get("smaa"))

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

def plot_equity_cash(trades_or_ds: pd.DataFrame, df_raw: pd.DataFrame | None = None) -> go.Figure:
    """
    優先接受含 ['equity','cash'] 欄位的 daily_state。
    若傳進來的是交易表（沒有 equity/cash），才嘗試由交易重建。
    
    Args:
        trades_or_ds: 每日狀態 DataFrame 或交易表 DataFrame
        df_raw: 價格數據 DataFrame（用於重建權益曲線）
    
    Returns:
        go.Figure: Plotly 圖表物件
    """
    if isinstance(trades_or_ds, pd.DataFrame) and {'equity','cash'}.issubset(trades_or_ds.columns):
        # 優先使用 daily_state
        ds = trades_or_ds.copy()
        if not np.issubdtype(ds.index.dtype, np.datetime64):
            ds.index = pd.to_datetime(ds.index)
        ds = ds.sort_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ds.index, y=ds['equity'], name='Equity', line=dict(color='dodgerblue')))
        fig.add_trace(go.Scatter(x=ds.index, y=ds['cash'],   name='Cash',   line=dict(color='gray')))
        fig.update_layout(title='權益 & 現金', xaxis_title='日期', yaxis_title='金額', template='plotly_white')
        return fig
    else:
        # 舊路徑：交易表 → 標準化後重建（Ensemble 的交易表通常缺 shares，不保證成功）
        try:
            from SSS_EnsembleTab import normalize_trades_for_plots
            trades_df = normalize_trades_for_plots(trades_or_ds, price_series=df_raw.get('open') if df_raw is not None else None)
            ds = reconstruct_equity_cash_from_trades(trades_df, df_raw)  # 你既有的重建邏輯
            
            # 保底欄位
            for col in ['equity', 'cash']:
                if col not in ds.columns:
                    raise KeyError(f"daily_state 缺少必要欄位: {col}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ds.index, y=ds['equity'], name='Equity', line=dict(color='dodgerblue')))
            fig.add_trace(go.Scatter(x=ds.index, y=ds['cash'],   name='Cash',   line=dict(color='gray')))
            fig.update_layout(title='權益 & 現金', xaxis_title='日期', yaxis_title='金額', template='plotly_white')
            return fig
        except Exception as e:
            logger.warning(f"無法從交易表重建權益曲線: {e}")
            return go.Figure()


def plot_weight_series(daily_state: pd.DataFrame, trades_df: pd.DataFrame = None) -> go.Figure:
    """
    繪製持有權重變化圖，並標示變化點
    
    Args:
        daily_state: 每日狀態 DataFrame，需包含權重相關欄位
        trades_df: 交易資料 DataFrame，用於標示變化點（可選）
    
    Returns:
        go.Figure: Plotly 圖表物件
    """
    ds = daily_state.copy()
    if not isinstance(ds.index, pd.DatetimeIndex):
        ds.index = pd.to_datetime(ds.index, errors='coerce')
    ds = ds.sort_index()

    # 優先順序：w → invested_pct → cash_pct（用 1 - cash_pct）
    if 'w' in ds.columns:
        w = ds['w']
    elif 'invested_pct' in ds.columns:
        w = ds['invested_pct']
    elif 'cash_pct' in ds.columns:
        w = 1 - ds['cash_pct']
    else:
        return go.Figure()  # 無可用欄位時回傳空圖

    fig = go.Figure()
    
    # 主要權重曲線
    fig.add_trace(go.Scatter(
        x=w.index, 
        y=w, 
        name='持有權重',
        line=dict(color='dodgerblue', width=2),
        mode='lines'
    ))
    
    # 如果有交易資料，標示變化點
    if trades_df is not None and not trades_df.empty:
        # 確保交易資料有必要的欄位
        if 'trade_date' in trades_df.columns and 'type' in trades_df.columns:
            # 過濾有效的交易日期
            valid_trades = trades_df[
                (trades_df['trade_date'].notna()) & 
                (trades_df['trade_date'].isin(w.index))
            ].copy()
            
            if not valid_trades.empty:
                # 將交易日期轉換為datetime
                valid_trades['trade_date'] = pd.to_datetime(valid_trades['trade_date'])
                
                # 為每筆交易找到對應的權重值
                trade_points = []
                for _, trade in valid_trades.iterrows():
                    trade_date = trade['trade_date']
                    if trade_date in w.index:
                        weight_value = w.loc[trade_date]
                        trade_type = trade['type']
                        
                        # 根據交易類型決定標記樣式
                        if trade_type in ('buy', 'add'):
                            marker_symbol = 'triangle-up'
                            marker_color = 'green'
                            marker_size = 10
                        elif trade_type == 'sell':
                            marker_symbol = 'triangle-down'
                            marker_color = 'red'
                            marker_size = 10
                        else:
                            marker_symbol = 'circle'
                            marker_color = 'orange'
                            marker_size = 8
                        
                        trade_points.append({
                            'date': trade_date,
                            'weight': weight_value,
                            'type': trade_type,
                            'symbol': marker_symbol,
                            'color': marker_color,
                            'size': marker_size
                        })
                
                # 添加變化點標記
                if trade_points:
                    # 買入/加碼點
                    buy_points = [p for p in trade_points if p['type'] in ('buy', 'add')]
                    if buy_points:
                        fig.add_trace(go.Scatter(
                            x=[p['date'] for p in buy_points],
                            y=[p['weight'] for p in buy_points],
                            mode='markers',
                            name='買入/加碼',
                            marker=dict(
                                symbol='triangle-up',
                                size=10,
                                color='green',
                                line=dict(width=1, color='darkgreen')
                            ),
                            hovertemplate='<b>買入/加碼</b><br>日期: %{x}<br>權重: %{y:.4f}<extra></extra>'
                        ))
                    
                    # 賣出點
                    sell_points = [p for p in trade_points if p['type'] == 'sell']
                    if sell_points:
                        fig.add_trace(go.Scatter(
                            x=[p['date'] for p in sell_points],
                            y=[p['weight'] for p in sell_points],
                            mode='markers',
                            name='賣出',
                            marker=dict(
                                symbol='triangle-down',
                                size=10,
                                color='red',
                                line=dict(width=1, color='darkred')
                            ),
                            hovertemplate='<b>賣出</b><br>日期: %{x}<br>權重: %{y:.4f}<extra></extra>'
                        ))
                    
                    # 其他類型交易點
                    other_points = [p for p in trade_points if p['type'] not in ('buy', 'add', 'sell')]
                    if other_points:
                        fig.add_trace(go.Scatter(
                            x=[p['date'] for p in other_points],
                            y=[p['weight'] for p in other_points],
                            mode='markers',
                            name='其他交易',
                            marker=dict(
                                symbol='circle',
                                size=8,
                                color='orange',
                                line=dict(width=1, color='darkorange')
                            ),
                            hovertemplate='<b>其他交易</b><br>日期: %{x}<br>權重: %{y:.4f}<extra></extra>'
                        ))
    
    # 更新圖表佈局
    fig.update_layout(
        title='持有權重變化',
        xaxis_title='日期',
        yaxis_title='權重(0~1)',
        template='plotly_white',
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def reconstruct_equity_cash_from_trades(trades_df: pd.DataFrame, df_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    從交易表重建權益和現金曲線
    
    Args:
        trades_df: 標準化後的交易表，需有 trade_date, type, price 欄位
        df_raw: 價格數據 DataFrame，用於計算權益
        
    Returns:
        DataFrame: 包含 equity, cash 欄位的每日狀態表
    """
    if trades_df is None or trades_df.empty or df_raw is None or df_raw.empty:
        return pd.DataFrame()
    
    # 確保有必要的欄位
    required_cols = ['trade_date', 'type', 'price']
    if not all(col in trades_df.columns for col in required_cols):
        logger.warning(f"交易表缺少必要欄位: {required_cols}")
        return pd.DataFrame()
    
    # 建立每日時間軸
    dates = df_raw.index
    initial_cash = 1_000_000.0  # 假設初始資金
    
    cash_series = pd.Series(initial_cash, index=dates, dtype=float)
    shares_series = pd.Series(0, index=dates, dtype=float)
    
    # 標準化交易表
    trades_df = trades_df.copy()
    trades_df['trade_date'] = pd.to_datetime(trades_df['trade_date'])
    trades_df['price'] = pd.to_numeric(trades_df['price'], errors='coerce')
    
    # 逐筆交易更新持股和現金
    for _, row in trades_df.iterrows():
        dt = row['trade_date']
        px = row['price']
        
        if pd.isna(dt) or pd.isna(px) or dt not in dates:
            continue
            
        # 假設每次交易都是 1000 股（因為 Ensemble 交易表通常沒有 shares 欄位）
        shares = 1000
        
        if row['type'] in ('buy', 'add'):
            shares_series.loc[dt:] += shares
            cash_series.loc[dt:] -= shares * px
        elif row['type'] == 'sell':
            shares_series.loc[dt:] = 0
            cash_series.loc[dt:] += shares * px
    
    # 計算每日浮動權益 = 現金 + 持股 x 收盤價
    equity_series = cash_series + shares_series * df_raw['close']
    
    # 構建結果 DataFrame
    result = pd.DataFrame({
        'equity': equity_series,
        'cash': cash_series
    })
    
    return result
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
    page = st.sidebar.selectbox("📑 頁面導航", ["策略回測", "投資組合權重分析", "各版本沿革紀錄", "快取管理"])

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

    # 快取管理頁面
    if page == "快取管理":
        st.title("快取管理")
        st.write("管理系統快取和股價數據更新")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("清除快取")
            if st.button("🗑️ 清除所有快取"):
                if clear_all_caches():
                    st.success("✅ 所有快取已清除")
                else:
                    st.error("❌ 清除快取失敗")
            
            st.write("**清除內容包括：**")
            st.write("- Joblib 計算快取")
            st.write("- SMAA 指標快取")
            st.write("- Optuna 權益曲線快取")
        
        with col2:
            st.subheader("股價數據更新")
            ticker_to_update = st.selectbox(
                "選擇要更新的股票：",
                ["全部更新", "00631L.TW", "^TWII", "2414.TW", "2412.TW"],
                index=0
            )
            
            if st.button("🔄 強制更新股價"):
                if ticker_to_update == "全部更新":
                    success = force_update_price_data()
                else:
                    success = force_update_price_data(ticker_to_update)
                
                if success:
                    st.success(f"✅ {ticker_to_update} 股價數據已更新")
                else:
                    st.error(f"❌ {ticker_to_update} 股價數據更新失敗")
        
        st.subheader("快取狀態")
        cache_info = []
        
        # 檢查各類快取
        try:
            # Joblib 快取
            joblib_cache_size = len(list(cfg.MEMORY.store_backend.get_items()))
            cache_info.append(f"Joblib 快取: {joblib_cache_size} 個項目")
        except:
            cache_info.append("Joblib 快取: 無法檢查")
        
        # SMAA 快取
        smaa_cache_dir = CACHE_DIR / "cache_smaa"
        if smaa_cache_dir.exists():
            smaa_files = len(list(smaa_cache_dir.glob("*.joblib")))
            cache_info.append(f"SMAA 快取: {smaa_files} 個檔案")
        else:
            cache_info.append("SMAA 快取: 目錄不存在")
        
        # Optuna 快取
        optuna_cache_dir = CACHE_DIR / "optuna16_equity"
        if optuna_cache_dir.exists():
            optuna_files = len(list(optuna_cache_dir.glob("*.npy")))
            cache_info.append(f"Optuna 快取: {optuna_files} 個檔案")
        else:
            cache_info.append("Optuna 快取: 目錄不存在")
        
        for info in cache_info:
            st.write(f"• {info}")
        
        return

    # 投資組合權重分析頁面
    if page == "投資組合權重分析":
        st.title("📊 投資組合權重分析")
        st.write("分析多策略投資組合的權重分配與績效表現")
        
        # 側邊欄設定
        st.sidebar.header("投資組合設定")
        
        # 選擇策略
        strategy_names = list(param_presets.keys())
        selected_strategies = st.sidebar.multiselect(
            "選擇策略組合:",
            strategy_names,
            default=strategy_names[:5]  # 預設選擇前5個策略
        )
        
        if not selected_strategies:
            st.warning("請至少選擇一個策略")
            return
        
        # 權重分配方法
        weight_method = st.sidebar.selectbox(
            "權重分配方法:",
            ["等權重", "夏普比率權重", "方差倒數權重", "動態調整"],
            index=0
        )
        
        # 重新平衡頻率
        rebalance_freq = st.sidebar.selectbox(
            "重新平衡頻率:",
            ["季度", "年度", "不重新平衡"],
            index=1
        )
        
        # 回看期間
        lookback_period = st.sidebar.slider(
            "回看期間 (交易日):",
            min_value=60,
            max_value=504,  # 約2年
            value=252,  # 約1年
            step=21
        )
        
        # 基本設定
        ticker = st.sidebar.selectbox("股票代號:", ["00631L.TW", "2330.TW", "AAPL", "VOO"], index=0)
        start_date = st.sidebar.text_input("起始日期:", "2010-01-01")
        end_date = st.sidebar.text_input("結束日期:", "")
        discount = st.sidebar.slider("券商折數:", min_value=0.1, max_value=0.70, value=0.30, step=0.01)
        
        # 執行分析按鈕
        run_analysis = st.sidebar.button("🚀 執行投資組合分析")
        
        if run_analysis:
            with st.spinner("正在計算投資組合分析..."):
                # 載入數據
                df_raw, df_factor = load_data(ticker, start_date=start_date, 
                                             end_date=end_date if end_date else None, 
                                             smaa_source="Self", force_update=False)
                
                if df_raw.empty:
                    st.error(f"無法載入 {ticker} 的數據")
                    return
                
                # 計算各策略回測結果
                strategy_results = {}
                equity_curves = {}
                
                for strategy in selected_strategies:
                    params = param_presets[strategy]
                    strategy_type = params["strategy_type"]
                    smaa_source = params.get("smaa_source", "Self")
                    
                    # 載入對應的數據
                    df_raw_strategy, df_factor_strategy = load_data(
                        ticker, start_date=start_date, 
                        end_date=end_date if end_date else None,
                        smaa_source=smaa_source, force_update=False
                    )
                    
                    # 計算策略指標
                    if strategy_type == 'ssma_turn':
                        calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 
                                     'buy_shift', 'exit_shift', 'vol_window', 'signal_cooldown_days', 
                                     'quantile_win']
                        ssma_params = {k: v for k, v in params.items() if k in calc_keys}
                        backtest_params = ssma_params.copy()
                        backtest_params['stop_loss'] = params.get('stop_loss', 0.0)
                        
                        df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(
                            df_raw_strategy, df_factor_strategy, **ssma_params, smaa_source=smaa_source
                        )
                        
                        if not df_ind.empty:
                            result = backtest_unified(
                                df_ind, strategy_type, backtest_params, buy_dates, sell_dates,
                                discount=discount, trade_cooldown_bars=3, bad_holding=False
                            )
                            strategy_results[strategy] = result
                            if 'equity_curve' in result:
                                equity_curves[strategy] = result['equity_curve']
                    else:
                        if strategy_type == 'single':
                            df_ind = compute_single(
                                df_raw_strategy, df_factor_strategy, params["linlen"], params["factor"], 
                                params["smaalen"], params["devwin"], smaa_source=smaa_source
                            )
                        elif strategy_type == 'dual':
                            df_ind = compute_dual(
                                df_raw_strategy, df_factor_strategy, params["linlen"], params["factor"], 
                                params["smaalen"], params["short_win"], params["long_win"], smaa_source=smaa_source
                            )
                        elif strategy_type == 'RMA':
                            df_ind = compute_RMA(
                                df_raw_strategy, df_factor_strategy, params["linlen"], params["factor"], 
                                params["smaalen"], params["rma_len"], params["dev_len"], smaa_source=smaa_source
                            )
                        
                        if not df_ind.empty:
                            result = backtest_unified(
                                df_ind, strategy_type, params, discount=discount,
                                trade_cooldown_bars=3, bad_holding=False
                            )
                            strategy_results[strategy] = result
                            if 'equity_curve' in result:
                                equity_curves[strategy] = result['equity_curve']
                
                if not equity_curves:
                    st.error("沒有可用的權益曲線數據")
                    return
                
                # 計算投資組合權重和績效
                portfolio_analysis = calculate_portfolio_weights_and_performance(
                    equity_curves, weight_method, rebalance_freq, lookback_period
                )
                
                # 顯示結果
                display_portfolio_analysis(portfolio_analysis, selected_strategies, weight_method, rebalance_freq)
        
        return

    # 側邊欄
    st.sidebar.header("設定")
    default_tickers = ["00631L.TW", "2330.TW", "AAPL", "VOO"]
    ticker = st.sidebar.selectbox("股票代號:", default_tickers, index=0)
    start_date_input = st.sidebar.text_input("數據起始日期 (YYYY-MM-DD):", "2010-01-01")
    end_date_input = st.sidebar.text_input("數據結束日期 (YYYY-MM-DD):", "")
    trade_cooldown_bars = st.sidebar.number_input("冷卻期 (bars):", min_value=0, max_value=20, value=3, step=1, format="%d")
    discount = st.sidebar.slider("券商折數(0.7=7折, 0.1=1折)", min_value=0.1, max_value=0.70, value=0.30, step=0.01)
    st.sidebar.markdown("買進手續費 = 0.1425% x 折數,賣出成本 = 0.1425% x 折數 + 0.3%(交易税).")
    bad_holding = st.sidebar.checkbox("賣出報酬率<-20%,等待下次賣點", value=False)
    
    # 添加快取管理選項
    st.sidebar.header("快取管理")
    force_update = st.sidebar.checkbox("強制更新股價數據", value=False)
    clear_cache_before_run = st.sidebar.checkbox("執行前清除快取", value=False)
    
    run_backtests = st.sidebar.button("🚀 一鍵執行所有回測")

    # 載入數據
    df_raw, df_factor = load_data(ticker, start_date=start_date_input, 
                                  end_date=end_date_input if end_date_input else None, 
                                  smaa_source="Self", force_update=force_update)
    if df_raw.empty:
        st.error(f"無法載入 {ticker} 的數據.請檢查股票代碼或稍後重試.")
        return

    # 回測標籤頁
    strategy_names = list(param_presets.keys())
    tabs = st.tabs(strategy_names + ["買賣點比較"])
    results = {}

    if run_backtests:
        # 執行前清除快取
        if clear_cache_before_run:
            clear_all_caches()
            st.info("已清除所有快取")
        
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
smaa_source=smaa_source,
force_update=force_update
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
                # 使用 Markdown 顯示策略説明
                if strategy in ["Single ★", "Single ●", "Single ▲"]:
                    tooltip = f"{strategy} 是 Single 策略的參數變體,展示不同參數表現."
                elif strategy == "Dual-Scale":
                    tooltip = f"{strategy} 是 Dual-Scale 策略,使用短期和長期 SMAA 進行交易."
                elif strategy in ["SSMA_turn_1", "SSMA_turn_2"]:
                    tooltip = f"{strategy} 是基於 SMAA 峯谷檢測的成交量過濾策略,專注於穩健的買賣時機."
                else:
                    tooltip = "策略説明"

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
                smaa_source=smaa_source,
                force_update=force_update
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
            has_trades = ('trade_df' in result) and (result['trade_df'] is not None) and (not result['trade_df'].empty)
            if has_trades:
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

# 新增投資組合分析函數
def calculate_portfolio_weights_and_performance(equity_curves, weight_method, rebalance_freq, lookback_period):
    """計算投資組合權重和績效"""
    if not equity_curves:
        return {}
    
    # 找到共同的時間範圍
    all_dates = set()
    for equity_curve in equity_curves.values():
        all_dates.update(equity_curve.index)
    
    common_dates = sorted(list(all_dates))
    if len(common_dates) < 2:
        return {}
    
    # 初始化投資組合
    portfolio_equity = pd.Series(index=common_dates, dtype=float)
    portfolio_equity.iloc[0] = 1.0
    
    # 權重歷史記錄
    weight_history = {strategy: [] for strategy in equity_curves.keys()}
    date_history = []
    
    # 初始權重
    n_strategies = len(equity_curves)
    current_weights = {strategy: 1.0 / n_strategies for strategy in equity_curves.keys()}
    
    # 重新平衡頻率轉換
    if rebalance_freq == "季度":
        rebalance_months = 3
    elif rebalance_freq == "年度":
        rebalance_months = 12
    else:
        rebalance_months = None
    
    # 計算重新平衡日期
    rebalance_dates = []
    if rebalance_months:
        current_date = pd.to_datetime(common_dates[0])
        end_date = pd.to_datetime(common_dates[-1])
        
        while current_date <= end_date:
            rebalance_dates.append(current_date)
            # 計算下一個重新平衡日期
            if current_date.month + rebalance_months > 12:
                year = current_date.year + 1
                month = (current_date.month + rebalance_months) % 12
                if month == 0:
                    month = 12
            else:
                year = current_date.year
                month = current_date.month + rebalance_months
            
            current_date = current_date.replace(year=year, month=month)
    
    # 執行投資組合計算
    for i, date in enumerate(common_dates):
        if i == 0:
            continue
        
        # 檢查是否需要重新平衡
        if rebalance_months and pd.to_datetime(date) in rebalance_dates:
            # 計算新的權重
            performance_metrics = {}
            for strategy_name, equity_curve in equity_curves.items():
                if date in equity_curve.index:
                    # 計算過去期間的表現
                    lookback_start = max(0, i - lookback_period)
                    if lookback_start < len(equity_curve):
                        past_equity = equity_curve.iloc[lookback_start:i+1]
                        if len(past_equity) > 1:
                            if weight_method == "等權重":
                                performance_metrics[strategy_name] = 1.0
                            elif weight_method == "夏普比率權重":
                                returns = past_equity.pct_change().dropna()
                                if len(returns) > 0 and returns.std() > 0:
                                    sharpe = returns.mean() / returns.std() * np.sqrt(252)
                                    performance_metrics[strategy_name] = max(sharpe, 0)
                                else:
                                    performance_metrics[strategy_name] = 0
                            elif weight_method == "方差倒數權重":
                                returns = past_equity.pct_change().dropna()
                                if len(returns) > 0 and returns.std() > 0:
                                    performance_metrics[strategy_name] = 1.0 / (returns.std() ** 2)
                                else:
                                    performance_metrics[strategy_name] = 0
                            elif weight_method == "動態調整":
                                performance = (past_equity.iloc[-1] / past_equity.iloc[0]) - 1
                                performance_metrics[strategy_name] = max(performance, 0)
                            else:
                                performance_metrics[strategy_name] = 1.0
                        else:
                            performance_metrics[strategy_name] = 0
                    else:
                        performance_metrics[strategy_name] = 0
                else:
                    performance_metrics[strategy_name] = 0
            
            # 重新計算權重
            total_performance = sum(performance_metrics.values())
            if total_performance > 0:
                current_weights = {name: perf / total_performance for name, perf in performance_metrics.items()}
            else:
                # 如果沒有正收益，使用等權重
                current_weights = {name: 1.0 / n_strategies for name in equity_curves.keys()}
        
        # 記錄權重
        date_history.append(date)
        for strategy in equity_curves.keys():
            weight_history[strategy].append(current_weights.get(strategy, 0))
        
        # 計算當日組合價值
        daily_return = 0
        for strategy_name, equity_curve in equity_curves.items():
            if date in equity_curve.index and i > 0:
                if i > 0 and common_dates[i-1] in equity_curve.index:
                    strategy_return = (equity_curve.loc[date] / equity_curve.loc[common_dates[i-1]]) - 1
                    daily_return += strategy_return * current_weights[strategy_name]
        
        # 更新組合權益
        portfolio_equity.loc[date] = portfolio_equity.loc[common_dates[i-1]] * (1 + daily_return)
    
    # 計算投資組合指標
    total_return = (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0]) - 1
    years = (pd.to_datetime(common_dates[-1]) - pd.to_datetime(common_dates[0])).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # 計算最大回撤
    cumulative_max = portfolio_equity.expanding().max()
    drawdown = (portfolio_equity - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    # 計算夏普比率
    daily_returns = portfolio_equity.pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    return {
        'portfolio_equity': portfolio_equity,
        'weight_history': weight_history,
        'date_history': date_history,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'rebalance_freq': rebalance_freq,
        'weight_method': weight_method,
        'individual_equity_curves': equity_curves
    }

def display_portfolio_analysis(portfolio_analysis, selected_strategies, weight_method, rebalance_freq):
    """顯示投資組合分析結果"""
    if not portfolio_analysis:
        st.error("無法計算投資組合分析")
        return
    
    # 創建標籤頁
    tabs = st.tabs(["📈 權益曲線", "⚖️ 權重歷史", "📊 績效指標", "🔍 權重檢查"])
    
    # 權益曲線標籤頁
    with tabs[0]:
        st.subheader("投資組合與個別策略權益曲線")
        
        fig = go.Figure()
        
        # 添加投資組合權益曲線
        portfolio_equity = portfolio_analysis['portfolio_equity']
        fig.add_trace(go.Scatter(
            x=portfolio_equity.index,
            y=portfolio_equity.values,
            name='投資組合',
            line=dict(color='red', width=3),
            mode='lines'
        ))
        
        # 添加個別策略權益曲線
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        for i, strategy in enumerate(selected_strategies):
            if strategy in portfolio_analysis['individual_equity_curves']:
                equity_curve = portfolio_analysis['individual_equity_curves'][strategy]
                fig.add_trace(go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve.values,
                    name=strategy,
                    line=dict(color=colors[i % len(colors)], width=1),
                    mode='lines'
                ))
        
        fig.update_layout(
            title=f"投資組合權益曲線 ({weight_method}, {rebalance_freq})",
            xaxis_title="日期",
            yaxis_title="權益",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 權重歷史標籤頁
    with tabs[1]:
        st.subheader("策略權重歷史變化")
        
        # 創建權重歷史圖表
        fig = go.Figure()
        
        weight_history = portfolio_analysis['weight_history']
        date_history = portfolio_analysis['date_history']
        
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        for i, strategy in enumerate(selected_strategies):
            if strategy in weight_history:
                weights = weight_history[strategy]
                fig.add_trace(go.Scatter(
                    x=date_history,
                    y=weights,
                    name=strategy,
                    line=dict(color=colors[i % len(colors)]),
                    mode='lines',
                    stackgroup='one'
                ))
        
        fig.update_layout(
            title=f"策略權重歷史 ({weight_method}, {rebalance_freq})",
            xaxis_title="日期",
            yaxis_title="權重",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 顯示權重統計
        st.subheader("權重統計")
        weight_stats = []
        for strategy in selected_strategies:
            if strategy in weight_history:
                weights = weight_history[strategy]
                weight_stats.append({
                    '策略': strategy,
                    '平均權重': f"{np.mean(weights):.3f}",
                    '最大權重': f"{np.max(weights):.3f}",
                    '最小權重': f"{np.min(weights):.3f}",
                    '權重標準差': f"{np.std(weights):.3f}"
                })
        
        if weight_stats:
            st.dataframe(pd.DataFrame(weight_stats))
    
    # 績效指標標籤頁
    with tabs[2]:
        st.subheader("投資組合績效指標")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("總回報率", f"{portfolio_analysis['total_return']:.2%}")
        
        with col2:
            st.metric("年化回報率", f"{portfolio_analysis['annual_return']:.2%}")
        
        with col3:
            st.metric("最大回撤", f"{portfolio_analysis['max_drawdown']:.2%}")
        
        with col4:
            st.metric("夏普比率", f"{portfolio_analysis['sharpe_ratio']:.3f}")
        
        # 顯示詳細指標
        st.subheader("詳細績效分析")
        
        # 計算額外指標
        portfolio_equity = portfolio_analysis['portfolio_equity']
        daily_returns = portfolio_equity.pct_change().dropna()
        
        # 計算下行風險
        downside_returns = daily_returns[daily_returns < 0]
        downside_risk = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # 計算索提諾比率
        sortino_ratio = portfolio_analysis['annual_return'] / downside_risk if downside_risk > 0 else 0
        
        # 計算卡瑪比率
        calmar_ratio = portfolio_analysis['annual_return'] / abs(portfolio_analysis['max_drawdown']) if portfolio_analysis['max_drawdown'] < 0 else 0
        
        # 計算正報酬月份比例
        monthly_returns = portfolio_equity.resample('M').last().pct_change().dropna()
        positive_months = (monthly_returns > 0).sum() / len(monthly_returns) if len(monthly_returns) > 0 else 0
        
        detailed_metrics = {
            '指標': ['總回報率', '年化回報率', '年化波動率', '最大回撤', '夏普比率', '索提諾比率', '卡瑪比率', '正報酬月份比例'],
            '數值': [
                f"{portfolio_analysis['total_return']:.2%}",
                f"{portfolio_analysis['annual_return']:.2%}",
                f"{daily_returns.std() * np.sqrt(252):.2%}",
                f"{portfolio_analysis['max_drawdown']:.2%}",
                f"{portfolio_analysis['sharpe_ratio']:.3f}",
                f"{sortino_ratio:.3f}",
                f"{calmar_ratio:.3f}",
                f"{positive_months:.2%}"
            ]
        }
        
        st.dataframe(pd.DataFrame(detailed_metrics))
    
    # 權重檢查標籤頁
    with tabs[3]:
        st.subheader("權重檢查工具")
        
        # 日期選擇器
        if portfolio_analysis['date_history']:
            selected_date = st.selectbox(
                "選擇檢查日期:",
                portfolio_analysis['date_history'],
                index=len(portfolio_analysis['date_history']) - 1
            )
            
            # 顯示選定日期的權重
            st.subheader(f"日期: {selected_date} 的權重分配")
            
            date_index = portfolio_analysis['date_history'].index(selected_date)
            current_weights = {}
            
            for strategy in selected_strategies:
                if strategy in portfolio_analysis['weight_history']:
                    weight = portfolio_analysis['weight_history'][strategy][date_index]
                    current_weights[strategy] = weight
            
            # 創建權重圓餅圖
            fig = go.Figure(data=[go.Pie(
                labels=list(current_weights.keys()),
                values=list(current_weights.values()),
                hole=0.3
            )])
            
            fig.update_layout(
                title=f"權重分配 ({selected_date})",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 顯示權重表格
            weight_df = pd.DataFrame([
                {'策略': strategy, '權重': f"{weight:.3f}"}
                for strategy, weight in current_weights.items()
            ])
            
            st.dataframe(weight_df)
            
            # 顯示權重變化趨勢
            st.subheader("權重變化趨勢")
            
            # 選擇要顯示的策略
            selected_strategies_for_trend = st.multiselect(
                "選擇要顯示的策略:",
                selected_strategies,
                default=selected_strategies[:3]
            )
            
            if selected_strategies_for_trend:
                fig = go.Figure()
                
                colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
                for i, strategy in enumerate(selected_strategies_for_trend):
                    if strategy in portfolio_analysis['weight_history']:
                        weights = portfolio_analysis['weight_history'][strategy]
                        fig.add_trace(go.Scatter(
                            x=portfolio_analysis['date_history'],
                            y=weights,
                            name=strategy,
                            line=dict(color=colors[i % len(colors)])
                        ))
                
                fig.update_layout(
                    title="選定策略權重變化趨勢",
                    xaxis_title="日期",
                    yaxis_title="權重",
                    template="plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_app()                    