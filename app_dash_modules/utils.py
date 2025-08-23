# app_dash_modules/utils.py / 2025-08-23 03:07
import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import json
import io
from dash.dependencies import ALL
import shutil
import os
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from analysis import config as cfg
import yfinance as yf
import logging
import numpy as np
from urllib.parse import quote as urlparse

# 配置 logger - 使用統一日誌系統（按需初始化）
from analysis.logging_config import get_logger, init_logging
import os

# 設定環境變數（但不立即初始化）
os.environ["SSS_CREATE_LOGS"] = "1"

# 獲取日誌器（懶加載）
logger = get_logger("SSS.App")

def normalize_daily_state_columns(ds: pd.DataFrame) -> pd.DataFrame:
    """將不同來源的 daily_state 欄位語意統一：
    - 若 equity 實為倉位市值，改名為 position_value
    - 建立 portfolio_value = position_value + cash
    - 保證有 invested_pct / cash_pct
    """
    if ds is None or ds.empty:
        return ds
    ds = ds.copy()

    # 若已經有 position_value 與 cash，直接建立 portfolio_value
    if {'position_value','cash'}.issubset(ds.columns):
        ds['portfolio_value'] = ds['position_value'] + ds['cash']

    # 僅有 equity + cash 的情況 -> 判斷 equity 是總資產還是倉位
    elif {'equity','cash'}.issubset(ds.columns):
        # 判斷規則：若 equity/(equity+cash) 的中位數顯著 < 0.9，較像「倉位市值」
        ratio = (ds['equity'] / (ds['equity'] + ds['cash'])).replace([np.inf, -np.inf], np.nan).clip(0,1)
        if ratio.median(skipna=True) < 0.9:
            # 把 equity 當成倉位市值
            ds = ds.rename(columns={'equity':'position_value'})
            ds['portfolio_value'] = ds['position_value'] + ds['cash']
        else:
            # equity 已是總資產，反推倉位（若沒有 position_value）
            if 'position_value' not in ds.columns:
                ds['position_value'] = (ds['equity'] - ds['cash']).fillna(0.0)
            ds['portfolio_value'] = ds['equity']

    # 百分比欄位統一
    if 'portfolio_value' in ds.columns:
        pv = ds['portfolio_value'].replace(0, np.nan)
        if 'invested_pct' not in ds.columns and 'position_value' in ds.columns:
            ds['invested_pct'] = (ds['position_value'] / pv).fillna(0.0).clip(0,1)
        if 'cash_pct' not in ds.columns and 'cash' in ds.columns:
            ds['cash_pct'] = (ds['cash'] / pv).fillna(0.0).clip(0,1)

    # 為了向下相容：保留 equity = portfolio_value（供舊繪圖函式使用）
    if 'portfolio_value' in ds.columns:
        ds['equity'] = ds['portfolio_value']

    return ds

def _initialize_app_logging():
    """初始化應用程式日誌系統"""
    # 只在實際需要時才初始化檔案日誌
    init_logging(enable_file=True)
    logger.setLevel(logging.DEBUG)
    logger.info("=== App Dash 啟動 - 統一日誌系統 ===")
    logger.info("已啟用詳細調試模式 - 調試資訊將寫入日誌檔案")
    logger.info(f"日誌目錄: {os.path.abspath('analysis/log')}")
    return logger

# ATR 計算函數
def calculate_atr(df, window):
    """計算 ATR (Average True Range)"""
    try:
        # app_dash.py / 2025-08-22 14:30
        # 統一 OHLC 欄位對應：收盤、最高、最低價
        high_col = None
        low_col = None
        close_col = None
        
        # 優先檢查英文欄位名稱（標準格式）
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            high_col = 'high'
            low_col = 'low'
            close_col = 'close'
        # 檢查大寫英文欄位名稱
        elif 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
            high_col = 'High'
            low_col = 'Low'
            close_col = 'Close'
        # 檢查中文欄位名稱
        elif '最高價' in df.columns and '最低價' in df.columns and '收盤價' in df.columns:
            high_col = '最高價'
            low_col = '最低價'
            close_col = '收盤價'
        # 檢查其他可能的欄位名稱（降級處理）
        elif 'open' in df.columns and 'close' in df.columns:
            # 如果沒有高低價，用開盤價和收盤價近似
            high_col = 'open'
            low_col = 'close'
            close_col = 'close'
        
        if high_col and low_col and close_col:
            # 有高低價時，計算 True Range
            high = df[high_col]
            low = df[low_col]
            close = df[close_col]
            
            # 確保數據為數值型
            high = pd.to_numeric(high, errors='coerce')
            low = pd.to_numeric(low, errors='coerce')
            close = pd.to_numeric(close, errors='coerce')
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=window).mean()
        else:
            # 只有收盤價時，用價格變化近似
            if close_col:
                close = pd.to_numeric(df[close_col], errors='coerce')
            elif 'close' in df.columns:
                close = pd.to_numeric(df['close'], errors='coerce')
            elif 'Close' in df.columns:
                close = pd.to_numeric(df['Close'], errors='coerce')
            else:
                # 只記一次警告，避免重複刷屏
                if not hasattr(calculate_atr, '_warning_logged'):
                    logger.warning("找不到可用的價格欄位來計算 ATR，降級為 ATR-only 模式")
                    calculate_atr._warning_logged = True
                return pd.Series(index=df.index, dtype=float)
            
            price_change = close.diff().abs()
            atr = price_change.rolling(window=window).mean()
        
        # 檢查計算結果
        if atr is None or atr.empty or atr.isna().all():
            if not hasattr(calculate_atr, '_warning_logged'):
                logger.warning(f"ATR 計算結果無效，window={window}，降級為 ATR-only 模式")
                calculate_atr._warning_logged = True
            return pd.Series(index=df.index, dtype=float)
        
        return atr
    except Exception as e:
        if not hasattr(calculate_atr, '_warning_logged'):
            logger.warning(f"ATR 計算失敗: {e}，降級為 ATR-only 模式")
            calculate_atr._warning_logged = True
        return pd.Series(index=df.index, dtype=float)


def _build_benchmark_df(df_raw):
    """建立基準資料 DataFrame，統一處理欄位名稱和數據轉換"""
    # app_dash.py / 2025-08-22 14:30
    # 統一 OHLC 欄位對應：收盤、最高、最低價
    bench = pd.DataFrame(index=pd.to_datetime(df_raw.index))
    
    # 收盤價欄位 - 優先使用英文欄位，回退到中文欄位
    if 'close' in df_raw.columns:
        bench["收盤價"] = pd.to_numeric(df_raw["close"], errors="coerce")
    elif 'Close' in df_raw.columns:
        bench["收盤價"] = pd.to_numeric(df_raw["Close"], errors="coerce")
    elif '收盤價' in df_raw.columns:
        bench["收盤價"] = pd.to_numeric(df_raw["收盤價"], errors="coerce")
    
    # 最高價和最低價欄位 - 優先使用英文欄位，回退到中文欄位
    if 'high' in df_raw.columns and 'low' in df_raw.columns:
        bench["最高價"] = pd.to_numeric(df_raw["high"], errors="coerce")
        bench["最低價"] = pd.to_numeric(df_raw["low"], errors="coerce")
    elif 'High' in df_raw.columns and 'Low' in df_raw.columns:
        bench["最高價"] = pd.to_numeric(df_raw["High"], errors="coerce")
        bench["最低價"] = pd.to_numeric(df_raw["Low"], errors="coerce")
    elif '最高價' in df_raw.columns and '最低價' in df_raw.columns:
        bench["最高價"] = pd.to_numeric(df_raw["最高價"], errors="coerce")
        bench["最低價"] = pd.to_numeric(df_raw["最低價"], errors="coerce")
    
    return bench

def calculate_equity_curve(open_px, w, cap, atr_ratio):
    """計算權益曲線"""
    try:
        # 簡化的權益曲線計算
        # 這裡使用開盤價和權重的乘積來模擬權益變化
        equity = (open_px * w * cap).cumsum()
        return equity
    except Exception as e:
        logger.warning(f"權益曲線計算失敗: {e}")
        return None

def calculate_trades_from_equity(equity_curve, open_px, w, cap, atr_ratio):
    """從權益曲線計算交易記錄"""
    try:
        if equity_curve is None or equity_curve.empty:
            return None
        
        # 簡化的交易記錄生成
        # 這裡根據權重變化來識別交易
        weight_changes = w.diff().abs()
        trade_dates = weight_changes[weight_changes > 0.01].index
        
        trades = []
        for date in trade_dates:
            trades.append({
                'trade_date': date,
                'return': 0.0  # 簡化，實際應該計算報酬率
            })
        
        if trades:
            return pd.DataFrame(trades)
        else:
            return pd.DataFrame(columns=['trade_date', 'return'])
            
    except Exception as e:
        logger.warning(f"交易記錄計算失敗: {e}")
        return None

# 解包器函數：支援 pack_df/pack_series 和傳統 JSON 字串兩種格式
def df_from_pack(data):
    """從 pack_df 結果或 JSON 字串解包 DataFrame"""
    import io, json
    import pandas as pd
    
    # 如果已經是 DataFrame，直接返回
    if isinstance(data, pd.DataFrame):
        return data
    
    # 檢查是否為 None 或空字串
    if data is None:
        return pd.DataFrame()
    
    # 如果是字串，進行額外檢查
    if isinstance(data, str):
        if data == "" or data == "[]":
            return pd.DataFrame()
        # 先嘗試 split → 再退回預設
        for orient in ("split", None):
            try:
                kw = {"orient": orient} if orient else {}
                return pd.read_json(io.StringIO(data), **kw)
            except Exception:
                pass
        return pd.DataFrame()
    
    if isinstance(data, (list, dict)):
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()
    
    return pd.DataFrame()

def series_from_pack(data):
    """從 pack_series 結果或 JSON 字串解包 Series"""
    import io
    import pandas as pd
    
    # 如果已經是 Series，直接返回
    if isinstance(data, pd.Series):
        return data
    
    # 檢查是否為 None 或空字串
    if data is None:
        return pd.Series(dtype=float)
    
    # 如果是字串，進行額外檢查
    if isinstance(data, str):
        if data == "" or data == "[]":
            return pd.Series(dtype=float)
        # Series 也先試 split
        for orient in ("split", None):
            try:
                kw = {"orient": orient} if orient else {}
                return pd.read_json(io.StringIO(data), typ="series", **kw)
            except Exception:
                pass
        return pd.Series(dtype=float)
    
    if isinstance(data, (list, dict)):
        try:
            return pd.Series(data)
        except Exception:
            return pd.Series(dtype=float)
    
    return pd.Series(dtype=float)

from SSSv096 import (
    param_presets, load_data, compute_single, compute_dual, compute_RMA,
    compute_ssma_turn_combined, backtest_unified, plot_stock_price, plot_equity_cash, plot_weight_series, calculate_holding_periods
)

# 彈性匯入 pack_df/pack_series 函數
try:
    from sss_core.schemas import pack_df, pack_series
except Exception:
    from schemas import pack_df, pack_series

# 匯入權重欄位確保函式
try:
    from sss_core.normalize import _ensure_weight_columns
except Exception:
    # 如果無法匯入，定義一個空的函式作為 fallback
    def _ensure_weight_columns(df):
        return df

# 假設你有 get_version_history_html
try:
    from version_history import get_version_history_html
except ImportError:
    def get_version_history_html() -> str:
        return "<b>無法載入版本歷史記錄</b>"

# --- 保證放進 Store 的都是 JSON-safe ---
def _pack_any(x):
    import pandas as pd
    if isinstance(x, pd.DataFrame):
        return pack_df(x)          # orient="split" + date_format="iso"
    if isinstance(x, pd.Series):
        return pack_series(x)      # orient="split" + date_format="iso"
    return x

def _pack_result_for_store(result: dict) -> dict:
    # 統一把所有 pandas 物件轉成字串（JSON）
    keys = [
        'trade_df', 'trades_df', 'signals_df',
        'equity_curve', 'cash_curve', 'price_series',
        'daily_state', 'trade_ledger',
        'daily_state_std', 'trade_ledger_std',
        'weight_curve',
        # ➊ 新增：保存未套閥門 baseline
        'daily_state_base', 'trade_ledger_base', 'weight_curve_base',
        # ➋ 新增：保存 valve 版本
        'daily_state_valve', 'trade_ledger_valve', 'weight_curve_valve', 'equity_curve_valve'
    ]
    out = dict(result)
    for k in keys:
        if k in out:
            out[k] = _pack_any(out[k])
    # 另外把 datetime tuple 的 trades 轉可序列化（你原本也有做）
    if 'trades' in out and isinstance(out['trades'], list):
        out['trades'] = [
            (str(t[0]), t[1], str(t[2])) if isinstance(t, tuple) and len(t) == 3 else t
            for t in out['trades']
        ]
    return out

default_tickers = ["00631L.TW", "2330.TW", "AAPL", "VOO"]
strategy_names = list(param_presets.keys())

theme_list = ['theme-dark', 'theme-light', 'theme-blue']

def get_theme_label(theme):
    if theme == 'theme-dark':
        return '🌑 深色主題'
    elif theme == 'theme-light':
        return '🌕 淺色主題'
    else:
        return '💙 藍黃主題'

def get_column_display_name(column_name):
    """將英文欄位名轉換為中文顯示名稱"""
    column_mapping = {
        'trade_date': '交易日期',
        'signal_date': '信號日期',
        'type': '交易類型',
        'price': '價格',
        'weight_change': '權重變化',
        'w_before': '交易前權重',
        'w_after': '交易後權重',
        'delta_units': '股數變化',
        'exec_notional': '執行金額',
        'equity_after': '交易後權益',
        'cash_after': '交易後現金',
        'equity_pct': '權益%',
        'cash_pct': '現金%',
        'invested_pct': '投資比例',
        'position_value': '部位價值',
        'return': '報酬率',
        'comment': '備註'
    }
    return column_mapping.get(column_name, column_name)

# 顯示層欄名對應（→ 中文）
DISPLAY_NAME = {
    'trade_date': '交易日期',
    'signal_date': '訊號日期',
    'type': '交易類型',
    'price': '價格',
    'weight_change': '權重變化',
    'w_before': '交易前權重',
    'w_after': '交易後權重',
    'delta_units': '股數變化',
    'exec_notional': '執行金額',
    'equity_after': '交易後權益',
    'cash_after': '交易後現金',
    'equity_pct': '權益%',
    'cash_pct': '現金%',
    'invested_pct': '投資比例',
    'position_value': '部位市值',
    'return': '報酬',
    'comment': '備註',
}

# 顯示層「隱藏」欄位（計算保留、UI 不顯示）
HIDE_COLS = {
    'shares_before', 'shares_after', 'fee_buy', 'fee_sell', 'sell_tax', 'tax',
    'date', 'open', 'equity_open_after_trade'  # ← 你提到的雜欄，統一隱藏
}

# 顯示層欄位順序（存在才排，不存在就跳過）
PREFER_ORDER = [
    'trade_date','signal_date','type','price',
    'weight_change','w_before','w_after',
    'delta_units','exec_notional',
    'equity_after','cash_after','equity_pct','cash_pct',
    'invested_pct','position_value','return','comment'
]

def format_trade_like_df_for_display(df):
    """顯示層：隱藏雜欄 → 補百分比 → 格式化 → 中文欄名 → 安全排序"""
    import pandas as pd
    if df is None or len(df)==0:
        return df

    d = df.copy()

    # 1) 隱藏雜欄
    hide = [c for c in HIDE_COLS if c in d.columns]
    if hide:
        d = d.drop(columns=hide, errors='ignore')

    # 2) 必要欄位補齊百分比（若已存在就略過）
    if {'equity_after','cash_after'}.issubset(d.columns):
        tot = d['equity_after'] + d['cash_after']
        if 'equity_pct' not in d.columns:
            d['equity_pct'] = d.apply(
                lambda r: "" if pd.isna(r['equity_after']) or pd.isna(tot.loc[r.name]) or tot.loc[r.name] <= 0
                          else f"{(r['equity_after']/tot.loc[r.name]):.2%}", axis=1)
        if 'cash_pct' not in d.columns:
            d['cash_pct'] = d.apply(
                lambda r: "" if pd.isna(r['cash_after']) or pd.isna(tot.loc[r.name]) or tot.loc[r.name] <= 0
                          else f"{(r['cash_after']/tot.loc[r.name]):.2%}", axis=1)

    # 3) 格式化
    def _fmt_date_col(s):
        dt = pd.to_datetime(s, errors='coerce')
        return dt.dt.strftime('%Y-%m-%d').fillna("")

    for col in ['trade_date','signal_date']:
        if col in d.columns:
            d[col] = _fmt_date_col(d[col])

    if 'price' in d.columns:
        d['price'] = d['price'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")
    if 'weight_change' in d.columns:
        d['weight_change'] = d['weight_change'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    for col in ['w_before','w_after']:
        if col in d.columns:
            d[col] = d[col].apply(lambda x: f"{x:,.4f}" if pd.notnull(x) else "")
    for col in ['exec_notional','equity_after','cash_after','position_value']:
        if col in d.columns:
            d[col] = d[col].apply(lambda x: f"{int(round(x)):,}" if pd.notnull(x) else "")
    if 'delta_units' in d.columns:
        d['delta_units'] = d['delta_units'].apply(lambda x: f"{int(round(x)):,}" if pd.notnull(x) else "")
    if 'return' in d.columns:
        d['return'] = d['return'].apply(lambda x: "-" if pd.isna(x) else f"{x:.2%}")

    # 4) 安全排序（只排存在的欄位）
    exist = [c for c in PREFER_ORDER if c in d.columns]
    others = [c for c in d.columns if c not in exist]
    d = d[exist + others]

    # 5) 中文欄名
    d = d.rename(columns={k: DISPLAY_NAME.get(k, k) for k in d.columns})
    return d

def is_price_data_up_to_date(csv_path):
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            last_date = pd.to_datetime(df['date'].iloc[-1])
        else:
            last_date = pd.to_datetime(df.iloc[-1, 0])
        
        # 獲取台北時間的今天
        today = pd.Timestamp.now(tz='Asia/Taipei').normalize()
        
        # 檢查是否為工作日（週一到週五）
        if today.weekday() >= 5:  # 週六(5)或週日(6)
            # 如果是週末，檢查最後數據是否為上個工作日
            last_weekday = today - pd.Timedelta(days=today.weekday() - 4)  # 上個週五
            return last_date >= last_weekday
        else:
            # 如果是工作日，檢查是否為今天或昨天（考慮數據延遲）
            yesterday = today - pd.Timedelta(days=1)
            return last_date >= yesterday
    except Exception:
        return False

def fetch_yf_data(ticker: str, filename: Path, start_date: str = "2000-01-01", end_date: str | None = None):
    now_taipei = pd.Timestamp.now(tz='Asia/Taipei')
    try:
        end_date_str = end_date if end_date is not None else now_taipei.strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start_date, end=end_date_str, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError("下載的數據為空")
        df.to_csv(filename)
        print(f"成功下載 '{ticker}' 數據到 '{filename}'.")
    except Exception as e:
        print(f"警告: '{ticker}' 下載失敗: {e}")

def ensure_all_price_data_up_to_date(ticker_list, data_dir):
    """智能檢查並更新股價數據，只在必要時下載"""
    for ticker in ticker_list:
        filename = Path(data_dir) / f"{ticker.replace(':','_')}_data_raw.csv"
        if not is_price_data_up_to_date(filename):
            print(f"{ticker} 股價資料需要更新，開始下載...")
            fetch_yf_data(ticker, filename)
        else:
            print(f"{ticker} 股價資料已是最新，跳過下載。")

# 簡化的股價數據下載剎車機制
def should_download_price_data():
    """檢查是否需要下載股價數據的剎車機制"""
    try:
        # 檢查是否為交易時間（避免在交易時間頻繁下載）
        now = pd.Timestamp.now(tz='Asia/Taipei')
        if now.weekday() < 5:  # 工作日
            hour = now.hour
            if 9 <= hour <= 13:  # 交易時間
                print("當前為交易時間，跳過股價數據下載以避免幹擾")
                return False
        
        # 檢查數據文件是否存在且較新（避免重複下載）
        data_files_exist = all(
            os.path.exists(Path(DATA_DIR) / f"{ticker.replace(':','_')}_data_raw.csv")
            for ticker in TICKER_LIST
        )
        
        if data_files_exist:
            print("股價數據文件已存在，跳過初始下載")
            return False
        
        return True
    except Exception as e:
        print(f"剎車機制檢查失敗: {e}，允許下載")
        return True

# 在 app 啟動時呼叫（添加剎車機制）
TICKER_LIST = ['2330.TW', '2412.TW', '2414.TW', '^TWII']  # 依實際需求調整
DATA_DIR = 'data'  # 依實際路徑調整

# 安全的啟動機制
def safe_startup():
    """安全的啟動函數，避免線程衝突"""
    try:
        # 只有在剎車機制允許時才下載
        if should_download_price_data():
            ensure_all_price_data_up_to_date(TICKER_LIST, DATA_DIR)
        else:
            print("股價數據下載已由剎車機制阻止")
    except Exception as e:
        print(f"啟動時數據下載失敗: {e}，繼續啟動應用")

