# -*- coding: utf-8 -*-
"""
SSS_EnsembleTab.py

兩種可直接在 SSS 專案中使用的「多策略組合」方法：
1) Majority K-of-N（多數決，預設使用比例門檻 majority_k_pct=0.55）
2) Proportional（依多頭比例分配）

嚴格遵守 T+1：N 日收盤產生訊號 -> N+1 開盤用於交易；
權益以 Open-to-Open 報酬遞推。

使用方式（在專案根目錄執行）：
    python SSS_EnsembleTab.py --ticker 00631L.TW --method majority \
        --floor 0.2 --ema 3 --delta 0.3
    python SSS_EnsembleTab.py --ticker 00631L.TW --method proportional \
        --floor 0.2 --ema 3 --delta 0.3

輸出：
- sss_backtest_outputs/ensemble_*.csv：
  - ensemble_weights_<name>.csv  : 每日權重 w[t]
  - ensemble_equity_<name>.csv   : 權益曲線（Open→Open）
  - ensemble_trades_<name>.csv   : 依權重變化生成的交易事件（t 開盤生效）
- sss_backtest_outputs/ensemble_summary.csv（附加模式）：各組合方法摘要績效

注意：
- 本檔讀取 SSS 在 sss_backtest_outputs/ 下既有的 trades_*.csv 來重建各子策略的「次日開盤生效」部位序列。
- 優先使用 trades_from_results_*.csv（120檔策略），找不到才使用舊的 trades_*.csv（11檔策略）。
- 成本與滑點可在參數中設定（預設使用 param_presets 中的配置）；
  台股實盤：buy_fee_bp=4.27、sell_fee_bp=4.27、sell_tax_bp=30（單位為 bp=萬分之一）。
"""

from __future__ import annotations
import argparse
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 設置 logger - 使用按需初始化
from analysis.logging_config import get_logger, init_logging
import os

# 設定環境變數
os.environ["SSS_CREATE_LOGS"] = "1"

# 獲取日誌器（懶加載）
logger = get_logger("SSS.Ensemble")

def _initialize_ensemble_logging():
    """初始化 Ensemble 日誌系統"""
    init_logging(enable_file=True)
    logger.info("=== Ensemble Tab 啟動 - 統一日誌系統 ===")
    return logger

# ---------------------------------------------------------------------
# 安全取值工具函式
# ---------------------------------------------------------------------

def _safe_equity_at(equity, dt):
    """以 as-of 方式抓取 <= dt 的最近一筆權益值；若超界則夾在邊界。"""
    import pandas as pd
    if not isinstance(dt, pd.Timestamp):
        dt = pd.to_datetime(dt)
    idx = equity.index
    pos = idx.searchsorted(dt, side="right") - 1
    if pos < 0:
        pos = 0
    if pos >= len(equity):
        pos = len(equity) - 1
    return float(equity.iloc[pos])

# ---------------------------------------------------------------------
# 統一交易明細契約標準化函式
# ---------------------------------------------------------------------

# 使用 sss_core 統一版本的 normalize_trades_for_plots
from sss_core.normalize import normalize_trades_for_plots


# 使用 sss_core 統一版本的 normalize_trades_for_ui
from sss_core.normalize import normalize_trades_for_ui

# ---------------------------------------------------------------------
# 序列化工具函式
# ---------------------------------------------------------------------

# 使用 sss_core 統一版本的 pack_df 和 pack_series
from sss_core.schemas import pack_df, pack_series

# ---------------------------------------------------------------------
# 路徑設定：以目前檔案所在資料夾為工作根目錄（符合你的習慣）
# ---------------------------------------------------------------------
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
# === 第3步：統一路徑，確保app與SSS使用相同的trades來源 ===
OUT_DIR = BASE_DIR / "sss_backtest_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 資料讀取：自動解析你上傳的 Yahoo-like CSV（6 欄或 7 欄）
# ---------------------------------------------------------------------

def _read_market_csv_auto(path: Path) -> pd.DataFrame:
    """將多種常見匯出格式自動對應到 [open, high, low, close, volume]。
    優先處理標準 yfinance CSV 格式，避免價格數據錯位。
    回傳 DataFrame（index=DatetimeIndex），缺值做合理清理，open/close 皆為正值。
    """
    # 先嘗試標準 Yahoo CSV（yfinance）
    try:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        need = {"open", "high", "low", "close", "volume"}
        if {"open", "high", "low", "close", "volume"}.issubset(set(cols)):
            # 標準格式
            df["date"] = pd.to_datetime(df.get(cols.get("date", df.columns[0])), format='%Y-%m-%d', errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date").sort_index()
            for k in ["open", "high", "low", "close", "volume"]:
                df[k] = pd.to_numeric(df[cols[k]], errors="coerce")
            df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "close"])
            return df
    except Exception:
        pass

    # 否則退回舊有 6/7 欄推測（原邏輯）
    raw = pd.read_csv(path, skiprows=2, header=None)
    if raw.shape[1] == 7:
        raw.columns = ["date", "price", "close", "high", "low", "open", "volume"]
        raw["date"] = pd.to_datetime(raw["date"], format='%Y-%m-%d', errors="coerce")
        df = raw.dropna(subset=["date"]).set_index("date").sort_index()
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    elif raw.shape[1] == 6:
        # 只剩 date + 五欄數值的變種
        raw.columns = ["date", "c1", "c2", "c3", "c4", "c5"]
        raw["date"] = pd.to_datetime(raw["date"], format='%Y-%m-%d', errors="coerce")
        raw = raw.dropna(subset=["date"]).set_index("date").sort_index()
        for c in ["c1", "c2", "c3", "c4", "c5"]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        # volume 以大量級判別
        counts = {c: (raw[c] > 1e5).sum() for c in ["c1", "c2", "c3", "c4", "c5"]}
        vol_col = max(counts, key=counts.get)
        rem = [c for c in ["c1", "c2", "c3", "c4", "c5"] if c != vol_col]
        
        # 剩餘四欄按大小排序：open < low < high < close
        sorted_cols = sorted(rem, key=lambda c: raw[c].mean())
        if len(sorted_cols) >= 4:
            df = pd.DataFrame({
                "open": raw[sorted_cols[0]],
                "low": raw[sorted_cols[1]],
                "high": raw[sorted_cols[2]],
                "close": raw[sorted_cols[3]],
                "volume": raw[vol_col]
            })
            return df
    
    # 如果都失敗，回傳空 DataFrame
    logger.warning(f"無法解析 {path} 的格式")
    return pd.DataFrame()

def _parse_trade_csv_auto(path: Path) -> pd.DataFrame:
    """自動解析多種交易 CSV 格式，回傳標準化的交易記錄"""
    try:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        
        # 檢查是否有必要的欄位
        if "date" in cols and "type" in cols:
            date_col, action_col = "date", "type"
        elif "trade_date" in cols and "type" in cols:
            date_col, action_col = "trade_date", "type"
        elif "date" in cols and "action" in cols:
            date_col, action_col = "date", "action"
        else:
            logger.warning(f"{path.name}: 缺少必要的 date 和 type/action 欄位")
            return pd.DataFrame()
        
        # 標準化
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["type"] = df[action_col].str.lower()
        df = df.dropna(subset=["date"]).sort_values("date")
        
        return df[["date", "type"]]
        
    except Exception as e:
        logger.error(f"解析 {path} 失敗: {e}")
        return pd.DataFrame()

def build_position_from_trades(trade_csv: Path, index: pd.DatetimeIndex) -> pd.Series:
    """從交易記錄重建持倉序列（0/1），嚴格遵守 T+1"""
    try:
        t = _parse_trade_csv_auto(trade_csv)
        if t.empty:
            logger.warning(f"{trade_csv.name}: 無法解析，回傳全0持倉")
            return pd.Series(0.0, index=index)
        
        # 初始化持倉序列
        pos = pd.Series(0.0, index=index)
        
        # 逐筆交易更新持倉
        for _, row in t.iterrows():
            dt = row["date"]
            typ = row["type"]
            
            # 找到對應的日期索引
            if dt in pos.index:
                if typ in ("buy", "long"):
                    cur = 1.0
                elif typ in ("sell", "sell_forced", "force_sell"):
                    cur = 0.0
                # 從該日「開盤」開始生效
                pos.loc[pos.index >= dt] = cur
        
        # 記錄持倉統計
        pos_1_count = (pos == 1.0).sum()
        pos_0_count = (pos == 0.0).sum()
        logger.info(f"[Ensemble] {trade_csv.name}: 持倉1={pos_1_count}, 持倉0={pos_0_count}, 交易筆數={len(t)}")
        
        return pos
        
    except Exception as e:
        logger.error(f"處理 {trade_csv.name} 失敗: {e}")
        return pd.Series(0.0, index=index)

def load_positions_matrix(out_dir: Path, index: pd.DatetimeIndex, strategies: List[str], file_map: Dict[str, Path] = None) -> pd.DataFrame:
    """載入多個策略的持倉矩陣"""
    if file_map is None:
        file_map = {}
    
    # 使用 dict 先收集所有策略的持倉數據，避免 DataFrame 碎片化
    positions_dict = {}
    
    for strat in strategies:
        if strat in file_map:
            csv_path = file_map[strat]
        else:
            # 嘗試多種文件名格式
            possible_files = [
                out_dir / f"trades_from_results_{strat}.csv",
                out_dir / f"trades_{strat}.csv",
                out_dir / f"{strat}.csv"
            ]
            
            csv_path = None
            for f in possible_files:
                if f.exists():
                    csv_path = f
                    break
            
            if csv_path is None:
                logger.warning(f"找不到 {strat} 的交易文件")
                continue
        
        # 重建持倉序列
        pos = build_position_from_trades(csv_path, index)
        if not pos.empty:
            positions_dict[strat] = pos
            file_map[strat] = csv_path
    
    # 一次性構建 DataFrame，避免碎片化
    if positions_dict:
        pos_df = pd.DataFrame(positions_dict, index=index)
    else:
        pos_df = pd.DataFrame(index=index)
    
    # 策略健康檢查/過濾：移除全0或持倉過少的子策略
    logger.info(f"[Ensemble] 持倉矩陣加載完成: {pos_df.shape[0]} 天 × {pos_df.shape[1]} 策略")
    
    # 檢查每個策略的持倉分佈，並標記不健康的策略
    unhealthy_strategies = []
    for strat in strategies:
        if strat in pos_df.columns:
            pos_series = pos_df[strat]
            if pos_series.isna().all() or (pos_series == 0).all():
                unhealthy_strategies.append(strat)
                logger.warning(f"策略 {strat} 持倉異常，將被過濾")
    
    # 移除不健康的策略
    if unhealthy_strategies:
        pos_df = pos_df.drop(columns=unhealthy_strategies)
        logger.info(f"過濾後剩餘策略數: {pos_df.shape[1]}")
    
    return pos_df

# ---------------------------------------------------------------------
# Ensemble 策略核心邏輯
# ---------------------------------------------------------------------

@dataclass
class EnsembleParams:
    floor: float = 0.2         # 底倉
    ema_span: int = 3          # EMA 平滑天數
    delta_cap: float = 0.3     # 每日 |Δw| 上限（0~1）
    majority_k: int = 6        # 多數決門檻（K-of-N，僅在沒有 majority_k_pct 時使用）
    min_cooldown_days: int = 1 # 最小冷卻天數（避免頻繁調整，與param_presets一致）
    min_trade_dw: float = 0.01 # 最小權重變化閾值（忽略微小調整，與param_presets一致）

def _smooth_and_cap(w_raw: pd.Series, span: int, delta_cap: float, min_cooldown_days: int = 1, min_trade_dw: float = 0.01) -> pd.Series:
    """平滑權重並限制每日變化幅度"""
    if span <= 1:
        w_smooth = w_raw
    else:
        w_smooth = w_raw.ewm(span=span, adjust=False).mean()
    
    # 限制每日變化幅度
    w_capped = w_smooth.copy()
    for i in range(1, len(w_capped)):
        delta = w_capped.iloc[i] - w_capped.iloc[i-1]
        if abs(delta) > delta_cap:
            # 限制變化幅度
            if delta > 0:
                w_capped.iloc[i] = w_capped.iloc[i-1] + delta_cap
            else:
                w_capped.iloc[i] = w_capped.iloc[i-1] - delta_cap
    
    # 應用冷卻期和最小變化閾值
    w_final = w_capped.copy()
    last_change_idx = 0
    
    for i in range(1, len(w_final)):
        delta = abs(w_final.iloc[i] - w_final.iloc[i-1])
        
        # 檢查冷卻期
        if i - last_change_idx < min_cooldown_days:
            w_final.iloc[i] = w_final.iloc[i-1]
            continue
        
        # 檢查最小變化閾值
        if delta < min_trade_dw:
            w_final.iloc[i] = w_final.iloc[i-1]
        else:
            last_change_idx = i
    
    return w_final

def weights_majority(pos_df: pd.DataFrame, p: EnsembleParams) -> pd.Series:
    """多數決權重：當多頭策略數 >= K 時，權重 = 1，否則 = floor"""
    S = pos_df.sum(axis=1)  # 每日多頭策略數
    w_raw = (S >= p.majority_k).astype(float)
    w_raw = w_raw * (1 - p.floor) + p.floor  # 加上底倉
    return _smooth_and_cap(w_raw, p.ema_span, p.delta_cap, p.min_cooldown_days, p.min_trade_dw)

def weights_proportional(pos_df: pd.DataFrame, p: EnsembleParams) -> pd.Series:
    """比例權重：權重 = 多頭策略數 / 總策略數"""
    S = pos_df.sum(axis=1)  # 每日多頭策略數
    N = pos_df.shape[1]     # 總策略數
    w_raw = S / max(N, 1)   # 避免除零
    w_raw = w_raw * (1 - p.floor) + p.floor  # 加上底倉
    return _smooth_and_cap(w_raw, p.ema_span, p.delta_cap, p.min_cooldown_days, p.min_trade_dw)

@dataclass
class CostParams:
    buy_fee_bp: float = 4.27   # 買進費率（bp）
    sell_fee_bp: float = 4.27  # 賣出費率（bp）
    sell_tax_bp: float = 30.0  # 賣出證交税（bp）

    @property
    def buy_rate(self) -> float:
        return self.buy_fee_bp / 10000.0

    @property
    def sell_rate(self) -> float:
        return (self.sell_fee_bp + self.sell_tax_bp) / 10000.0

    @property
    def sell_tax_rate(self) -> float:
        return self.sell_tax_bp / 10000.0

def build_portfolio_ledger(open_px: pd.Series, w: pd.Series, cost: CostParams, 
                          initial_capital: float = 1_000_000.0, lot_size: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    依照每日 open 價與目標權重 w_t（含 floor、delta_cap 等限制後的最終 w_t），
    產出兩個 DataFrame：
      1) daily_state: 每日現金/持倉/總資產/權重
      2) trade_ledger: 只有權重變動日的交易明細（買賣金額、費用、稅、交易後資產）
    """
    if cost is None:
        cost = CostParams()
    
    # 計算日報酬率
    r = open_px.shift(-1) / open_px - 1
    r = r.dropna()
    
    # 權益曲線
    E = pd.Series(index=r.index, dtype=float)
    E.iloc[0] = initial_capital
    
    # 現金曲線
    cash = pd.Series(index=r.index, dtype=float)
    cash.iloc[0] = initial_capital
    
    # 持倉價值曲線
    position_value = pd.Series(index=r.index, dtype=float)
    position_value.iloc[0] = 0.0
    
    # 交易記錄
    trades = []
    
    # 交易流水帳（詳細記錄）
    trade_ledger = []
    
    for i in range(1, len(r)):
        prev_w = w.iloc[i-1] if i-1 < len(w) else 0
        curr_w = w.iloc[i] if i < len(w) else 0
        
        # 權重變化
        dw = curr_w - prev_w
        
        if abs(dw) > 0.001:  # 有顯著變化
            # 1) 以「總權益」計執行名目金額
            exec_notional = abs(dw) * E.iloc[i-1]
            
            if dw > 0:  # 買入
                c = exec_notional * cost.buy_rate
                tax = 0.0
                total_cost = c

                # 計算顯示用股數（僅供 UI/報表，不影響邏輯）
                px = float(open_px.iloc[i])
                shares_before = (prev_w * E.iloc[i-1]) / px if px > 0 else 0.0
                delta_units   =  exec_notional / px if px > 0 else 0.0
                shares_after  = shares_before + delta_units

                # 交易明細（給圖表/表格）—補上權重欄位（買入也有）
                trades.append({
                    'trade_date': r.index[i],
                    'type': 'buy',
                    'price_open': float(open_px.iloc[i]),
                    'price': float(open_px.iloc[i]),
                    'weight_change': float(dw),          # signed dw
                    'w_before': float(prev_w),
                    'w_after': float(curr_w),
                    'shares_before': float(shares_before),
                    'shares_after': float(shares_after),
                    'exec_notional': float(exec_notional),
                    'equity_after': float(E.iloc[i-1]),  # 交易發生在 i 開盤，交易後的總資產(名目)≈前一日 E 扣費用
                    'cash_after': np.nan                 # 真正的日終現金在 daily_state 內，這裡僅供 UI 百分比推導（可選）
                })

                # 交易流水帳（詳細）
                trade_ledger.append({
                    'date': r.index[i],
                    'type': 'buy',
                    'open': open_px.iloc[i],
                    'delta_units': delta_units,
                    'exec_notional': exec_notional,
                    'fee_buy': c,
                    'fee_sell': 0.0,
                    'tax': tax,  # 賣出才會有證交稅，這裡為 0
                    'shares_before': shares_before,
                    'shares_after': shares_after,
                    'w_before': prev_w,
                    'w_after': curr_w,
                    'cash_after': cash.iloc[i-1] - exec_notional - c,
                    'equity_open_after_trade': E.iloc[i-1] - total_cost,
                    'equity_after': E.iloc[i-1] * (1 + r.iloc[i] * curr_w) - total_cost
                })
            else:  # 賣出
                c = exec_notional * cost.sell_rate
                tax = exec_notional * cost.sell_tax_rate
                total_cost = c + tax

                # 計算顯示用股數（僅供 UI/報表，不影響邏輯）
                px = float(open_px.iloc[i])
                shares_before = (prev_w * E.iloc[i-1]) / px if px > 0 else 0.0
                delta_units   = -exec_notional / px if px > 0 else 0.0
                shares_after  = shares_before + delta_units

                # 交易明細（給圖表/表格）—補上權重欄位
                trades.append({
                    'trade_date': r.index[i],
                    'type': 'sell',
                    'price_open': float(open_px.iloc[i]),
                    'price': float(open_px.iloc[i]),
                    'weight_change': float(dw),          # 負數
                    'w_before': float(prev_w),
                    'w_after': float(curr_w),
                    'shares_before': float(shares_before),
                    'shares_after': float(shares_after),
                    'exec_notional': float(exec_notional),
                    'equity_after': float(E.iloc[i-1]),
                    'cash_after': np.nan
                })

                # 交易流水帳（詳細）
                trade_ledger.append({
                    'date': r.index[i],
                    'type': 'sell',
                    'open': open_px.iloc[i],
                    'delta_units': delta_units,
                    'exec_notional': exec_notional,
                    'fee_buy': 0.0,
                    'fee_sell': c,
                    'tax': tax,
                    'shares_before': shares_before,
                    'shares_after': shares_after,
                    'w_before': prev_w,
                    'w_after': curr_w,
                    'cash_after': cash.iloc[i-1] + exec_notional - total_cost,
                    'equity_open_after_trade': E.iloc[i-1] - total_cost,
                    'equity_after': E.iloc[i-1] * (1 + r.iloc[i] * curr_w) - total_cost
                })
            
            # 扣除交易成本
            E.iloc[i] = E.iloc[i-1] * (1 + r.iloc[i] * curr_w) - total_cost
            cash.iloc[i] = cash.iloc[i-1] - (dw * E.iloc[i-1] + total_cost) if dw > 0 else cash.iloc[i-1] + (abs(dw) * E.iloc[i-1] - total_cost)
        else:
            E.iloc[i] = E.iloc[i-1] * (1 + r.iloc[i] * curr_w)
            cash.iloc[i] = cash.iloc[i-1]
        
        # 計算持倉價值
        position_value.iloc[i] = E.iloc[i] - cash.iloc[i]
    
    # 補齊權益曲線和現金曲線（包括沒有交易的天數）
    E = E.reindex(open_px.index).ffill().fillna(initial_capital)
    cash = cash.reindex(open_px.index).ffill().fillna(initial_capital)
    position_value = position_value.reindex(open_px.index).ffill().fillna(0.0)
    
    # 轉換交易記錄為 DataFrame
    if trades:
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = pd.DataFrame(columns=['trade_date', 'type', 'price_open', 'weight_change', 'cost'])
    
    # 轉換交易流水帳為 DataFrame
    if trade_ledger:
        trade_ledger_df = pd.DataFrame(trade_ledger)
    else:
        trade_ledger_df = pd.DataFrame(columns=['date', 'type', 'open', 'delta_units', 'exec_notional', 'fee_buy', 'fee_sell', 'tax', 'cash_after', 'equity_after'])
    
    # 構建每日狀態 DataFrame
    daily_state = pd.DataFrame({
        'equity': E,
        'cash': cash,
        'position_value': position_value,
        'w': w.reindex(open_px.index).fillna(0),
        'invested_pct': position_value / E,
        'cash_pct': cash / E
    })
    
    return daily_state, trade_ledger_df


def equity_open_to_open(open_px: pd.Series, w: pd.Series, cost: CostParams | None = None,
                        start_equity: float = 1.0) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """計算 Open-to-Open 權益曲線、交易記錄、每日狀態和交易流水帳"""
    if cost is None:
        cost = CostParams()
    
    # 計算日報酬率
    r = open_px.shift(-1) / open_px - 1
    r = r.dropna()
    
    # 權益曲線
    E = pd.Series(index=r.index, dtype=float)
    E.iloc[0] = start_equity
    
    # 現金曲線
    cash = pd.Series(index=r.index, dtype=float)
    cash.iloc[0] = start_equity
    
    # 交易記錄
    trades = []
    
    # 交易流水帳（詳細記錄）
    trade_ledger = []
    
    for i in range(1, len(r)):
        prev_w = w.iloc[i-1] if i-1 < len(w) else 0
        curr_w = w.iloc[i] if i < len(w) else 0
        
        # 權重變化
        dw = curr_w - prev_w
        
        if abs(dw) > 0.001:  # 有顯著變化
            # 1) 以「總權益」計執行名目金額
            exec_notional = abs(dw) * E.iloc[i-1]
            
            if dw > 0:  # 買入
                c = exec_notional * cost.buy_rate
                tax = 0.0
                total_cost = c

                # 計算顯示用股數（僅供 UI/報表，不影響邏輯）
                px = float(open_px.iloc[i])
                shares_before = (prev_w * E.iloc[i-1]) / px if px > 0 else 0.0
                delta_units   =  exec_notional / px if px > 0 else 0.0
                shares_after  = shares_before + delta_units

                # 交易明細（給圖表/表格）—補上權重欄位（買入也有）
                trades.append({
                    'trade_date': r.index[i],
                    'type': 'buy',
                    'price_open': open_px.iloc[i],
                    'weight_change': dw,          # 用 signed dw，買為正、賣為負
                    'w_before': prev_w,
                    'w_after': curr_w,
                    'shares_before': shares_before,
                    'shares_after': shares_after,
                    'cost': total_cost
                })

                # 交易流水帳（詳細）
                trade_ledger.append({
                    'date': r.index[i],
                    'type': 'buy',
                    'open': open_px.iloc[i],
                    'delta_units': delta_units,
                    'exec_notional': exec_notional,
                    'fee_buy': c,
                    'fee_sell': 0.0,
                    'tax': tax,  # 賣出才會有證交稅，這裡為 0
                    'shares_before': shares_before,
                    'shares_after': shares_after,
                    'w_before': prev_w,
                    'w_after': curr_w,
                    'cash_after': cash.iloc[i-1] - exec_notional - c,
                    'equity_open_after_trade': E.iloc[i-1] - total_cost,
                    'equity_after': E.iloc[i-1] * (1 + r.iloc[i] * curr_w) - total_cost
                })
            else:  # 賣出
                c = exec_notional * cost.sell_rate
                tax = exec_notional * cost.sell_tax_rate
                total_cost = c + tax

                # 計算顯示用股數（僅供 UI/報表，不影響邏輯）
                px = float(open_px.iloc[i])
                shares_before = (prev_w * E.iloc[i-1]) / px if px > 0 else 0.0
                delta_units   = -exec_notional / px if px > 0 else 0.0
                shares_after  = shares_before + delta_units

                # 交易明細（給圖表/表格）—補上權重欄位
                trades.append({
                    'trade_date': r.index[i],
                    'type': 'sell',
                    'price_open': open_px.iloc[i],
                    'weight_change': dw,          # 注意：保留 signed（賣出為負）
                    'w_before': prev_w,
                    'w_after': curr_w,
                    'shares_before': shares_before,
                    'shares_after': shares_after,
                    'cost': total_cost
                })

                # 交易流水帳（詳細）
                trade_ledger.append({
                    'date': r.index[i],
                    'type': 'sell',
                    'open': open_px.iloc[i],
                    'delta_units': delta_units,
                    'exec_notional': exec_notional,
                    'fee_buy': 0.0,
                    'fee_sell': c,
                    'tax': tax,
                    'shares_before': shares_before,
                    'shares_after': shares_after,
                    'w_before': prev_w,
                    'w_after': curr_w,
                    'cash_after': cash.iloc[i-1] + exec_notional - total_cost,
                    'equity_open_after_trade': E.iloc[i-1] - total_cost,
                    'equity_after': E.iloc[i-1] * (1 + r.iloc[i] * curr_w) - total_cost
                })
            
            # 扣除交易成本
            E.iloc[i] = E.iloc[i-1] * (1 + r.iloc[i] * curr_w) - total_cost
            cash.iloc[i] = cash.iloc[i-1] - (dw * E.iloc[i-1] + total_cost) if dw > 0 else cash.iloc[i-1] + (abs(dw) * E.iloc[i-1] - total_cost)
        else:
            E.iloc[i] = E.iloc[i-1] * (1 + r.iloc[i] * curr_w)
            cash.iloc[i] = cash.iloc[i-1]
    
    # 補齊權益曲線和現金曲線（包括沒有交易的天數）
    E = E.reindex(open_px.index).ffill().fillna(start_equity)
    cash = cash.reindex(open_px.index).ffill().fillna(start_equity)
    
    # 轉換交易記錄為 DataFrame
    if trades:
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = pd.DataFrame(columns=['trade_date', 'type', 'price_open', 'weight_change', 'cost'])
    
    # 轉換交易流水帳為 DataFrame
    if trade_ledger:
        trade_ledger_df = pd.DataFrame(trade_ledger)
    else:
        trade_ledger_df = pd.DataFrame(columns=['date', 'type', 'open', 'delta_units', 'exec_notional', 'fee_buy', 'fee_sell', 'tax', 'cash_after', 'equity_after'])
    
    # 構建每日狀態 DataFrame
    daily_state = pd.DataFrame({
        'equity': E,
        'cash': cash,
        'w': w.reindex(open_px.index).fillna(0),
        'position_value': E - cash
    })
    
    return E, trades_df, daily_state, trade_ledger_df

def perf_stats(equity: pd.Series, w: pd.Series) -> Dict[str, float]:
    """計算績效指標"""
    if len(equity) < 2:
        return {}
    
    # 計算日報酬率
    r = equity.pct_change().dropna()
    
    # 基本指標
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    annual_return = total_return * (252 / len(equity))
    
    # 最大回撤
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # 風險調整指標
    sharpe_ratio = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    sortino_ratio = r.mean() / r[r < 0].std() * np.sqrt(252) if len(r[r < 0]) > 0 and r[r < 0].std() > 0 else 0
    
    # 其他指標
    time_in_market = (w > 0.5).mean()
    turnover_py = w.diff().abs().sum() / len(w) * 252
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
        'time_in_market': time_in_market,
        'turnover_py': turnover_py,
        'num_trades': len(w.diff()[w.diff() != 0])
    }

# ---------------------------------------------------------------------
# 主流程：讀價、讀子策略部位 -> 生成權重 -> 權益與交易 -> 輸出
# ---------------------------------------------------------------------

@dataclass
class RunConfig:
    ticker: str
    method: str                  # "majority" | "proportional"
    strategies: List[str] | None = None  # None => 自動從 trades_* 推斷
    params: EnsembleParams = None
    cost: CostParams = None      # 預設不加成本；要貼近實盤可調整
    file_map: Dict[str, Path] = None  # 策略名 -> 文件路徑的映射
    majority_k_pct: float = None  # 比例門檻（0.0~1.0），優先於固定 majority_k
    
    def __post_init__(self):
        if self.params is None:
            self.params = EnsembleParams()
        if self.cost is None:
            self.cost = CostParams()

def run_ensemble(cfg: RunConfig) -> Tuple[pd.Series, pd.Series, pd.DataFrame, Dict[str, float], str, pd.Series, pd.DataFrame, pd.DataFrame]:
    """回傳：(open 價), (每日權重 w), (交易紀錄 trades), (績效指標 dict), (方法名稱), (權益曲線), (每日狀態 daily_state), (交易流水帳 trade_ledger)"""
    # 讀價（Open）
    px_path = DATA_DIR / f"{cfg.ticker.replace(':','_')}_data_raw.csv"
    px = _read_market_csv_auto(px_path)
    
    # 調試信息：價格數據
    logger.info(f"[Ensemble] 價資料天數={len(px)}, 首末={px.index.min()}~{px.index.max()}")

    # 推斷策略列表
    if cfg.strategies is None:
        # 優先使用 trades_from_results_*.csv（120檔策略）
        strat_names = []
        file_map = {}  # 策略名 -> 文件路徑的映射
        
        # 先找 trades_from_results_*.csv
        trades_files = list(OUT_DIR.glob("trades_from_results_*.csv"))
        if trades_files:
            logger.info(f"[Ensemble] 找到 {len(trades_files)} 個 trades_from_results_*.csv 文件（120檔策略）")
            for f in sorted(trades_files):
                # 從文件名推斷策略名稱
                name = f.stem.replace("trades_from_results_", "")
                strat_names.append(name)
                file_map[name] = f
        else:
            # 找不到再使用舊的 trades_*.csv（11檔策略）
            trades_files = list(OUT_DIR.glob("trades_*.csv"))
            logger.info(f"[Ensemble] 找到 {len(trades_files)} 個 trades_*.csv 文件（11檔策略）")
            for f in sorted(trades_files):
                # 從文件名推斷策略名稱
                name = f.stem.replace("trades_", "")
                strat_names.append(name)
                file_map[name] = f
        
        if not strat_names:
            raise ValueError(f"在 {OUT_DIR} 中找不到任何交易文件")
        
        cfg.strategies = strat_names
        cfg.file_map = file_map
    
    # 載入持倉矩陣
    pos_df = load_positions_matrix(OUT_DIR, px.index, cfg.strategies, cfg.file_map)
    
    if pos_df.empty:
        raise ValueError("無法載入任何策略的持倉數據")
    
    N = pos_df.shape[1]  # 策略數量
    logger.info(f"[Ensemble] 載入 {N} 個策略的持倉數據")
    
    # 處理 majority_k 參數
    if cfg.method.lower() == "majority":
        if cfg.majority_k_pct is not None:
            # 使用比例門檻
            k_req = int(math.ceil(N * cfg.majority_k_pct))
            logger.info(f"[Ensemble] 使用比例門檻 majority_k_pct={cfg.majority_k_pct}, N={N}, 計算得到 K={k_req}")
        else:
            # 使用固定 K 值
            k_req = cfg.params.majority_k
            logger.info(f"[Ensemble] 使用固定門檻 majority_k={k_req}, N={N}")
        
        # 方案 A：夾擠到合法範圍
        k_eff = max(1, min(int(k_req), N))
        
        # 方案 B：若 k>n 採用動態多數決（建議）：ceil(N*0.5)
        if k_req > N:
            logger.warning(f"majority_k({k_req}) > N({N}); fallback to ceil(N/2)={math.ceil(N*0.5)}")
            k_eff = int(math.ceil(N*0.5))
        
        # 用 dataclasses.replace 覆寫參數後再算權重與命名
        import dataclasses
        cfg.params = dataclasses.replace(cfg.params, majority_k=k_eff)
        
        # 檢查 N 是否足夠（可選但建議）
        if N < 8:
            logger.warning(f"策略數量 N={N} < 8，可能影響 ensemble 效果。建議確保有足夠的子策略。")

    # 調試信息：子策略多倉統計
    S = pos_df.sum(axis=1)
    logger.info(f"[Ensemble] 多頭計數S分佈: mean={S.mean():.2f}, 1%={S.quantile(0.01):.2f}, 99%={S.quantile(0.99):.2f}")

    # 權重
    if cfg.method.lower() == "majority":
        w = weights_majority(pos_df, cfg.params)
        # 命名也用調整後的 k_eff 與 N
        method_name = f"Majority_{k_eff}_of_{N}"
    elif cfg.method.lower() == "proportional":
        w = weights_proportional(pos_df, cfg.params)
        method_name = f"Proportional_N{N}"
    else:
        raise ValueError("method 必須是 'majority' 或 'proportional'")
    
    # 調試信息：權重統計
    w_raw = (S >= cfg.params.majority_k).astype(float) if cfg.method.lower() == 'majority' else (S / max(N, 1))
    logger.info(f"[Ensemble] w_raw(min/mean/max)={w_raw.min():.2f}/{w_raw.mean():.2f}/{w_raw.max():.2f}")
    logger.info(f"[Ensemble] w_smooth(min/mean/max)={w.min():.2f}/{w.mean():.2f}/{w.max():.2f}")

    # 權益與事件（Open→Open）
    equity, trades, daily_state, trade_ledger = equity_open_to_open(px["open"], w, cfg.cost, start_equity=1.0)
    
    # 以開盤價模擬資產曲線
    open_px = px['open'].copy()  # 你現有的價格列
    daily_state, trade_ledger = build_portfolio_ledger(
        open_px=open_px,
        w=w,                    # 你要落地的最終權重序列（已經過 floor/ema/delta_cap/cooldown）
        cost=cfg.cost,          # 你的 CostParams
        initial_capital=1_000_000.0,
        lot_size=None            # 若要整股就給 1000 or 1，否則 None 允許小數
    )
    
    # 統一欄位格式（避免之後序列化麻煩）
    daily_state = daily_state.copy()
    daily_state.index = pd.to_datetime(daily_state.index)
    daily_state.index.name = 'date'
    for col in ['equity', 'cash', 'position_value', 'invested_pct', 'cash_pct', 'w']:
        if col not in daily_state.columns:
            daily_state[col] = np.nan
    
    # 調試信息：Open→Open 報酬統計
    r_oo = (px['open'].shift(-1) / px['open'] - 1).dropna()
    logger.info(f"[Ensemble] r_oo mean={r_oo.mean():.4f}, std={r_oo.std():.4f}, count={len(r_oo)}")

    # 績效指標
    stats = perf_stats(equity, w)
    
    # 調試信息：績效摘要
    logger.info(f"[Ensemble] 績效摘要: 總報酬={stats.get('total_return', 0):.4f}, 年化={stats.get('annual_return', 0):.4f}, 最大回撤={stats.get('max_drawdown', 0):.4f}")
    
    # 標準化交易明細為統一契約
    trades_ui = normalize_trades_for_ui(trades)
    trade_ledger_ui = normalize_trades_for_ui(trade_ledger)
    
    # 調試信息：標準化後的交易明細
    logger.info(f"[Ensemble] 標準化後 trades_ui 欄位: {list(trades_ui.columns)}")
    logger.info(f"[Ensemble] 標準化後 trade_ledger_ui 欄位: {list(trade_ledger_ui.columns)}")
    logger.info(f"[Ensemble] 交易筆數: trades={len(trades_ui)}, trade_ledger={len(trade_ledger_ui)}")
    
    # 保存調試信息到文件
    debug_path = OUT_DIR / f"ensemble_debug_{method_name}.txt"
    with open(debug_path, 'w', encoding='utf-8') as f:
        f.write(f"Ensemble 調試信息\n")
        f.write(f"================\n")
        f.write(f"方法: {method_name}\n")
        f.write(f"策略數量: {N}\n")
        f.write(f"參數: {cfg.params}\n")
        f.write(f"成本: {cfg.cost}\n")
        f.write(f"權重統計: min={w.min():.4f}, mean={w.mean():.4f}, max={w.max():.4f}\n")
        f.write(f"績效: {stats}\n")
        f.write(f"交易明細欄位: trades={list(trades_ui.columns)}, trade_ledger={list(trade_ledger_ui.columns)}\n")
        f.write(f"交易筆數: trades={len(trades_ui)}, trade_ledger={len(trade_ledger_ui)}\n")
    
    logger.info(f"[Ensemble] 調試信息已保存到: {debug_path}")
    
    # 回傳標準化後的交易明細，確保前端顯示一致性
    return px["open"], w, trades_ui, stats, method_name, equity, daily_state, trade_ledger_ui

def save_outputs(method_name: str, open_px: pd.Series, w: pd.Series, trades: pd.DataFrame, stats: Dict[str, float], equity: pd.Series = None, cost: CostParams = None, daily_state: pd.DataFrame = None, trade_ledger: pd.DataFrame = None):
    """保存 ensemble 輸出文件"""
    # 權重
    w_df = pd.DataFrame({'date': w.index, 'weight': w})
    w_df.to_csv(OUT_DIR / f"ensemble_weights_{method_name}.csv", index=False)
    
    # 權益曲線
    if equity is not None:
        equity_df = pd.DataFrame({'date': equity.index, 'equity': equity})
        equity_df.to_csv(OUT_DIR / f"ensemble_equity_{method_name}.csv", index=False)
    
    # 交易記錄
    if not trades.empty:
        trades.to_csv(OUT_DIR / f"ensemble_trades_{method_name}.csv", index=False)
    
    # 每日狀態
    if daily_state is not None and not daily_state.empty:
        daily_state.to_csv(OUT_DIR / f"ensemble_daily_state_{method_name}.csv", index=False)
    
    # 交易流水帳
    if trade_ledger is not None and not trade_ledger.empty:
        trade_ledger.to_csv(OUT_DIR / f"ensemble_trade_ledger_{method_name}.csv", index=False)

    # 附加寫入 summary（不存在則新建）
    summ_path = OUT_DIR / "ensemble_summary.csv"
    
    # 構建方法名稱，包含成本信息
    method_with_cost = method_name
    if cost is not None:
        cost_info = []
        if cost.buy_fee_bp > 0:
            cost_info.append(f"fee{cost.buy_fee_bp}")
        if cost.sell_tax_bp > 0:
            cost_info.append(f"tax{cost.sell_tax_bp}")
        if cost_info:
            method_with_cost = f"{method_name}+{'+'.join(cost_info)}"
    
    row = {
        "method": method_with_cost,
        "total_return": stats.get("total_return"),
        "annual_return": stats.get("annual_return"),
        "max_drawdown": stats.get("max_drawdown"),
        "calmar_ratio": stats.get("calmar_ratio"),
        "sharpe_ratio": stats.get("sharpe_ratio"),
        "sortino_ratio": stats.get("sortino_ratio"),
        "time_in_market": stats.get("time_in_market"),
        "turnover_py": stats.get("turnover_py"),
        "num_trades": stats.get("num_trades"),
    }
    if summ_path.exists():
        df = pd.read_csv(summ_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(summ_path, index=False)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SSS Ensemble Tab (Strict T+1, Open-to-Open)")
    ap.add_argument("--ticker", type=str, required=True)
    ap.add_argument("--method", type=str, choices=["majority", "proportional"], required=True)
    ap.add_argument("--k", type=int, default=6, help="Majority 門檻 K（僅 majority 有效）")
    ap.add_argument("--floor", type=float, default=0.2, help="底倉（0~1）")
    ap.add_argument("--ema", type=int, default=3, help="EMA 平滑 span")
    ap.add_argument("--delta", type=float, default=0.3, help="每日 |Δw| 上限（0~1）")
    ap.add_argument("--cooldown", type=int, default=1, help="最小冷卻天數（避免頻繁調整）")
    ap.add_argument("--min_trade_dw", type=float, default=0.01, help="最小權重變化閾值（忽略微小調整，0~1）")
    ap.add_argument("--buy_fee_bp", type=float, default=4.27)
    ap.add_argument("--sell_fee_bp", type=float, default=4.27)
    ap.add_argument("--sell_tax_bp", type=float, default=30.0)
    return ap.parse_args()


def main():
    args = _parse_args()
    params = EnsembleParams(
        floor=args.floor,
        ema_span=args.ema,
        delta_cap=args.delta,
        majority_k=args.k,
        min_cooldown_days=args.cooldown,
        min_trade_dw=args.min_trade_dw,
    )
    cost = CostParams(
        buy_fee_bp=args.buy_fee_bp,
        sell_fee_bp=args.sell_fee_bp,
        sell_tax_bp=args.sell_tax_bp,
    )
    
    cfg = RunConfig(
        ticker=args.ticker,
        method=args.method,
        params=params,
        cost=cost
    )
    
    # 強制使用比例門檻 majority_k_pct=0.55
    if args.method == "majority":
        cfg.majority_k_pct = 0.55
        logger.info(f"[Ensemble] 強制設定 majority_k_pct=0.55")
    
    # 運行 ensemble
    open_px, w, trades, stats, method_name, equity, daily_state, trade_ledger = run_ensemble(cfg)
    
    # 保存輸出
    save_outputs(method_name, open_px, w, trades, stats, equity, cost, daily_state, trade_ledger)
    
    print(f"Ensemble 策略執行完成: {method_name}")
    print(f"績效: {stats}")

def streamlit_ensemble_ui():
    """Streamlit UI 的入口函數"""
    import streamlit as st
    
    st.title("SSS Ensemble 策略")
    
    # 參數輸入
    method = st.selectbox("方法", ["majority", "proportional"])
    ticker = st.text_input("股票代號", "00631L.TW")
    
    col1, col2 = st.columns(2)
    
    with col1:
        floor = st.slider("底倉", 0.0, 0.5, 0.2, 0.05)
        ema_span = st.slider("EMA 平滑天數", 1, 30, 3, 1)
        delta_cap = st.slider("每日權重變化上限", 0.05, 0.50, 0.3, 0.01)
    
    with col2:
        if method == "majority":
            majority_k_pct = st.slider("多數決比例門檻", 0.1, 0.9, 0.55, 0.05)
        else:
            majority_k_pct = 0.55
        
        min_cooldown_days = st.slider("最小冷卻天數", 1, 10, 1, 1)
        min_trade_dw = st.slider("最小權重變化閾值", 0.00, 0.10, 0.01, 0.01)
    
    # 成本參數
    st.subheader("交易成本")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        buy_fee_bp = st.number_input("買進費率 (bp)", 0.0, 100.0, 4.27, 0.01)
    with col4:
        sell_fee_bp = st.number_input("賣出費率 (bp)", 0.0, 100.0, 4.27, 0.01)
    with col5:
        sell_tax_bp = st.number_input("賣出證交税 (bp)", 0.0, 100.0, 30.0, 0.1)
    
    if st.button("執行 Ensemble 策略"):
        try:
            # 創建配置
            params = EnsembleParams(
                floor=floor,
                ema_span=ema_span,
                delta_cap=delta_cap,
                majority_k=6,  # 這個值會被 majority_k_pct 覆蓋
                min_cooldown_days=min_cooldown_days,
                min_trade_dw=min_trade_dw,
            )
            
            cost = CostParams(
                buy_fee_bp=buy_fee_bp,
                sell_fee_bp=sell_fee_bp,
                sell_tax_bp=sell_tax_bp,
            )
            
            cfg = RunConfig(
                ticker=ticker,
                method=method,
                params=params,
                cost=cost,
                majority_k_pct=majority_k_pct
            )
            
            # 運行 ensemble
            with st.spinner("執行中..."):
                open_px, w, trades, stats, method_name, equity, daily_state, trade_ledger = run_ensemble(cfg)
            
            # 顯示結果
            st.success(f"執行完成: {method_name}")
            
            # 績效指標
            col6, col7, col8 = st.columns(3)
            with col6:
                st.metric("總報酬率", f"{stats.get('total_return', 0):.2%}")
                st.metric("年化報酬率", f"{stats.get('annual_return', 0):.2%}")
            with col7:
                st.metric("最大回撤", f"{stats.get('max_drawdown', 0):.2%}")
                st.metric("夏普比率", f"{stats.get('sharpe_ratio', 0):.2f}")
            with col8:
                st.metric("卡瑪比率", f"{stats.get('calmar_ratio', 0):.2f}")
                st.metric("交易次數", stats.get('num_trades', 0))
            
            # === 新增：交易明細顯示 ===
            st.subheader("交易明細 (trade_ledger)")
            if trade_ledger is not None and not trade_ledger.empty:
                # 使用 normalize_trades_for_ui 標準化交易明細
                trades_ui = normalize_trades_for_ui(trade_ledger)
                
                # === 新增：刪除費用欄位（僅UI層移除顯示） ===
                cols_to_hide = ['fee_buy','fee_sell','sell_tax','tax','shares_before','shares_after']
                trades_ui = trades_ui.drop(columns=[c for c in cols_to_hide if c in trades_ui.columns], errors='ignore')
                
                # === 新增：統一數字格式 ===
                # 價格格式化
                if 'price' in trades_ui.columns:
                    trades_ui['price'] = trades_ui['price'].apply(
                        lambda x: f"{x:,.2f}" if pd.notnull(x) else ""
                    )
                
                # 金額/數量/權重等格式化
                for col in ['exec_notional','weight_change','w_before','w_after','delta_units',
                            'cash_after','equity_after','position_value']:
                    if col in trades_ui.columns:
                        trades_ui[col] = trades_ui[col].apply(
                            lambda x: f"{x:,.4f}" if pd.notnull(x) else ""
                        )
                
                # 百分比格式化
                for col in ['invested_pct','cash_pct']:
                    if col in trades_ui.columns:
                        trades_ui[col] = trades_ui[col].apply(
                            lambda x: f"{x:.2%}" if pd.notnull(x) else ""
                        )
                
                # 將欄位名稱轉換為中文
                trades_ui_zh = trades_ui.copy()
                column_mapping = {
                    'trade_date': '交易日期',
                    'type': '交易類型',
                    'price': '價格',
                    'weight_change': '權重變化',
                    'delta_units': '股數變化',
                    'exec_notional': '執行金額',
                    'w_before': '交易前權重',
                    'w_after': '交易後權重',
                    'shares_before': '交易前股數',
                    'shares_after': '交易後股數',
                    'equity_after': '交易後權益',
                    'cash_after': '交易後現金',
                    'invested_pct': '投資比例',
                    'cash_pct': '現金比例',
                    'position_value': '部位價值',
                    'comment': '備註'
                }
                # 只重命名存在的欄位
                existing_columns = {k: v for k, v in column_mapping.items() if k in trades_ui_zh.columns}
                trades_ui_zh = trades_ui_zh.rename(columns=existing_columns)
                st.dataframe(trades_ui_zh)
            else:
                st.info("無交易明細資料")
            
            # === 新增：資金權重顯示 ===
            st.subheader("資金權重")
            if daily_state is not None and not daily_state.empty:
                # 選擇要顯示的欄位（只顯示存在的欄位）
                display_columns = ['equity', 'cash', 'w', 'invested_pct', 'cash_pct', 'position_value']
                available_columns = [col for col in display_columns if col in daily_state.columns]
                
                if available_columns:
                    daily_state_display = daily_state[available_columns].copy()
                    # 將欄位名稱轉換為中文
                    column_mapping_ds = {
                        'equity': '權益',
                        'cash': '現金',
                        'w': '權重',
                        'invested_pct': '投資比例',
                        'cash_pct': '現金比例',
                        'position_value': '部位價值'
                    }
                    # 只重命名存在的欄位
                    existing_columns_ds = {k: v for k, v in column_mapping_ds.items() if k in daily_state_display.columns}
                    daily_state_display = daily_state_display.rename(columns=existing_columns_ds)
                    st.dataframe(daily_state_display)
                else:
                    st.info("無可顯示的資金權重資料")
            else:
                st.info("無每日狀態資料")
            
            # === 新增：持有權重變化圖 ===
            st.subheader("持有權重變化")
            if daily_state is not None and not daily_state.empty:
                # 🔍 偵錯輸出：把真正要畫的資料直接吐出 csv
                try:
                    print(f"🔍 Streamlit 開始偵錯輸出：{method}")
                    print(f"   daily_state type: {type(daily_state)}")
                    if hasattr(daily_state, 'shape'):
                        print(f"   daily_state shape: {daily_state.shape}")
                    if hasattr(daily_state, 'columns'):
                        print(f"   daily_state columns: {list(daily_state.columns)}")
                    
                    from sss_core.plotting import dump_equity_cash, dump_timeseries
                    dump_equity_cash(f"streamlit_ensemble_{method}", daily_state)
                    
                    # 同時把權重與價格也吐出，避免 index 對不齊
                    if 'w' in daily_state.columns:
                        print(f"   weight column 'w' found, shape: {daily_state['w'].shape}")
                        dump_timeseries(f"streamlit_ensemble_{method}", weight=daily_state['w'])
                    else:
                        print(f"   ⚠️ 沒有 'w' 欄位在 daily_state 中")
                        print(f"   available columns: {list(daily_state.columns)}")
                except Exception as e:
                    print(f"❌ Streamlit 偵錯輸出失敗：{e}")
                    import traceback
                    traceback.print_exc()
                
                from sss_core.plotting import plot_weight_series
                fig_w = plot_weight_series(daily_state, trade_ledger)
                st.plotly_chart(fig_w, use_container_width=True)
            else:
                st.info("無每日狀態資料，無法顯示權重變化圖")
            
            # 保存輸出
            save_outputs(method_name, open_px, w, trades, stats, equity, cost, daily_state, trade_ledger)
            st.info(f"結果已保存到 {OUT_DIR}")
            
        except Exception as e:
            st.error(f"執行失敗: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()

# === PATCH: Risk valve signals（兼容只有收盤價的情況）===
import numpy as np
import pandas as pd

def _get_col(df, *candidates):
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in cols:
            return cols[key]
    return None

def compute_risk_valve_signals(
    benchmark_df: pd.DataFrame,
    slope20_thresh: float = 0.0,
    slope60_thresh: float = 0.0,
    atr_win: int = 20,
    atr_ref_win: int = 60,
    atr_ratio_mult: float = 1.0,
    *,
    use_slopes: bool = True,
    slope_method: str = "polyfit",   # "polyfit"（增強法原版）或 "pct"
    atr_cmp: str = "gt"              # "gt"（>，增強法原版）或 "ge"（≥）
) -> pd.DataFrame:
    """
    回傳欄位：slope_20d, slope_60d, atr, atr_ratio, risk_trigger
    - 增強法原版：use_slopes=True, slope_method="polyfit", atr_cmp="gt"
    - ATR-only（你先前的新口徑）：use_slopes=False, atr_cmp="ge"
    """
    b = benchmark_df.copy()
    c_close = _get_col(b, "收盤價", "close")
    c_high  = _get_col(b, "最高價", "high")
    c_low   = _get_col(b, "最低價", "low")
    if c_close is None:
        raise ValueError("benchmark_df 需包含 收盤價/close")

    b = b.sort_index()
    close = pd.to_numeric(b[c_close], errors="coerce")
    prev_close = close.shift(1)

    # --- 斜率（兩種口徑擇一）---
    if use_slopes:
        if slope_method == "polyfit":
            def _poly(x, n):
                return np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == n else np.nan
            b["slope_20d"] = close.rolling(20).apply(lambda x: _poly(x, 20), raw=False)
            b["slope_60d"] = close.rolling(60).apply(lambda x: _poly(x, 60), raw=False)
        else:
            b["slope_20d"] = close.pct_change(20)
            b["slope_60d"] = close.pct_change(60)
    else:
        b["slope_20d"] = np.nan
        b["slope_60d"] = np.nan

    # --- ATR 與 ATR 比值 ---
    if c_high and c_low:
        high = pd.to_numeric(b[c_high], errors="coerce")
        low  = pd.to_numeric(b[c_low], errors="coerce")
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    else:
        tr = close.diff().abs()  # 無高低價時的近似

    b["atr"] = tr.rolling(atr_win, min_periods=atr_win).mean()
    b["atr_ref"] = b["atr"].rolling(atr_ref_win, min_periods=atr_ref_win).mean()
    b["atr_ratio"] = b["atr"] / b["atr_ref"]

    # --- 觸發邏輯：增強法原版 = 兩斜率皆 < 門檻 AND ATR 比值 > 倍數 ---
    atr_ok = (b["atr_ratio"] > atr_ratio_mult) if atr_cmp == "gt" else (b["atr_ratio"] >= atr_ratio_mult)
    if use_slopes:
        slp_ok = (b["slope_20d"] < slope20_thresh) & (b["slope_60d"] < slope60_thresh)
    else:
        slp_ok = True

    b["risk_trigger"] = slp_ok & atr_ok
    return b[["slope_20d", "slope_60d", "atr", "atr_ratio", "risk_trigger"]]


# === PATCH: 風險閥門施作在權重 ===
def apply_valve_to_weights(w: pd.Series,
                           risk_trigger: pd.Series,
                           mode: str = "cap",   # "cap" 或 "ban_add"
                           cap_level: float = 0.5) -> pd.Series:
    """對每日目標權重 w 施作風險閥門：
       - cap: 風險日將 w 限在 cap_level 以下
       - ban_add: 風險日禁止「提高」倉位（允許減倉），用迭代法確保單調"""
    w = w.copy().reindex(risk_trigger.index).ffill().fillna(0.0)
    out = w.copy()

    if mode == "cap":
        mask = risk_trigger.reindex(w.index).fillna(False).astype(bool)
        out[mask] = np.minimum(out[mask], cap_level)

    elif mode == "ban_add":
        mask = risk_trigger.reindex(w.index).fillna(False).astype(bool)
        for i in range(1, len(out)):
            if mask.iloc[i] and (out.iloc[i] > out.iloc[i-1]):
                out.iloc[i] = out.iloc[i-1]  # 禁止增加
    else:
        raise ValueError("mode 需為 'cap' 或 'ban_add'")
    return out.clip(0.0, 1.0)


# === PATCH: 由 trade_ledger 建立「每筆盈虧%」與快照（引用 re.txt 做法）===
def build_trade_snapshots_from_ledger(trade_ledger: pd.DataFrame) -> pd.DataFrame:
    """以交易後權益/現金為資產快照，計算每筆盈虧%（只在有變動的交易日記一筆）"""
    df = trade_ledger.copy()
    # 欄名對齊
    if "date" in df.columns: df.rename(columns={"date": "交易日期"}, inplace=True)
    if "type" in df.columns: df.rename(columns={"type": "交易類型"}, inplace=True)
    if "equity_after" in df.columns: df.rename(columns={"equity_after": "交易後權益"}, inplace=True)
    if "cash_after" in df.columns: df.rename(columns={"cash_after": "交易後現金"}, inplace=True)
    if "open" in df.columns: df.rename(columns={"open": "開盤價"}, inplace=True)
    if "w_before" in df.columns: df.rename(columns={"w_before": "交易前權重"}, inplace=True)
    if "w_after" in df.columns: df.rename(columns={"w_after": "交易後權重"}, inplace=True)

    df["交易日期"] = pd.to_datetime(df["交易日期"], errors="coerce")
    df = df.dropna(subset=["交易日期"]).sort_values("交易日期").reset_index(drop=True)

    df["總資產"] = pd.to_numeric(df.get("交易後權益", 0), errors="coerce").fillna(0) + \
                   pd.to_numeric(df.get("交易後現金", 0), errors="coerce").fillna(0)
    df["前次總資產"] = df["總資產"].shift(1)
    df["每筆盈虧"] = df["總資產"] - df["前次總資產"]
    df["每筆盈虧%"] = (df["每筆盈虧"] / df["前次總資產"]).replace([np.inf, -np.inf], np.nan) * 100
    df.loc[df.index.min(), ["每筆盈虧", "每筆盈虧%"]] = 0.0
    return df


# === PATCH: 只用賣出列計交易統計 & MDD 以日資產曲線算 ===
def _sell_returns_pct_from_ledger(trade_ledger: pd.DataFrame) -> pd.Series:
    tl = trade_ledger.copy()
    if "type" in tl.columns: tl.rename(columns={"type":"交易類型"}, inplace=True)
    if "equity_after" in tl.columns: tl.rename(columns={"equity_after":"交易後權益"}, inplace=True)
    if "equity_open_after_trade" in tl.columns: tl.rename(columns={"equity_open_after_trade":"交易後開盤權益"}, inplace=True)
    tl = tl.sort_values("date" if "date" in tl.columns else "交易日期")
    # 用「交易後開盤權益 → 交易日終權益」的變化當該筆交易的日損益（含成本）
    if "交易後開盤權益" in tl.columns and "交易後權益" in tl.columns:
        delta = (tl["交易後權益"] - tl["交易後開盤權益"]) / tl["交易後開盤權益"]
        tl["盈虧%"] = delta * 100.0
    elif "盈虧%" not in tl.columns:
        tl["盈虧%"] = 0.0
    mask_sell = tl["交易類型"].astype(str).str.lower().eq("sell")
    return pd.to_numeric(tl.loc[mask_sell, "盈虧%"], errors="coerce").dropna()

def _mdd_from_daily_equity(equity: pd.Series) -> float:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if eq.empty: return 0.0
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())

def risk_valve_backtest(open_px: pd.Series,
                        w: pd.Series,
                        cost,
                        benchmark_df: pd.DataFrame,
                        mode: str = "cap",
                        cap_level: float = 0.5,
                        slope20_thresh: float = 0.0,
                        slope60_thresh: float = 0.0,
                        atr_win: int = 20,
                        atr_ref_win: int = 60,
                        atr_ratio_mult: float = 1.0,
                        *,
                        use_slopes: bool = True,
                        slope_method: str = "polyfit",
                        atr_cmp: str = "gt") -> dict:
    """回傳：原始與閥門版本的績效、右尾削減度、以及兩版 equity/ledger"""
    sig = compute_risk_valve_signals(benchmark_df, slope20_thresh, slope60_thresh,
                                     atr_win, atr_ref_win, atr_ratio_mult,
                                     use_slopes=use_slopes, slope_method=slope_method, atr_cmp=atr_cmp)
    w2 = apply_valve_to_weights(w, sig["risk_trigger"], mode, cap_level)

    # 原始版本
    daily_state1, ledger1 = build_portfolio_ledger(open_px, w, cost)
    # 閥門版本
    daily_state2, ledger2 = build_portfolio_ledger(open_px, w2, cost)

    # 交易統計（只用賣出列；符合 re.txt 對實現損益的口徑）
    r1 = _sell_returns_pct_from_ledger(ledger1)
    r2 = _sell_returns_pct_from_ledger(ledger2)
    pf1 = (r1[r1>0].sum() / abs(r1[r1<0].sum())) if (r1[r1<0].sum()!=0) else np.inf
    pf2 = (r2[r2>0].sum() / abs(r2[r2<0].sum())) if (r2[r2<0].sum()!=0) else np.inf
    mdd1 = _mdd_from_daily_equity(daily_state1["equity"])
    mdd2 = _mdd_from_daily_equity(daily_state2["equity"])

    # 右尾削減：比較兩版的「>P90 的正報酬總和」
    def right_tail_sum(x: pd.Series):
        if x.empty: return 0.0
        thr = x.quantile(0.90)
        return float(x[x > max(thr, 0)].sum())
    rt1 = right_tail_sum(r1)
    rt2 = right_tail_sum(r2)
    right_tail_reduction = (rt1 - rt2)

    return {
        "signals": sig,
        "weights_orig": w,
        "weights_valve": w2,
        "daily_state_orig": daily_state1,
        "daily_state_valve": daily_state2,
        "trade_ledger_orig": ledger1,
        "trade_ledger_valve": ledger2,
        "metrics": {
            "pf_orig": float(pf1), "pf_valve": float(pf2),
            "mdd_orig": float(mdd1), "mdd_valve": float(mdd2),
            "right_tail_sum_orig": float(rt1), "right_tail_sum_valve": float(rt2),
            "right_tail_reduction": float(right_tail_reduction)
        }
    }


# === PATCH: 交易貢獻拆解（按加碼/減碼階段）===
def trade_contribution_by_phase(trade_ledger: pd.DataFrame,
                                daily_equity: pd.Series,
                                min_spacing_days: int = 0,
                                cooldown_days: int = 0) -> pd.DataFrame:
    """將交易依『權重變化正/負』分段，統計各階段的賣出實現報酬與該段內的 MDD。"""
    tl = trade_ledger.copy()
    # 欄對齊
    for a,b in [("date","交易日期"),("type","交易類型"),("w_before","交易前權重"),("w_after","交易後權重")]:
        if a in tl.columns: tl.rename(columns={a:b}, inplace=True)
    tl["交易日期"] = pd.to_datetime(tl["交易日期"], errors="coerce")
    tl = tl.dropna(subset=["交易日期"]).sort_values("交易日期").reset_index(drop=True)
    tl["dw"] = pd.to_numeric(tl["交易後權重"], errors="coerce") - pd.to_numeric(tl["交易前權重"], errors="coerce")

    # 可選：施加最小間距/冷卻（在分段前對過密買單降噪）
    if min_spacing_days > 0:
        last_buy_dt = None
        for i, row in tl.iterrows():
            if row["dw"] > 0:
                if last_buy_dt is not None and (row["交易日期"] - last_buy_dt).days < min_spacing_days:
                    tl.at[i, "dw"] = 0.0  # 忽略這筆微小加碼
                else:
                    last_buy_dt = row["交易日期"]
    if cooldown_days > 0:
        cooldown_until = None
        for i, row in tl.iterrows():
            if cooldown_until and row["交易日期"] <= cooldown_until and row["dw"] > 0:
                tl.at[i, "dw"] = 0.0
            if row["dw"] > 0:
                cooldown_until = row["交易日期"] + pd.Timedelta(days=cooldown_days)

    # 分段：連續 dw>0 視為 accumulation；連續 dw<0 視為 distribution
    phases = []
    if tl.empty: 
        return pd.DataFrame(columns=["階段","開始日期","結束日期","交易筆數","賣出報酬總和(%)","階段內MDD(%)"])

    cur_sign = 0
    cur_start = tl.loc[0, "交易日期"]
    for i, row in tl.iterrows():
        s = 1 if row["dw"] > 0 else (-1 if row["dw"] < 0 else cur_sign)
        if i == 0:
            cur_sign = s
            cur_start = row["交易日期"]
            continue
        if s != cur_sign:
            prev_dt = tl.loc[i-1, "交易日期"]
            phases.append((cur_sign, cur_start, prev_dt))
            cur_sign = s
            cur_start = row["交易日期"]
    phases.append((cur_sign, cur_start, tl.loc[tl.index.max(), "交易日期"]))

    # 計算每段的 賣出實現報酬總和 + 段內 MDD
    out_rows = []
    eq = daily_equity.copy()
    eq.index = pd.to_datetime(eq.index)
    sell_ret = _sell_returns_pct_from_ledger(trade_ledger)
    tl_idxed = tl.copy()  # 使用降噪後的資料
    tl_idxed["交易日期"] = pd.to_datetime(tl_idxed["date"] if "date" in tl_idxed.columns else tl_idxed["交易日期"])

    for sign, sdt, edt in phases:
        phase = "加碼階段" if sign>0 else ("減碼階段" if sign<0 else "持平階段")
        mask_t = (tl_idxed["交易日期"]>=sdt) & (tl_idxed["交易日期"]<=edt)
        sells_in_phase = _sell_returns_pct_from_ledger(tl_idxed.loc[mask_t])
        ret_sum = float(sells_in_phase.sum())
        # 日內 MDD
        eq_slice = eq.loc[(eq.index>=sdt) & (eq.index<=edt)]
        mdd = _mdd_from_daily_equity(eq_slice)
        out_rows.append({
            "階段": phase, 
            "開始日期": sdt.strftime("%m/%d"), 
            "結束日期": edt.strftime("%m/%d"),
            "交易筆數": int(mask_t.sum()),
            "賣出報酬總和(%)": round(ret_sum, 1),
            "階段內MDD(%)": round(mdd*100.0, 1)
        })
    
    # 創建結果 DataFrame
    result_df = pd.DataFrame(out_rows)
    
    # ---- ★ 新增：以原始日期算『階段淨貢獻(%)』『是否成功』 ----
    if {"開始日期","結束日期"}.issubset(result_df.columns):
        # 先保存原始日期（避免被格式化覆蓋）
        start_raw = []
        end_raw = []
        for _, row in result_df.iterrows():
            if row["開始日期"] and row["結束日期"]:
                # 從 MM/DD 格式反推完整日期（假設為當前年份）
                try:
                    start_dt = pd.to_datetime(f"2024/{row['開始日期']}", format="%Y/%m/%d")
                    end_dt = pd.to_datetime(f"2024/{row['結束日期']}", format="%Y/%m/%d")
                    start_raw.append(start_dt)
                    end_raw.append(end_dt)
                except:
                    start_raw.append(None)
                    end_raw.append(None)
            else:
                start_raw.append(None)
                end_raw.append(None)
        
        # 計算階段淨貢獻
        start_eq = []
        end_eq = []
        for start_dt, end_dt in zip(start_raw, end_raw):
            if start_dt is not None and end_dt is not None:
                start_eq.append(_safe_equity_at(eq, start_dt))
                end_eq.append(_safe_equity_at(eq, end_dt))
            else:
                start_eq.append(None)
                end_eq.append(None)
        
        # 計算淨貢獻百分比和是否成功
        net_contrib = []
        is_success = []
        for start_val, end_val in zip(start_eq, end_eq):
            if start_val is not None and end_val is not None and start_val != 0:
                net_pct = (end_val / start_val - 1.0) * 100.0
                net_contrib.append(round(net_pct, 2))
                is_success.append(net_pct > 0)
            else:
                net_contrib.append(None)
                is_success.append(None)
        
        result_df["階段淨貢獻(%)"] = net_contrib
        result_df["是否成功"] = is_success
    
    # 數字欄位 round（如已處理可略）
    for col in ["賣出報酬總和(%)","階段內MDD(%)"]:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce").round(2)
    
    # 欄序
    order = ["階段","開始日期","結束日期","交易筆數",
             "階段淨貢獻(%)","賣出報酬總和(%)","階段內MDD(%)","是否成功"]
    result_df = result_df[[c for c in order if c in result_df.columns]]
    
    # ---- ★ 最後才把日期顯示成 MM/DD ----
    for c in ("開始日期","結束日期"):
        if c in result_df.columns:
            # 重新格式化為 MM/DD
            for i, row in result_df.iterrows():
                if row[c] and isinstance(row[c], str) and "/" in row[c]:
                    # 已經是 MM/DD 格式，保持不變
                    pass
                elif row[c]:
                    try:
                        dt = pd.to_datetime(row[c])
                        result_df.at[i, c] = dt.strftime("%m/%d")
                    except:
                        pass
    
    # 添加統計摘要行
    if not result_df.empty:
        # 計算平均每段報酬 vs 平均每段 MDD
        avg_return = result_df["賣出報酬總和(%)"].mean()
        avg_mdd = result_df["階段內MDD(%)"].mean()
        
        # 添加摘要行
        summary_row = {
            "階段": f"📊 統計摘要 (間距:{min_spacing_days}天, 冷卻:{cooldown_days}天)",
            "開始日期": "",
            "結束日期": "",
            "交易筆數": len(result_df),
            "階段淨貢獻(%)": "",
            "賣出報酬總和(%)": f"平均: {avg_return:.1f}%",
            "階段內MDD(%)": f"平均: {avg_mdd:.1f}%",
            "是否成功": ""
        }
        
        # 將摘要行添加到結果中
        result_df = pd.concat([result_df, pd.DataFrame([summary_row])], ignore_index=True)
    
    return result_df
