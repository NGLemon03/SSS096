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

# 設置 logger
logger = logging.getLogger(__name__)

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
    
    # 構建持倉矩陣
    pos_df = pd.DataFrame(index=index)
    
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
            pos_df[strat] = pos
            file_map[strat] = csv_path
    
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

def build_portfolio_ledger(open_px, w, cost: CostParams, initial_capital=1_000_000.0, lot_size=None):
    """
    依照每日 open 價與目標權重 w_t（含 floor、delta_cap 等限制後的最終 w_t），
    產出兩個 DataFrame：
      1) daily_state: 每日現金/持倉/總資產/權重
      2) trades: 只有權重變動日的交易明細（買賣金額、費用、稅、交易後資產）
    """
    import math
    import pandas as pd

    df = pd.DataFrame({"open": open_px, "w": w}).dropna().copy()
    df["w_prev"] = df["w"].shift(1).fillna(0.0)
    df["dw"] = df["w"] - df["w_prev"]

    cash = initial_capital
    units = 0.0
    rows_daily, rows_trades = [], []

    for dt, row in df.iterrows():
        px = float(row["open"])

        # --- before state
        pos_val_before = units * px
        equity_before  = cash + pos_val_before

        # --- target units
        target_notional = row["w"] * equity_before
        target_units = target_notional / px
        if lot_size:
            target_units = math.floor(target_units / lot_size) * lot_size

        delta_units = target_units - units
        if abs(delta_units) < 1e-12:  # 防極小波動
            delta_units = 0.0
        notional = abs(delta_units) * px

        # --- fees/taxes (bp)
        fee_buy  = (cost.buy_fee_bp  / 10000.0) * notional if delta_units > 0 else 0.0
        fee_sell = (cost.sell_fee_bp / 10000.0) * notional if delta_units < 0 else 0.0
        tax      = (cost.sell_tax_bp / 10000.0) * notional if delta_units < 0 else 0.0
        fees_total = fee_buy + fee_sell + tax

        # --- cash update
        if delta_units > 0:
            cash -= notional + fee_buy
            side = "BUY"
        elif delta_units < 0:
            cash += notional - fee_sell - tax
            side = "SELL"
        else:
            side = "HOLD"

        units = target_units

        # --- after state
        pos_val_after = units * px
        equity_after  = cash + pos_val_after
        actual_w = (pos_val_after / equity_after) if equity_after != 0 else 0.0
        trade_pct = (notional / equity_before) if equity_before != 0 else 0.0

        rows_daily.append({
            "date": dt, "open": px,
            "w_prev": row["w_prev"], "w": row["w"], "dw": row["dw"],
            "units": units,
            "cash_before": cash + (notional + fee_buy) - (notional - fee_sell - tax) if side != "HOLD" else cash,  # 可選
            "position_value_before": pos_val_before, "equity_before": equity_before,
            "cash": cash, "position_value": pos_val_after, "equity": equity_after,
            "cash_pct": (cash / equity_after) if equity_after != 0 else 0.0,
            "invested_pct": (pos_val_after / equity_after) if equity_after != 0 else 0.0,
            "actual_w": actual_w,
        })

        if delta_units != 0:
            rows_trades.append({
                "date": dt, "open": px, "side": side,
                "w_prev": row["w_prev"], "w": row["w"], "dw": row["dw"],
                "delta_units": delta_units, "exec_notional": notional,
                "fee_buy": fee_buy, "fee_sell": fee_sell, "tax": tax, "fees_total": fees_total,
                "trade_pct": trade_pct,
                "cash_after": cash, "position_value_after": pos_val_after, "equity_after": equity_after,
                "actual_w_after": actual_w,
            })

    daily_state = pd.DataFrame(rows_daily).set_index("date")
    trades = pd.DataFrame(rows_trades).set_index("date")
    return daily_state, trades

def equity_open_to_open(open_px: pd.Series, w: pd.Series, cost: CostParams | None = None,
                        start_equity: float = 1.0) -> Tuple[pd.Series, pd.DataFrame]:
    """計算 Open-to-Open 權益曲線和交易記錄"""
    if cost is None:
        cost = CostParams()
    
    # 計算日報酬率
    r = open_px.shift(-1) / open_px - 1
    r = r.dropna()
    
    # 權益曲線
    E = pd.Series(index=r.index, dtype=float)
    E.iloc[0] = start_equity
    
    # 交易記錄
    trades = []
    
    for i in range(1, len(r)):
        prev_w = w.iloc[i-1] if i-1 < len(w) else 0
        curr_w = w.iloc[i] if i < len(w) else 0
        
        # 權重變化
        dw = curr_w - prev_w
        
        if abs(dw) > 0.001:  # 有顯著變化
            if dw > 0:  # 買入
                c = dw * cost.buy_rate
                trades.append({
                    'trade_date': r.index[i],
                    'type': 'buy',
                    'price_open': open_px.iloc[i],
                    'weight_change': dw,
                    'cost': c
                })
            else:  # 賣出
                c = abs(dw) * cost.sell_rate
                trades.append({
                    'trade_date': r.index[i],
                    'type': 'sell',
                    'price_open': open_px.iloc[i],
                    'weight_change': abs(dw),
                    'cost': c
                })
            
            # 扣除交易成本
            E.iloc[i] = E.iloc[i-1] * (1 + r.iloc[i] * curr_w) - c
        else:
            E.iloc[i] = E.iloc[i-1] * (1 + r.iloc[i] * curr_w)
    
    # 補齊權益曲線（包括沒有交易的天數）
    E = E.reindex(open_px.index).fillna(method='ffill').fillna(start_equity)
    
    # 轉換交易記錄為 DataFrame
    if trades:
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = pd.DataFrame(columns=['trade_date', 'type', 'price_open', 'weight_change', 'cost'])
    
    return E, trades_df

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
    initial_capital: float = 1_000_000.0  # 初始資金
    lot_size: int | None = None  # 整股單位（100/1000，None=允許零股）
    
    def __post_init__(self):
        if self.params is None:
            self.params = EnsembleParams()
        if self.cost is None:
            self.cost = CostParams()

def run_ensemble(cfg: RunConfig) -> Tuple[pd.Series, pd.Series, pd.DataFrame, Dict[str, float], str, pd.Series, pd.DataFrame, pd.DataFrame]:
    """回傳：(open 價), (每日權重 w), (交易紀錄 trades), (績效指標 dict), (方法名稱), (權益曲線), (每日資產表), (交易流水帳)"""
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
    equity, trades = equity_open_to_open(px["open"], w, cfg.cost, start_equity=1.0)
    
    # 調試信息：Open→Open 報酬統計
    r_oo = (px['open'].shift(-1) / px['open'] - 1).dropna()
    logger.info(f"[Ensemble] r_oo mean={r_oo.mean():.4f}, std={r_oo.std():.4f}, count={len(r_oo)}")

    # 績效指標
    stats = perf_stats(equity, w)
    
    # 調試信息：績效摘要
    logger.info(f"[Ensemble] 績效摘要: 總報酬={stats.get('total_return', 0):.4f}, 年化={stats.get('annual_return', 0):.4f}, 最大回撤={stats.get('max_drawdown', 0):.4f}")
    
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
    
    logger.info(f"[Ensemble] 調試信息已保存到: {debug_path}")
    
    # 建立投資組合流水帳（ledger）
    daily_state, trade_ledger = build_portfolio_ledger(
        open_px=px["open"],
        w=w,                            # 最終權重序列
        cost=cfg.cost,                  # CostParams
        initial_capital=getattr(cfg, 'initial_capital', 1_000_000.0),  # 沒設定就給 1_000_000
        lot_size=getattr(cfg, 'lot_size', None) or None
    )
    
    # 存檔（與回測報告同一時間戳）
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    outdir = Path("results")
    outdir.mkdir(exist_ok=True, parents=True)
    name_tag = f"{method_name}_{ts}"
    daily_state.to_csv(outdir / f"ensemble_daily_state_{name_tag}.csv", encoding="utf-8-sig")
    trade_ledger.to_csv(outdir / f"ensemble_trade_ledger_{name_tag}.csv", encoding="utf-8-sig")
    
    # 印出最後一列摘要（final_w、cash_pct、invested_pct、last_trade exec_notional/fees_total）
    if not daily_state.empty and not trade_ledger.empty:
        latest_state = daily_state.iloc[-1]
        latest_trade = trade_ledger.iloc[-1]
        logger.info(f"[Ensemble] 投資組合流水帳摘要: final_w={latest_state['w']:.3f}, "
                   f"cash_pct={latest_state['cash_pct']:.1%}, invested_pct={latest_state['invested_pct']:.1%}, "
                   f"last_trade={latest_trade['side']} ${latest_trade['exec_notional']:,.0f} "
                   f"(fees_total=${latest_trade['fees_total']:.2f})")
    elif not daily_state.empty:
        latest_state = daily_state.iloc[-1]
        logger.info(f"[Ensemble] 投資組合流水帳摘要: final_w={latest_state['w']:.3f}, "
                   f"cash_pct={latest_state['cash_pct']:.1%}, invested_pct={latest_state['invested_pct']:.1%}, "
                   f"無交易")
    
    logger.info(f"[Ensemble] 投資組合流水帳已保存到: {outdir}")
    
    return px["open"], w, trades, stats, method_name, equity, daily_state, trade_ledger

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

    # 新增：保存 daily_state 和 trade_ledger
    if daily_state is not None:
        daily_state.to_csv(OUT_DIR / f"ensemble_ledger_daily_{method_name}.csv", index=True)
    
    if trade_ledger is not None:
        trade_ledger.to_csv(OUT_DIR / f"ensemble_ledger_trades_{method_name}.csv", index=True)

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
    ap.add_argument("--initial_capital", type=float, default=1_000_000.0, help="初始資金")
    ap.add_argument("--lot_size", type=int, default=None, help="整股單位（100/1000，None=允許零股）")
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
        cost=cost,
        initial_capital=args.initial_capital,
        lot_size=args.lot_size
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
    
    # 資金與交易參數
    st.subheader("資金與交易參數")
    col6, col7 = st.columns(2)
    
    with col6:
        initial_capital = st.number_input("初始資金", 100000.0, 10000000.0, 1000000.0, 100000.0, format="%.0f")
    with col7:
        lot_size_options = ["允許零股", "100股", "1000股"]
        lot_size_choice = st.selectbox("整股單位", lot_size_options, index=0)
        lot_size = None if lot_size_choice == "允許零股" else int(lot_size_choice.split("股")[0])
    
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
                majority_k_pct=majority_k_pct,
                initial_capital=initial_capital,
                lot_size=lot_size
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
            
            # 保存輸出
            save_outputs(method_name, open_px, w, trades, stats, equity, cost, daily_state, trade_ledger)
            st.info(f"結果已保存到 {OUT_DIR}")
            
            # 顯示投資組合流水帳摘要
            st.subheader("投資組合流水帳摘要")
            
            # 當前狀態卡片
            if not daily_state.empty:
                latest = daily_state.iloc[-1]
                col9, col10, col11, col12 = st.columns(4)
                with col9:
                    st.metric("當前權重", f"{latest['w']:.2%}")
                    st.metric("權重變化", f"{latest['dw']:.2%}")
                with col10:
                    st.metric("現金比例", f"{latest['cash_pct']:.2%}")
                    st.metric("投入比例", f"{latest['invested_pct']:.2%}")
                with col11:
                    st.metric("現金", f"${latest['cash']:,.0f}")
                    st.metric("持倉價值", f"${latest['position_value']:,.0f}")
                with col12:
                    st.metric("總資產", f"${latest['equity']:,.0f}")
                    st.metric("持倉單位", f"{latest['units']:,.0f}")
            
            # 今日交易摘要（如果有）
            if not trade_ledger.empty:
                latest_trade = trade_ledger.iloc[-1]
                st.subheader("最新交易")
                col13, col14, col15, col16 = st.columns(4)
                with col13:
                    st.metric("交易方向", latest_trade['side'])
                    st.metric("交易日期", str(latest_trade.name.date()))
                with col14:
                    st.metric("執行價格", f"${latest_trade['open']:.2f}")
                    st.metric("權重變化", f"{latest_trade['dw']:.2%}")
                with col15:
                    st.metric("交易金額", f"${latest_trade['exec_notional']:,.0f}")
                    st.metric("買進費用", f"${latest_trade['fee_buy']:.2f}")
                with col16:
                    st.metric("賣出費用", f"${latest_trade['fee_sell']:.2f}")
                    st.metric("證交稅", f"${latest_trade['tax']:.2f}")
            
            # 表格顯示
            st.subheader("每日資產表")
            st.dataframe(daily_state)
            
            st.subheader("交易明細表")
            st.dataframe(trade_ledger)
            
            # 匯出按鈕
            if st.button("下載交易流水帳 CSV"):
                csv = trade_ledger.to_csv(index=True, encoding='utf-8-sig')
                st.download_button(
                    label="下載 ensemble_trade_ledger.csv",
                    data=csv,
                    file_name=f"ensemble_trade_ledger_{method_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"執行失敗: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
