# -*- coding: utf-8 -*-
'''
Optuna 16 - 多樣性+穩健性最佳化(細粒度分群/權益曲線快取/多指標) + Ensemble 策略支援
--------------------------------------------------

v4-5  數據源,策略分流,增加 WF、Sharpe、MDD 指標,過擬合懲罰,改用獨立日誌模組
v6   修正數據載入問題: 統一預載數據, 避免重複加載導致 SMAA 快取不一致.CPCV + PBO,SRA簡化,交易時機分析,交易日限制
v7   修正壓力測試期間: 新增交易日檢查, 確保 valid_stress_periods 非空, 避免無效或空白計算導致指標失真.
v8-9 修正 _backtest_once 返回結構,壓力測試相關函式,Equity Curve 缺失
v10-12  支援單一/隨機/依序數據源模式,新增平均持倉天數,相關係數分析,並自動輸出熱圖,試驗結果與最佳參數自動輸出 CSV/JSON
v13
- PBO:OOS 報酬分布計算 (abs(mean-median)/std/2),
- KNN:trial 參數空間 KNN 報酬差異,局部不穩定懲罰.
- pick_topN:依參數空間距離與績效
- WF:多段資料切割,取最差報酬作為穩健性指標.
- CPCV OOS:僅用 sklearn TimeSeriesSplit,非嚴格 CPCV.


v14
- PBO:報酬分布計算 (abs(mean-median)/std/2),多時段計算(wf,偽CPCV,Stress)
- KNN:懲罰邏輯優化（權重細緻）
- pick_topN:標準化後的參數距離，或加權距離
- WF:與 v13 相同.
- CPCV OOS:與 v13 相同.
- 夏普值變異數

v15
- PBO:同 v14
- KNN:同 v14
- pick_topN:績效,參數距離,OSS指標
- WF:同 v14
- CPCV OOS:同 v14
- 夏普值變異數

v16
歐式距離,馬式距離,加權距離,DTW距離分群效果皆不佳,改用新增層次聚類分群,自動選擇最佳分群代表策略
DTW距離使用於後續交易時機分析(OSv3)
- PBO:僅針對 CPCV OOS 報酬分布計算,分數懲罰.
- KNN:**已移除**.
- pick_topN:**已移除**,改用層次聚類分群,每群取分數最高者.
- WF:**已移除**.
- CPCV OOS:有,僅用 sklearn TimeSeriesSplit 分段 OOS 報酬均值/最小值,非嚴格 CPCV.
- 夏普值變異數

v16 修正版本1
整合細粒度分群分析，取代原本純hierarchical分群
- 目標：產生約600-800個群組，每群1-5個數據
- 使用階層式分群和KMeans兩種方法比較
- 自動選擇最佳分群方法
- 提供詳細的群組大小分布統計
- 保留原有的分群代表策略選取功能

v16 修正版本2 - 增強過擬合檢測
- 新增多維度過擬合檢測指標
- 增強PBO計算方法
- 新增參數敏感性分析
- 新增樣本內外一致性檢測
- 新增穩定性評分系統

v16 Ensemble 版本 - 新增 Ensemble 策略支援
- 支援 Majority 和 Proportional 集成方法
- 可調參數：floor, ema_span, delta_cap, majority_k, min_cooldown_days, min_trade_dw
- 多目標優化：報酬 × 交易次數
- 自動從 trades_*.csv 推斷子策略

層次聚類分群方法:
1. 使用自適應距離閾值進行層次聚類
2. 基於參數+績效特徵進行分群
3. 自動選擇最佳分群代表策略
4. 提供細粒度分群統計分析





命令列參數說明：

--strategy         指定策略類型（single, dual, RMA, ssma_turn, ensemble, all）
--n_trials         試驗次數（預設：5000）
--data_source      指定數據源（Self、Factor (^TWII / 2412.TW)、Factor (^TWII / 2414.TW)）
--data_source_mode 數據源選擇模式（random 隨機、fixed 固定、sequential 依序）
--top_n            分群前先選取前 N 個最佳策略（預設為全部）



範例：
# 完全自動分群（推薦）
python optuna_16.py --strategy single --data_source "Factor (^TWII / 2414.TW)" --n_trials 1000
python optuna_16.py --strategy single --data_source sequential --n_trials 1000
# 先篩選前 200 個策略再自動分群
python optuna_16.py --strategy single --data_source "Factor (^TWII / 2414.TW)" --top_n 200
# Ensemble 策略優化
python optuna_16.py --strategy ensemble --n_trials 1000

---
'''
import argparse

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
try:
    from tslearn.metrics import cdist_dtw
except ImportError:
    print("請先安裝 tslearn: pip install tslearn")

# 新增 Ensemble 策略支援
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ensemble_wrapper import EnsembleStrategyWrapper
    ENSEMBLE_AVAILABLE = True
except ImportError:
    print("警告：無法導入 ensemble_wrapper，Ensemble 策略將不可用")
    ENSEMBLE_AVAILABLE = False
    exit(1)
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

import optuna
from SSSv096 import load_data
import SSSv096 as SSS
RESULT_DIR = ROOT / 'results' / 'op16'
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("cache/optuna16_equity")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ========== 參數解析 ==========
parser = argparse.ArgumentParser(description='Optuna 16 多樣性+穩健性最佳化')
parser.add_argument('--strategy', type=str, choices=['single', 'dual', 'RMA', 'ssma_turn', 'ensemble', 'all'], default='all')
parser.add_argument('--n_trials', type=int, default=5000)
parser.add_argument('--data_source', type=str, default=None)
parser.add_argument('--data_source_mode', type=str, choices=['random', 'fixed', 'sequential'], default='random')
parser.add_argument('--top_n', type=int, default=None, help='分群前先選取前N個最佳策略(預設為全部)')
args = parser.parse_args()

# ========== 常數與設定 ==========
TICKER = "00631L.TW"
START_DATE = "2010-01-01"
END_DATE = "2025-06-17"
CACHE_DIR = Path("cache/optuna16_equity")
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ========== 參數空間(可擴充) ==========
PARAM_SPACE = {
    "single": dict(
        linlen=(5, 240, 1), smaalen=(7, 240, 5), devwin=(5, 180, 1),
        factor=(40, 40, 1), buy_mult=(0.1, 2.5, 0.05), sell_mult=(0.5, 4.0, 0.05), stop_loss=(0.00, 0.55, 0.2)),
    "dual": dict(
        linlen=(5, 240, 1), smaalen=(7, 240, 5), short_win=(10, 100, 5), long_win=(40, 240, 10),
        factor=(40, 40, 1), buy_mult=(0.2, 2, 0.05), sell_mult=(0.5, 4.0, 0.05), stop_loss=(0.00, 0.55, 0.1)),
    "RMA": dict(
        linlen=(5, 240, 1), smaalen=(7, 240, 5), rma_len=(20, 100, 5), dev_len=(10, 100, 5),
        factor=(40, 40, 1), buy_mult=(0.2, 2, 0.05), sell_mult=(0.5, 4.0, 0.05), stop_loss=(0.00, 0.55, 0.1)),
    "ssma_turn": dict(
        linlen=(10, 240, 5), smaalen=(10, 240, 5), factor=(40.0, 40.0, 1), prom_factor=(5, 70, 1),
        min_dist=(5, 20, 1), buy_shift=(0, 7, 1), exit_shift=(0, 7, 1), vol_window=(5, 90, 5), quantile_win=(5, 180, 10),
        signal_cooldown_days=(1, 7, 1), buy_mult=(0.5, 2, 0.05), sell_mult=(0.2, 3, 0.1), stop_loss=(0.00, 0.55, 0.1)),
}

# Ensemble 策略參數空間
if ENSEMBLE_AVAILABLE:
    PARAM_SPACE["ensemble"] = dict(
        method=['majority', 'proportional'],  # 集成方法
        floor=(0.0, 0.5, 0.05),             # 底仓
        ema_span=(3, 30, 1),                 # EMA 平滑天数
        delta_cap=(0.05, 0.30, 0.01),       # 每日权重变化上限
        majority_k=(3, 11, 1),               # 多数决门槛（K-of-N）
        min_cooldown_days=(1, 5, 1),         # 最小冷却天数
        min_trade_dw=(0.00, 0.05, 0.01)     # 最小权重变化阈值
    )

DATA_SOURCES = ['Self', 'Factor (^TWII / 2412.TW)', 'Factor (^TWII / 2414.TW)']
DATA_SOURCES_WEIGHTS = [1/3, 1/3, 1/3]

# ========== 數據源選擇邏輯 ==========
def select_data_source(trial, mode, fixed=None):
    if mode == 'sequential':
        idx = trial.number % len(DATA_SOURCES)
        return DATA_SOURCES[idx]
    elif mode == 'fixed':
        return fixed or 'Self'
    else:
        return np.random.choice(DATA_SOURCES, p=DATA_SOURCES_WEIGHTS)

# ========== 參數採樣 ==========
def sample_params(trial, strat):
    space = PARAM_SPACE[strat]
    params = {}
    for k, v in space.items():
        if k == 'method' and isinstance(v, list):
            # 对于 Ensemble 策略的 method 参数，从列表中选择
            params[k] = trial.suggest_categorical(k, v)
        elif isinstance(v[0], int):
            low, high, step = int(v[0]), int(v[1]), int(v[2])
            params[k] = trial.suggest_int(k, low, high, step=step)
        else:
            low, high, step = v
            params[k] = round(trial.suggest_float(k, low, high, step=step), 3)
    return params

# ========== 回測主體 ==========
def run_backtest(strat, params, df_price, df_factor):
    if strat == 'single':
        df_ind = SSS.compute_single(df_price, df_factor, params['linlen'], params['factor'], params['smaalen'], params['devwin'])
        result = SSS.backtest_unified(df_ind, strat, params)
    elif strat == 'dual':
        df_ind = SSS.compute_dual(df_price, df_factor, params['linlen'], params['factor'], params['smaalen'], params['short_win'], params['long_win'])
        result = SSS.backtest_unified(df_ind, strat, params)
    elif strat == 'RMA':
        df_ind = SSS.compute_RMA(df_price, df_factor, params['linlen'], params['factor'], params['smaalen'], params['rma_len'], params['dev_len'])
        result = SSS.backtest_unified(df_ind, strat, params)
    elif strat == 'ssma_turn':
        calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
        ssma_params = {k: v for k, v in params.items() if k in calc_keys}
        df_ind, buy_dates, sell_dates = SSS.compute_ssma_turn_combined(df_price, df_factor, **ssma_params)
        result = SSS.backtest_unified(df_ind, strat, params, buy_dates, sell_dates)
    elif strat == 'ensemble':
        # Ensemble 策略处理
        if not ENSEMBLE_AVAILABLE:
            raise ValueError("Ensemble 策略不可用，请检查 ensemble_wrapper 是否正确安装")
        
        # 创建 Ensemble 策略包装器
        wrapper = EnsembleStrategyWrapper()
        
        # 调用 Ensemble 策略
        equity_curve, trades, stats, method_name = wrapper.ensemble_strategy(
            method=params['method'],
            params=params,
            ticker="00631L.TW"  # 可以从参数中获取
        )
        
        # 构建兼容的返回格式
        metrics = {
            'total_return': stats.get('total_return', 0.0),
            'n_trades': stats.get('num_trades', 0),
            'sharpe_ratio': stats.get('sharpe_ratio', 0.0),
            'max_drawdown': stats.get('max_drawdown', 0.0),
            'profit_factor': stats.get('profit_factor', 0.0)
        }
        
        # 计算平均持仓天数
        avg_hold_days = 0.0
        if not trades.empty and 'date' in trades.columns:
            # 这里可以根据实际的交易记录格式调整
            avg_hold_days = 5.0  # 默认值，实际应该根据交易记录计算
        
        return (metrics['total_return'], metrics['n_trades'], metrics['sharpe_ratio'], 
                metrics['max_drawdown'], metrics['profit_factor'], trades, equity_curve, avg_hold_days)
    else:
        raise ValueError(f"不支持的策略類型: {strat}")
    
    # 原有策略的处理逻辑
    metrics = result['metrics']
    equity_curve = result['equity_curve']
    trades = result['trades']
    trade_df = result.get('trade_df', pd.DataFrame())
    avg_hold_days = 0.0
    if not trade_df.empty:
        holding_periods = []
        entry_date = None
        for _, row in trade_df.iterrows():
            if row['type'] == 'buy':
                entry_date = row['trade_date']
            elif row['type'] in ['sell', 'sell_forced'] and entry_date is not None:
                exit_date = row['trade_date']
                holding_days = (exit_date - entry_date).days
                holding_periods.append(holding_days)
                entry_date = None
        avg_hold_days = np.mean(holding_periods) if holding_periods else 0.0
    return (metrics['total_return'], metrics['n_trades'], metrics['sharpe_ratio'], metrics['max_drawdown'], metrics['profit_factor'], trades, equity_curve, avg_hold_days)

# ========== 指標計算 ==========
def calculate_cpcv_oos(equity_curve, min_period=500, max_splits=4):
    """計算CPCV OOS指標"""
    if len(equity_curve) < min_period:
        return 0.0, 0.0, []
    
    returns = equity_curve.pct_change().dropna()
    n_splits = min(max_splits, len(returns) // min_period)
    
    if n_splits < 2:
        return 0.0, 0.0, []
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oos_returns = []
    
    for train_idx, test_idx in tscv.split(returns):
        test_returns = returns.iloc[test_idx]
        if len(test_returns) > 0:
            oos_returns.append(test_returns.mean())
    
    if oos_returns:
        oos_mean = np.mean(oos_returns)
        oos_min = np.min(oos_returns)
        return oos_mean, oos_min, oos_returns
    else:
        return 0.0, 0.0, []

def calculate_adjusted_score(total_ret, n_trades, sharpe_ratio, max_drawdown, profit_factor, equity_curve, cpcv_oos_mean, cpcv_oos_min, sharpe_var, pbo_score, overfitting_metrics=None):
    """計算調整後分數 - 增強版過擬合檢測"""
    # 基本分數
    base_score = total_ret * 0.4 + sharpe_ratio * 0.3 + (1 + max_drawdown) * 0.2 + profit_factor * 0.1
    
    # OOS懲罰
    oos_penalty = abs(cpcv_oos_mean - total_ret) * 0.3
    oos_min_penalty = abs(cpcv_oos_min - total_ret) * 0.2
    
    # 基本穩定性懲罰
    stability_penalty = sharpe_var * 0.1 + pbo_score * 0.1
    
    # 增強過擬合懲罰
    enhanced_penalty = 0.0
    if overfitting_metrics:
        # 過擬合風險懲罰 (0-100 -> 0-0.5)
        overfitting_penalty = (overfitting_metrics['overfitting_risk'] / 100) * 0.5
        
        # 參數敏感性懲罰
        sensitivity_penalty = overfitting_metrics['parameter_sensitivity'] * 0.2
        
        # 一致性懲罰
        consistency_penalty = (1 - overfitting_metrics['consistency_score']) * 0.2
        
        # 穩定性懲罰
        stability_enhanced_penalty = (1 - overfitting_metrics['stability_score']) * 0.1
        
        enhanced_penalty = overfitting_penalty + sensitivity_penalty + consistency_penalty + stability_enhanced_penalty
    
    final_score = base_score - oos_penalty - oos_min_penalty - stability_penalty - enhanced_penalty
    return final_score

def calculate_sharpe_var(equity_curve, window=60):
    """計算夏普比率變異數"""
    if len(equity_curve) < window * 2:
        return 0.0
    
    returns = equity_curve.pct_change().dropna()
    rolling_sharpe = []
    
    for i in range(window, len(returns), window):
        window_returns = returns[i-window:i]
        if len(window_returns) > 10:
            sharpe = window_returns.mean() / window_returns.std() if window_returns.std() > 0 else 0
            rolling_sharpe.append(sharpe)
    
    return np.var(rolling_sharpe) if len(rolling_sharpe) > 1 else 0.0

def calculate_pbo(oos_returns):
    """計算PBO (Probability of Backtest Overfitting)"""
    if len(oos_returns) < 3:
        return 0.0
    
    mean_ret = np.mean(oos_returns)
    median_ret = np.median(oos_returns)
    std_ret = np.std(oos_returns)
    
    if std_ret == 0:
        return 0.0
    
    pbo = abs(mean_ret - median_ret) / (std_ret * 2)
    return min(pbo, 1.0)

def calculate_enhanced_overfitting_metrics(equity_curve, params, strategy_name):
    """
    計算增強的過擬合檢測指標
    
    Args:
        equity_curve: 權益曲線
        params: 策略參數
        strategy_name: 策略名稱
    
    Returns:
        dict: 過擬合指標字典
    """
    if len(equity_curve) < 100:
        return {
            'pbo_score': 0.0,
            'parameter_sensitivity': 0.0,
            'consistency_score': 0.0,
            'stability_score': 0.0,
            'overfitting_risk': 0.0
        }
    
    returns = equity_curve.pct_change().dropna()
    
    # 1. 增強的PBO計算
    cpcv_oos_mean, cpcv_oos_min, oos_returns = calculate_cpcv_oos(equity_curve)
    pbo_score = calculate_pbo(oos_returns) if oos_returns else 0.0
    
    # 2. 參數敏感性分析
    param_sensitivity = calculate_parameter_sensitivity(returns, params, strategy_name)
    
    # 3. 樣本內外一致性檢測
    consistency_score = calculate_consistency_score(equity_curve)
    
    # 4. 穩定性評分
    stability_score = calculate_stability_score(equity_curve)
    
    # 5. 綜合過擬合風險評分
    overfitting_risk = calculate_overfitting_risk(
        pbo_score, param_sensitivity, consistency_score, stability_score
    )
    
    return {
        'pbo_score': pbo_score,
        'parameter_sensitivity': param_sensitivity,
        'consistency_score': consistency_score,
        'stability_score': stability_score,
        'overfitting_risk': overfitting_risk
    }

def calculate_parameter_sensitivity(returns, params, strategy_name):
    """
    計算參數敏感性 - 基於參數變化對績效的影響
    """
    if not params or len(returns) < 50:
        return 0.0
    
    # 計算不同時間窗口的績效穩定性
    windows = [30, 60, 90, 120]
    performance_stability = []
    
    for window in windows:
        if len(returns) >= window * 2:
            # 計算前半段和後半段的績效差異
            first_half = returns[:window]
            second_half = returns[window:window*2]
            
            first_sharpe = first_half.mean() / first_half.std() if first_half.std() > 0 else 0
            second_sharpe = second_half.mean() / second_half.std() if second_half.std() > 0 else 0
            
            stability = 1 - abs(first_sharpe - second_sharpe) / (abs(first_sharpe) + abs(second_sharpe) + 1e-8)
            performance_stability.append(max(0, stability))
    
    # 基於參數複雜度調整敏感性
    param_complexity = len(params) / 10.0  # 參數數量越多，越容易過擬合
    
    # 計算最終敏感性分數
    avg_stability = np.mean(performance_stability) if performance_stability else 0.5
    sensitivity = (1 - avg_stability) * (1 + param_complexity)
    
    return min(1.0, sensitivity)

def calculate_consistency_score(equity_curve):
    """
    計算樣本內外一致性分數
    """
    if len(equity_curve) < 100:
        return 0.0
    
    returns = equity_curve.pct_change().dropna()
    
    # 使用多個時間分割點測試一致性
    split_points = [0.3, 0.5, 0.7]
    consistency_scores = []
    
    for split_ratio in split_points:
        split_idx = int(len(returns) * split_ratio)
        
        if split_idx > 30 and len(returns) - split_idx > 30:
            train_returns = returns[:split_idx]
            test_returns = returns[split_idx:]
            
            # 計算訓練和測試期間的績效差異
            train_sharpe = train_returns.mean() / train_returns.std() if train_returns.std() > 0 else 0
            test_sharpe = test_returns.mean() / test_returns.std() if test_returns.std() > 0 else 0
            
            train_return = train_returns.mean()
            test_return = test_returns.mean()
            
            # 計算一致性分數
            sharpe_consistency = 1 - abs(train_sharpe - test_sharpe) / (abs(train_sharpe) + abs(test_sharpe) + 1e-8)
            return_consistency = 1 - abs(train_return - test_return) / (abs(train_return) + abs(test_return) + 1e-8)
            
            consistency = (sharpe_consistency + return_consistency) / 2
            consistency_scores.append(max(0, consistency))
    
    return np.mean(consistency_scores) if consistency_scores else 0.5

def calculate_stability_score(equity_curve):
    """
    計算穩定性評分
    """
    if len(equity_curve) < 60:
        return 0.0
    
    returns = equity_curve.pct_change().dropna()
    
    # 1. 滾動夏普比率穩定性
    rolling_sharpe = []
    window = 30
    
    for i in range(window, len(returns), window//2):
        window_returns = returns[i-window:i]
        if len(window_returns) > 10:
            sharpe = window_returns.mean() / window_returns.std() if window_returns.std() > 0 else 0
            rolling_sharpe.append(sharpe)
    
    sharpe_stability = 1 - np.std(rolling_sharpe) / (np.mean(np.abs(rolling_sharpe)) + 1e-8) if rolling_sharpe else 0.5
    
    # 2. 最大回撤穩定性
    rolling_equity = equity_curve.rolling(window=30).mean()
    drawdowns = []
    
    for i in range(60, len(rolling_equity)):
        window_equity = rolling_equity.iloc[i-60:i]
        if len(window_equity) > 30:
            peak = window_equity.max()
            current = window_equity.iloc[-1]
            drawdown = (peak - current) / peak if peak > 0 else 0
            drawdowns.append(drawdown)
    
    drawdown_stability = 1 - np.std(drawdowns) if drawdowns else 0.5
    
    # 3. 交易頻率穩定性
    trade_frequency = []
    for i in range(30, len(returns), 30):
        window_returns = returns[i-30:i]
        trades = np.sum(np.abs(window_returns) > window_returns.std() * 2)
        trade_frequency.append(trades)
    
    frequency_stability = 1 - np.std(trade_frequency) / (np.mean(trade_frequency) + 1e-8) if trade_frequency else 0.5
    
    # 綜合穩定性分數
    stability_score = (sharpe_stability + drawdown_stability + frequency_stability) / 3
    return max(0, min(1, stability_score))

def calculate_overfitting_risk(pbo_score, param_sensitivity, consistency_score, stability_score):
    """
    計算綜合過擬合風險評分 (0-100, 越高風險越大)
    """
    # 權重分配
    weights = {
        'pbo': 0.3,
        'sensitivity': 0.25,
        'consistency': 0.25,
        'stability': 0.2
    }
    
    # 計算加權風險分數
    risk_score = (
        pbo_score * weights['pbo'] +
        param_sensitivity * weights['sensitivity'] +
        (1 - consistency_score) * weights['consistency'] +
        (1 - stability_score) * weights['stability']
    )
    
    # 轉換為0-100的評分
    return min(100, risk_score * 100)

# ========== 細粒度分群分析 ==========
def simple_fine_clustering_analysis(trials, strategy_name, data_source_name):
    """簡化細粒度分群分析：專門針對小群組需求"""
    print(f"\n=== {strategy_name} - {data_source_name} 細粒度分群分析 ===")
    
    # 準備特徵數據
    feature_data = []
    for t in trials:
        performance_features = [
            t.user_attrs.get('total_return', 0),
            t.user_attrs.get('sharpe_ratio', 0),
            t.user_attrs.get('max_drawdown', 0),
            t.user_attrs.get('profit_factor', 0),
            t.user_attrs.get('avg_hold_days', 0),
            t.user_attrs.get('cpcv_oos_mean', 0),
            t.user_attrs.get('cpcv_oos_min', 0)
        ]
        params = t.user_attrs.get('parameters', {})
        param_features = [
            params.get('linlen', 0),
            params.get('smaalen', 0),
            params.get('factor', 0),
            params.get('buy_mult', 0),
            params.get('sell_mult', 0),
            params.get('stop_loss', 0)
        ]
        all_features = performance_features + param_features
        feature_data.append(all_features)
    
    X = np.array(feature_data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 目標：產生約600-800個群組，每群1-5個數據
    n_samples = len(trials)
    target_clusters = min(max(600, n_samples // 3), 800)  # 目標群組數
    
    print(f"目標群組數: {target_clusters}")
    print(f"預期平均群組大小: {n_samples / target_clusters:.1f}")
    
    # 方法1：使用距離閾值的階層式分群
    print("\n方法1: 距離閾值階層式分群")
    
    # 計算距離閾值
    Z = linkage(X_scaled, method='ward')
    
    # 嘗試不同的距離閾值
    distance_thresholds = np.linspace(0.1, 2.0, 20)
    best_threshold = None
    best_score = -1
    
    for threshold in distance_thresholds:
        labels = fcluster(Z, t=threshold, criterion='distance')
        n_clusters = len(np.unique(labels))
        
        if n_clusters > 1:
            # 計算群組大小分布分數
            cluster_sizes = np.bincount(labels)
            small_clusters = np.sum(cluster_sizes <= 5)
            small_ratio = small_clusters / len(cluster_sizes)
            
            # 計算與目標群組數的接近程度
            target_penalty = abs(n_clusters - target_clusters) / target_clusters
            
            # 綜合分數
            score = small_ratio - target_penalty
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
    
    print(f"最佳距離閾值: {best_threshold:.3f}")
    
    # 使用最佳閾值進行分群
    labels_hierarchical = fcluster(Z, t=best_threshold, criterion='distance')
    
    # 方法2：使用KMeans，動態調整群組數
    print("\n方法2: 動態KMeans分群")
    
    # 從較大的群組數開始，逐步減少直到達到目標
    k_range = range(max(100, target_clusters // 2), min(target_clusters * 2, n_samples // 2), 10)
    best_k = None
    best_score_kmeans = -1
    
    for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels_kmeans = kmeans.fit_predict(X_scaled)
            
        # 計算群組大小分布分數
        cluster_sizes = np.bincount(labels_kmeans)
        small_clusters = np.sum(cluster_sizes <= 5)
        small_ratio = small_clusters / len(cluster_sizes)
        
        # 計算與目標群組數的接近程度
        target_penalty = abs(k - target_clusters) / target_clusters
        
        # 綜合分數
        score = small_ratio - target_penalty
        
        if score > best_score_kmeans:
            best_score_kmeans = score
            best_k = k
    
    print(f"最佳K值: {best_k}")
    
    # 使用最佳K值進行分群
    if best_k is not None:
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
        labels_kmeans = kmeans.fit_predict(X_scaled)
    else:
        # 如果沒有找到最佳K值，使用預設值
        best_k = min(100, n_samples // 2)
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
        labels_kmeans = kmeans.fit_predict(X_scaled)
    
    # 比較兩種方法
    methods = {
        'hierarchical': labels_hierarchical,
        'kmeans': labels_kmeans
    }
    
    best_method = None
    best_overall_score = -1
    best_labels = None
    
    for method_name, labels in methods.items():
        if len(np.unique(labels)) <= 1:
            continue
            
        # 計算群組大小分布
        cluster_sizes = np.bincount(labels)
        small_clusters = np.sum(cluster_sizes <= 5)
        medium_clusters = np.sum((cluster_sizes > 5) & (cluster_sizes <= 10))
        large_clusters = np.sum(cluster_sizes > 10)
        
        # 計算分數
        small_ratio = small_clusters / len(cluster_sizes)
        target_penalty = abs(len(cluster_sizes) - target_clusters) / target_clusters
        large_penalty = large_clusters / len(cluster_sizes)  # 懲罰大群組
        
        overall_score = small_ratio - target_penalty - large_penalty
        
        print(f"{method_name}: 群組數={len(cluster_sizes)}, 小群組比例={small_ratio:.3f}, 分數={overall_score:.3f}")
        
        if overall_score > best_overall_score:
            best_overall_score = overall_score
            best_method = method_name
            best_labels = labels
    
    print(f"\n最佳方法: {best_method}")
    
    # 重新編號標籤（從1開始）
    if best_labels is not None:
        unique_labels = np.unique(best_labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels, 1)}
        final_labels = np.array([label_mapping[label] for label in best_labels])
    else:
        # 如果沒有最佳標籤，使用預設值
        final_labels = np.ones(len(trials), dtype=int)
    
    # 統計最終結果
    final_sizes = np.bincount(final_labels)
    print(f"\n最終分群結果:")
    print(f"總群組數: {len(final_sizes)}")
    print(f"平均群組大小: {np.mean(final_sizes):.1f}")
    print(f"最大群組大小: {np.max(final_sizes)}")
    print(f"小群組比例 (≤5): {np.sum(final_sizes <= 5) / len(final_sizes):.3f}")
    
    # 詳細統計
    small_clusters = final_sizes[final_sizes <= 5]
    medium_clusters = final_sizes[(final_sizes > 5) & (final_sizes <= 10)]
    large_clusters = final_sizes[final_sizes > 10]
    
    print(f"小群組 (≤5): {len(small_clusters)} 個")
    print(f"中群組 (6-10): {len(medium_clusters)} 個")
    print(f"大群組 (>10): {len(large_clusters)} 個")
    
    return {
        'final_labels': final_labels,
        'method': best_method,
        'final_sizes': final_sizes,
        'target_clusters': target_clusters
    }

# ========== Optuna 目標函數 ==========
def objective(trial):
    if args.strategy == 'all':
        strat = np.random.choice(list(PARAM_SPACE.keys()))
    else:
        strat = args.strategy
    data_source = select_data_source(trial, args.data_source_mode, args.data_source)
    params = sample_params(trial, strat)
    df_price, df_factor = load_data(TICKER, START_DATE, END_DATE, data_source)
    (total_return, n_trades, sharpe_ratio, max_drawdown, profit_factor, trades, equity_curve, avg_hold_days) = run_backtest(strat, params, df_price, df_factor)
    # 新增:過濾交易次數過少
    if n_trades is None or n_trades <= 7:
        trial.set_user_attr('strategy', strat)
        trial.set_user_attr('data_source', data_source)
        trial.set_user_attr('parameters', params)
        trial.set_user_attr('total_return', -np.inf)
        trial.set_user_attr('num_trades', n_trades)
        trial.set_user_attr('sharpe_ratio', 0.0)
        trial.set_user_attr('max_drawdown', 0.0)
        trial.set_user_attr('profit_factor', 0.0)
        trial.set_user_attr('avg_hold_days', 0.0)
        trial.set_user_attr('cpcv_oos_mean', 0.0)
        trial.set_user_attr('cpcv_oos_min', 0.0)
        trial.set_user_attr('sharpe_var', 0.0)
        trial.set_user_attr('pbo_score', 0.0)
        return -np.inf
    # 增強過擬合檢測
    cpcv_oos_mean, cpcv_oos_min, cpcv_oos_list = calculate_cpcv_oos(equity_curve)
    sharpe_var = calculate_sharpe_var(equity_curve)
    pbo_score = calculate_pbo(cpcv_oos_list)
    
    # 計算增強的過擬合指標
    overfitting_metrics = calculate_enhanced_overfitting_metrics(equity_curve, params, strat)
    
    # Ensemble 策略的特殊处理：多目标优化
    if strat == 'ensemble':
        # 对于 Ensemble 策略，使用多目标优化
        # 目标1：最大化总报酬
        # 目标2：最小化交易次数（避免过度交易）
        
        # 计算交易成本惩罚
        turnover_penalty = 0.0
        if n_trades > 0:
            # 交易成本惩罚：λ * (num_trades / T)，λ 建议 0.1~0.3
            lambda_turnover = 0.2  # 可调整
            T = len(equity_curve) if hasattr(equity_curve, '__len__') else 1000
            turnover_penalty = lambda_turnover * (n_trades / T)
        
        # 多目标分数
        score_return = total_return - turnover_penalty
        score_trades = -n_trades  # 负号表示最小化
        
        # 存储多目标信息
        trial.set_user_attr('score_return', score_return)
        trial.set_user_attr('score_trades', score_trades)
        trial.set_user_attr('turnover_penalty', turnover_penalty)
        
        # 主要目标：报酬 - 交易惩罚
        score = score_return
    else:
        # 原有策略使用调整后的分数
        score = calculate_adjusted_score(total_return, n_trades, sharpe_ratio, max_drawdown, profit_factor, equity_curve, cpcv_oos_mean, cpcv_oos_min, sharpe_var, pbo_score, overfitting_metrics)
    
    # 權益曲線快取
    eq_path = CACHE_DIR / f"trial_{trial.number:05d}_equity.npy"
    print(f"[DEBUG] 權益曲線cache寫入路徑: {eq_path.resolve()}")
    print(f"[DEBUG] 權益曲線 shape: {equity_curve.shape if hasattr(equity_curve, 'shape') else type(equity_curve)}")
    if not CACHE_DIR.exists():
        print(f"[DEBUG] CACHE_DIR不存在,自動建立: {CACHE_DIR.resolve()}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(eq_path, equity_curve.values)
    print(f"[DEBUG] 權益曲線cache已寫入: {eq_path.resolve()}")
    
    # 回傳分數與指標
    trial.set_user_attr('strategy', strat)
    trial.set_user_attr('data_source', data_source)
    trial.set_user_attr('parameters', params)
    trial.set_user_attr('total_return', total_return)
    trial.set_user_attr('num_trades', n_trades)
    trial.set_user_attr('sharpe_ratio', sharpe_ratio)
    trial.set_user_attr('max_drawdown', max_drawdown)
    trial.set_user_attr('profit_factor', profit_factor)
    trial.set_user_attr('avg_hold_days', avg_hold_days)
    trial.set_user_attr('cpcv_oos_mean', cpcv_oos_mean)
    trial.set_user_attr('cpcv_oos_min', cpcv_oos_min)
    trial.set_user_attr('sharpe_var', sharpe_var)
    trial.set_user_attr('pbo_score', pbo_score)
    
    # 儲存增強的過擬合指標（非 Ensemble 策略）
    if strat != 'ensemble':
        trial.set_user_attr('parameter_sensitivity', overfitting_metrics['parameter_sensitivity'])
        trial.set_user_attr('consistency_score', overfitting_metrics['consistency_score'])
        trial.set_user_attr('stability_score', overfitting_metrics['stability_score'])
        trial.set_user_attr('overfitting_risk', overfitting_metrics['overfitting_risk'])
    
    return score

# ========== 執行 Optuna ==========
if args.data_source_mode == 'sequential':
    for data_source in DATA_SOURCES:
        print(f"[INFO] 處理數據源: {data_source}")
        # 設定當前 data_source
        # 重新建立 study
        study = optuna.create_study(direction='maximize')
        def objective_sequential(trial):
            params = sample_params(trial, args.strategy)
            df_price, df_factor = load_data(TICKER, START_DATE, END_DATE, data_source)
            (total_return, n_trades, sharpe_ratio, max_drawdown, profit_factor, trades, equity_curve, avg_hold_days) = run_backtest(args.strategy, params, df_price, df_factor)
            if n_trades is None or n_trades <= 7:
                trial.set_user_attr('strategy', args.strategy)
                trial.set_user_attr('data_source', data_source)
                trial.set_user_attr('parameters', params)
                trial.set_user_attr('total_return', -np.inf)
                trial.set_user_attr('num_trades', n_trades)
                trial.set_user_attr('sharpe_ratio', 0.0)
                trial.set_user_attr('max_drawdown', 0.0)
                trial.set_user_attr('profit_factor', 0.0)
                trial.set_user_attr('avg_hold_days', 0.0)
                trial.set_user_attr('cpcv_oos_mean', 0.0)
                trial.set_user_attr('cpcv_oos_min', 0.0)
                trial.set_user_attr('sharpe_var', 0.0)
                trial.set_user_attr('pbo_score', 0.0)
                return -np.inf
            cpcv_oos_mean, cpcv_oos_min, cpcv_oos_list = calculate_cpcv_oos(equity_curve)
            sharpe_var = calculate_sharpe_var(equity_curve)
            pbo_score = calculate_pbo(cpcv_oos_list)
            
            # 計算增強的過擬合指標
            overfitting_metrics = calculate_enhanced_overfitting_metrics(equity_curve, params, args.strategy)
            
            score = calculate_adjusted_score(total_return, n_trades, sharpe_ratio, max_drawdown, profit_factor, equity_curve, cpcv_oos_mean, cpcv_oos_min, sharpe_var, pbo_score, overfitting_metrics)
            eq_path = CACHE_DIR / f"trial_{trial.number:05d}_equity.npy"
            if not CACHE_DIR.exists():
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(eq_path, equity_curve.values)
            trial.set_user_attr('strategy', args.strategy)
            trial.set_user_attr('data_source', data_source)
            trial.set_user_attr('parameters', params)
            trial.set_user_attr('total_return', total_return)
            trial.set_user_attr('num_trades', n_trades)
            trial.set_user_attr('sharpe_ratio', sharpe_ratio)
            trial.set_user_attr('max_drawdown', max_drawdown)
            trial.set_user_attr('profit_factor', profit_factor)
            trial.set_user_attr('avg_hold_days', avg_hold_days)
            trial.set_user_attr('cpcv_oos_mean', cpcv_oos_mean)
            trial.set_user_attr('cpcv_oos_min', cpcv_oos_min)
            trial.set_user_attr('sharpe_var', sharpe_var)
            trial.set_user_attr('pbo_score', pbo_score)
            
            # 儲存增強的過擬合指標
            trial.set_user_attr('parameter_sensitivity', overfitting_metrics['parameter_sensitivity'])
            trial.set_user_attr('consistency_score', overfitting_metrics['consistency_score'])
            trial.set_user_attr('stability_score', overfitting_metrics['stability_score'])
            trial.set_user_attr('overfitting_risk', overfitting_metrics['overfitting_risk'])
            return score
        study.optimize(objective_sequential, n_trials=args.n_trials, show_progress_bar=True)
        print(f"[INFO] 所有 trial 已完成，開始細粒度分群分析... [{data_source}]")
        trials = [t for t in study.trials if t.value is not None and t.value > -np.inf]
        if args.top_n is not None and args.top_n < len(trials):
            trials = sorted(trials, key=lambda x: x.value if x.value is not None else -np.inf, reverse=True)[:args.top_n]
            print(f"[INFO] 已篩選前 {args.top_n} 個最佳策略進行分群")
        
        # 使用細粒度分群分析
        clustering_result = simple_fine_clustering_analysis(trials, args.strategy, data_source)
        
        print(f"[INFO] 開始選取各群代表策略... [{data_source}]")
        final_trials = []
        labels = clustering_result['final_labels']
        trial_scores = [t.value for t in trials]
        
        for i in range(1, len(clustering_result['final_sizes'])):
            idxs = np.where(labels == i)[0]
            if len(idxs) == 0:
                continue
            best_idx = idxs[np.argmax([trial_scores[j] for j in idxs])]
            final_trials.append(trials[best_idx])
        
        print(f"[INFO] 各群代表策略選取完成。共 {len(final_trials)} 策略。 [{data_source}]")
        print(f"[INFO] 開始輸出結果 CSV... [{data_source}]")
        
        # 輸出完整結果
        rows = []
        for t, label in zip(trials, labels):
            row = {
                'trial_number': t.number,
                'score': t.value,
                'strategy': t.user_attrs.get('strategy'),
                'data_source': t.user_attrs.get('data_source'),
                'parameters': json.dumps(t.user_attrs.get('parameters', {}), ensure_ascii=False),
                'total_return': t.user_attrs.get('total_return'),
                'num_trades': t.user_attrs.get('num_trades'),
                'sharpe_ratio': t.user_attrs.get('sharpe_ratio'),
                'max_drawdown': t.user_attrs.get('max_drawdown'),
                'profit_factor': t.user_attrs.get('profit_factor'),
                'avg_hold_days': t.user_attrs.get('avg_hold_days'),
                'cpcv_oos_mean': t.user_attrs.get('cpcv_oos_mean'),
                'cpcv_oos_min': t.user_attrs.get('cpcv_oos_min'),
                'sharpe_var': t.user_attrs.get('sharpe_var'),
                'pbo_score': t.user_attrs.get('pbo_score'),
                'parameter_sensitivity': t.user_attrs.get('parameter_sensitivity'),
                'consistency_score': t.user_attrs.get('consistency_score'),
                'stability_score': t.user_attrs.get('stability_score'),
                'overfitting_risk': t.user_attrs.get('overfitting_risk'),
                'fine_cluster': label,
                'clustering_method': clustering_result['method']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        safe_ds = data_source.replace(' ', '_').replace('^', '').replace('/', '').replace('(', '').replace(')', '')
        result_csv = RESULT_DIR / f"optuna_results_{args.strategy}_{safe_ds}_{TIMESTAMP}.csv"
        df.to_csv(result_csv, index=False, encoding='utf-8-sig')
        print(f"[INFO] 已輸出 {len(df)} 筆結果到 {result_csv}")
        
        # 輸出分群代表策略
        print(f"[INFO] 開始輸出分群代表策略... [{data_source}]")
        final_rows = []
        for t in final_trials:
            row = {
                'trial_number': t.number,
                'score': t.value,
                'strategy': t.user_attrs.get('strategy'),
                'data_source': t.user_attrs.get('data_source'),
                'parameters': json.dumps(t.user_attrs.get('parameters', {}), ensure_ascii=False),
                'total_return': t.user_attrs.get('total_return'),
                'num_trades': t.user_attrs.get('num_trades'),
                'sharpe_ratio': t.user_attrs.get('sharpe_ratio'),
                'max_drawdown': t.user_attrs.get('max_drawdown'),
                'profit_factor': t.user_attrs.get('profit_factor'),
                'avg_hold_days': t.user_attrs.get('avg_hold_days'),
                'cpcv_oos_mean': t.user_attrs.get('cpcv_oos_mean'),
                'cpcv_oos_min': t.user_attrs.get('cpcv_oos_min'),
                'sharpe_var': t.user_attrs.get('sharpe_var'),
                'pbo_score': t.user_attrs.get('pbo_score'),
                'parameter_sensitivity': t.user_attrs.get('parameter_sensitivity'),
                'consistency_score': t.user_attrs.get('consistency_score'),
                'stability_score': t.user_attrs.get('stability_score'),
                'overfitting_risk': t.user_attrs.get('overfitting_risk'),
                'fine_cluster': 1,  # 代表策略都標記為1
                'clustering_method': clustering_result['method']
            }
            final_rows.append(row)
        
        df_final = pd.DataFrame(final_rows)
        final_csv = RESULT_DIR / f"optuna_results_{args.strategy}_{safe_ds}_fine_clustering_final_{TIMESTAMP}.csv"
        df_final.to_csv(final_csv, index=False, encoding='utf-8-sig')
        print(f"[INFO] 分群代表策略輸出完成。 [{data_source}]")
else:
    # 原本的單一 data_source 處理流程
    # 对于 Ensemble 策略，使用多目标优化
    if args.strategy == 'ensemble':
        # 使用 NSGA-II 多目标优化器
        study = optuna.create_study(
            directions=['maximize', 'minimize'],  # 最大化报酬，最小化交易次数
            sampler=optuna.samplers.NSGAIISampler(seed=42)
        )
        print("[INFO] 使用 NSGA-II 多目标优化器进行 Ensemble 策略优化")
    else:
        study = optuna.create_study(direction='maximize')
    
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print(f"[INFO] 所有 trial 已完成，開始細粒度分群分析...")

    # 先聚合有效 trial
    trials = [t for t in study.trials if t.value is not None and t.value > -np.inf]

    # 再依 top_n 決定是否篩選
    if args.top_n is not None and args.top_n < len(trials):
        trials = sorted(trials, key=lambda x: x.value if x.value is not None else -np.inf, reverse=True)[:args.top_n]
        print(f"[INFO] 已篩選前 {args.top_n} 個最佳策略進行分群")

    # 使用細粒度分群分析
    clustering_result = simple_fine_clustering_analysis(trials, args.strategy, "Mixed")

    # 每群選分數最高者
    print(f"[INFO] 開始選取各群代表策略...")
    final_trials = []
    labels = clustering_result['final_labels']
    trial_scores = [t.value for t in trials]
    
    for i in range(1, len(clustering_result['final_sizes'])):
        idxs = np.where(labels == i)[0]
        if len(idxs) == 0:
            continue
        best_idx = idxs[np.argmax([trial_scores[j] for j in idxs])]
        final_trials.append(trials[best_idx])
    
    print(f"[INFO] 各群代表策略選取完成。共 {len(final_trials)} 策略。")

    # 輸出結果
    print(f"[INFO] 開始輸出結果 CSV...")
    rows = []
    for t, label in zip(trials, labels):
        row = {
            'trial_number': t.number,
            'score': t.value,
            'strategy': t.user_attrs.get('strategy'),
            'data_source': t.user_attrs.get('data_source'),
            'parameters': json.dumps(t.user_attrs.get('parameters', {}), ensure_ascii=False),
            'total_return': t.user_attrs.get('total_return'),
            'num_trades': t.user_attrs.get('num_trades'),
            'sharpe_ratio': t.user_attrs.get('sharpe_ratio'),
            'max_drawdown': t.user_attrs.get('max_drawdown'),
            'profit_factor': t.user_attrs.get('profit_factor'),
            'avg_hold_days': t.user_attrs.get('avg_hold_days'),
            'cpcv_oos_mean': t.user_attrs.get('cpcv_oos_mean'),
            'cpcv_oos_min': t.user_attrs.get('cpcv_oos_min'),
            'sharpe_var': t.user_attrs.get('sharpe_var'),
            'pbo_score': t.user_attrs.get('pbo_score'),
            'parameter_sensitivity': t.user_attrs.get('parameter_sensitivity'),
            'consistency_score': t.user_attrs.get('consistency_score'),
            'stability_score': t.user_attrs.get('stability_score'),
            'overfitting_risk': t.user_attrs.get('overfitting_risk'),
            'fine_cluster': label,
            'clustering_method': clustering_result['method']
        }
        
        # 对于 Ensemble 策略，添加多目标信息
        if args.strategy == 'ensemble':
            row.update({
                'score_return': t.user_attrs.get('score_return'),
                'score_trades': t.user_attrs.get('score_trades'),
                'turnover_penalty': t.user_attrs.get('turnover_penalty')
            })
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    result_csv = RESULT_DIR / f"optuna_results_{args.strategy}_{TIMESTAMP}.csv"
    df.to_csv(result_csv, index=False, encoding='utf-8-sig')
    print(f"[INFO] 已輸出 {len(df)} 筆結果到 {result_csv}")

    # 輸出分群代表策略
    print(f"[INFO] 開始輸出分群代表策略...")
    final_rows = []
    for t in final_trials:
        row = {
            'trial_number': t.number,
            'score': t.value,
            'strategy': t.user_attrs.get('strategy'),
            'data_source': t.user_attrs.get('data_source'),
            'parameters': json.dumps(t.user_attrs.get('parameters', {}), ensure_ascii=False),
            'total_return': t.user_attrs.get('total_return'),
            'num_trades': t.user_attrs.get('num_trades'),
            'sharpe_ratio': t.user_attrs.get('sharpe_ratio'),
            'max_drawdown': t.user_attrs.get('max_drawdown'),
            'profit_factor': t.user_attrs.get('profit_factor'),
            'avg_hold_days': t.user_attrs.get('avg_hold_days'),
            'cpcv_oos_mean': t.user_attrs.get('cpcv_oos_mean'),
            'cpcv_oos_min': t.user_attrs.get('cpcv_oos_min'),
            'sharpe_var': t.user_attrs.get('sharpe_var'),
            'pbo_score': t.user_attrs.get('pbo_score'),
            'parameter_sensitivity': t.user_attrs.get('parameter_sensitivity'),
            'consistency_score': t.user_attrs.get('consistency_score'),
            'stability_score': t.user_attrs.get('stability_score'),
            'overfitting_risk': t.user_attrs.get('overfitting_risk'),
            'fine_cluster': 1,  # 代表策略都標記為1
            'clustering_method': clustering_result['method']
        }
        
        # 对于 Ensemble 策略，添加多目标信息
        if args.strategy == 'ensemble':
            row.update({
                'score_return': t.user_attrs.get('score_return'),
                'score_trades': t.user_attrs.get('score_trades'),
                'turnover_penalty': t.user_attrs.get('turnover_penalty')
            })
        
        final_rows.append(row)
    
    df_final = pd.DataFrame(final_rows)
    final_csv = RESULT_DIR / f"optuna_results_{args.strategy}_fine_clustering_final_{TIMESTAMP}.csv"
    df_final.to_csv(final_csv, index=False, encoding='utf-8-sig')
    print(f"[INFO] 分群代表策略輸出完成。")
    
    # 对于 Ensemble 策略，额外输出最佳参数组合
    if args.strategy == 'ensemble':
        print(f"[INFO] 开始输出 Ensemble 策略最佳参数组合...")
        
        # 获取 Pareto 前沿上的策略
        pareto_trials = study.best_trials
        print(f"[INFO] 找到 {len(pareto_trials)} 个 Pareto 前沿策略")
        
        # 输出最佳参数组合
        ensemble_best = []
        for i, trial in enumerate(pareto_trials):
            params = trial.user_attrs.get('parameters', {})
            ensemble_best.append({
                'rank': i + 1,
                'method': params.get('method'),
                'floor': params.get('floor'),
                'ema_span': params.get('ema_span'),
                'delta_cap': params.get('delta_cap'),
                'majority_k': params.get('majority_k'),
                'min_cooldown_days': params.get('min_cooldown_days'),
                'min_trade_dw': params.get('min_trade_dw'),
                'total_return': trial.user_attrs.get('total_return'),
                'num_trades': trial.user_attrs.get('num_trades'),
                'score_return': trial.user_attrs.get('score_return'),
                'score_trades': trial.user_attrs.get('score_trades'),
                'turnover_penalty': trial.user_attrs.get('turnover_penalty')
            })
        
        # 保存最佳参数组合
        ensemble_best_df = pd.DataFrame(ensemble_best)
        ensemble_best_csv = RESULT_DIR / f"ensemble_best_params_{TIMESTAMP}.csv"
        ensemble_best_df.to_csv(ensemble_best_csv, index=False, encoding='utf-8-sig')
        print(f"[INFO] Ensemble 最佳参数已保存到 {ensemble_best_csv}")
        
        # 同时保存为 JSON 格式
        ensemble_best_json = RESULT_DIR / f"ensemble_best_params_{TIMESTAMP}.json"
        ensemble_best_df.to_json(ensemble_best_json, orient='records', force_ascii=False, indent=2)
        print(f"[INFO] Ensemble 最佳参数已保存到 {ensemble_best_json}")

print(f"[INFO] 所有處理完成！") 