import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import ast
import shutil
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_data
from SSSv095b2 import backtest_unified, param_presets, compute_single, compute_RMA, compute_ssma_turn_combined
from config import RESULT_DIR, WF_PERIODS, STRESS_PERIODS, CACHE_DIR
from metrics import calculate_max_drawdown
from logging_config import setup_logging
import logging

# 初始化快取目錄
shutil.rmtree(CACHE_DIR, ignore_errors=True)
(CACHE_DIR / "price").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "smaa").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "factor").mkdir(parents=True, exist_ok=True)

# Streamlit UI 配置
st.set_page_config(layout="wide")
st.title("00631L 策略回測與走查分析")
setup_logging()  # 初始化統一日誌設定
logger = logging.getLogger("OSv3")  # 修正為正確的日誌器名稱

# 動態生成走查區間
def generate_walk_forward_periods(data_index, n_splits, min_days=30):
    """根據數據日期範圍生成平分的走查區間"""
    if data_index.empty:
        st.error("數據索引為空，無法生成走查區間")
        logger.error("數據索引為空，無法生成走查區間")
        return []
    start_date = data_index.min()
    end_date = data_index.max()
    total_days = (end_date - start_date).days
    if total_days < min_days:
        st.error(f"數據範圍過短（{total_days} 天），無法生成走查區間")
        logger.error(f"數據範圍過短（{total_days} 天），無法生成走查區間")
        return []
    if total_days < min_days * n_splits:
        n_splits = max(1, total_days // min_days)
        logger.warning(f"數據範圍過短，調整分段數為 {n_splits}")
    days_per_split = total_days // n_splits
    periods = []
    current_start = start_date
    for i in range(n_splits):
        if i == n_splits - 1:
            current_end = end_date
        else:
            current_end = current_start + pd.Timedelta(days=days_per_split - 1)
            end_candidates = data_index[data_index <= current_end]
            if end_candidates.empty:
                break
            current_end = end_candidates[-1]
        if (current_end - current_start).days >= min_days:
            periods.append((current_start, current_end))
        current_start = current_end + pd.Timedelta(days=1)
        start_candidates = data_index[data_index >= current_start]
        if start_candidates.empty:
            break
        current_start = start_candidates[0]
    return periods

# 選擇前 20 個多樣化試驗
def pick_topN_by_diversity(trials, ind_keys, top_n=20, pct_threshold=25):
    trials_sorted = sorted(trials, key=lambda t: -t['score'])
    selected = []
    vectors = []
    for trial in trials_sorted:
        vec = np.array([trial['parameters'][k] for k in ind_keys])
        if not vectors:
            selected.append(trial)
            vectors.append(vec)
            continue
        dists = cdist([vec], vectors, metric="euclidean")[0]
        min_dist = min(dists)
        if len(dists) > 1:
            threshold = np.percentile(dists, pct_threshold)
        else:
            threshold = 0
        if min_dist >= threshold:
            selected.append(trial)
            vectors.append(vec)
        if len(selected) == top_n:
            break
    return selected

# 計算 PR 值
def calculate_pr_values(series, is_mdd, is_initial_period, hedge_mask):
    """
    計算 PR 值，根據是否為初始時段和避險掩碼進行處理
    :param series: 要計算 PR 的系列（報酬率或 MDD）
    :param is_mdd: 是否為 MDD（True）或報酬率（False）
    :param is_initial_period: 是否為初始時段
    :param hedge_mask: 避險掩碼，True 表示該策略在該時段避險
    :return: PR 值系列
    """
    pr_values = pd.Series(index=series.index, dtype=float)
    if is_initial_period:
        # 初始時段：排除避險策略
        valid_series = series[~hedge_mask]
    else:
        # 非初始時段：對避險策略設置 PR=100
        pr_values[hedge_mask] = 100
        valid_series = series[~hedge_mask]
    if not valid_series.empty:
        # 計算有效系列的 PR 值
        ranks = valid_series.rank(ascending=is_mdd)  # MDD 升序（低值高排名），報酬率降序（高值高排名）
        if len(valid_series) > 1:
            pr_values_valid = (ranks - 1) / (len(valid_series) - 1) * 100
        else:
            pr_values_valid = pd.Series(100, index=valid_series.index)
        pr_values[valid_series.index] = pr_values_valid
    else:
        # 如果沒有有效策略，設置為 NaN
        pr_values[:] = np.nan
    return pr_values

# 載入 Optuna 結果
# 動態尋找最新的optuna results文件
optuna_files = list(RESULT_DIR.glob("optuna_results_*.csv"))
if not optuna_files:
    st.error("未找到optuna results文件")
    st.stop()

# 按修改時間排序，選擇最新的文件
latest_file = max(optuna_files, key=lambda x: x.stat().st_mtime)
optuna_results = pd.read_csv(latest_file)
optuna_results['parameters'] = optuna_results['parameters'].apply(ast.literal_eval)
top_trials = optuna_results.sort_values(by='score', ascending=False)

st.info(f"載入optuna results文件: {latest_file.name}")

ind_keys = ['linlen', 'factor', 'smaalen', 'devwin']
selected_trials = pick_topN_by_diversity(top_trials.to_dict('records'), ind_keys)

# 建立策略清單
optuna_strategies = [
    {
        'name': f"trial_{trial['trial_number']}",
        'strategy_type': trial['strategy'],
        'params': trial['parameters'],
        'smaa_source': trial['data_source']
    } for trial in selected_trials
]
preset_strategies = [
    {
        'name': key,
        'strategy_type': value['strategy_type'],
        'params': {k: v for k, v in value.items() if k not in ['strategy_type', 'smaa_source']},
        'smaa_source': value['smaa_source']
    } for key, value in param_presets.items()
]
all_strategies = optuna_strategies + preset_strategies

# UI：選擇走查模式
st.sidebar.header("走查設定")
walk_forward_mode = st.sidebar.selectbox("走查模式", ["固定 WF_PERIODS", "動態平分區間"])
if walk_forward_mode == "動態平分區間":
    n_splits = st.sidebar.number_input("分段數", min_value=1, max_value=10, value=3, step=1)

# 新增：調試模式開關
debug_mode = st.sidebar.checkbox("啟用單一策略調試模式 (SSMA_turn 0)")

run_backtest = st.sidebar.button("執行回測")

# 載入數據並確定全域時間軸
smaa_sources = set(strategy['smaa_source'] for strategy in all_strategies)
df_price_dict = {}
df_factor_dict = {} # 新增: 用於儲存因子數據
all_indices = []
for source in smaa_sources:
    df_price, df_factor = load_data(ticker="00631L.TW", smaa_source=source)
    if not df_price.empty:
        all_indices.append(df_price.index)
    df_price_dict[source] = df_price
    df_factor_dict[source] = df_factor # 儲存因子數據

# 合併所有時間軸，創建一個全域時間軸
if all_indices:
    global_index = pd.Index([])
    for index in all_indices:
        global_index = global_index.union(index)
else:
    global_index = pd.Index([])

# 回測與分析
if run_backtest:
    if debug_mode:
        st.warning("調試模式已啟用：只會測試 SSMA_turn 0")
        logger.warning("調試模式已啟用：只會測試 SSMA_turn 0")
        # 在調試模式下，只使用 SSMA_turn 0 的參數
        debug_params = {
            "linlen": 25, "smaalen": 85, "factor": 80.0, "prom_factor": 9, 
            "min_dist": 8, "buy_shift": 0, "exit_shift": 6, "vol_window": 90, 
            "quantile_win": 65, "signal_cooldown_days": 7, "buy_mult": 0.15, 
            "sell_mult": 0.1, "stop_loss": 0.13, "smaa_source": "Factor (^TWII / 2414.TW)"
        }
        all_strategies = [{
            'name': 'SSMA_turn 0 (Debug)',
            'strategy_type': 'ssma_turn',
            'params': debug_params,
            'smaa_source': debug_params['smaa_source']
        }]
    
    with st.spinner("正在執行回測..."):
        results = {}
        initial_equity = 100000.0  # 初始權益值
        for strategy in all_strategies:
            name = strategy['name']
            strategy_type = strategy['strategy_type']
            params = strategy['params']
            smaa_source = strategy['smaa_source']
            df_price = df_price_dict[smaa_source]
            df_factor = df_factor_dict[smaa_source] # 獲取對應的 df_factor
            
            logger.info(f"處理策略 {name}，參數: {params}")
            
            if strategy_type == 'single':
                df_ind = compute_single(df_price, df_factor, params['linlen'], params['factor'], params['smaalen'], params['devwin'], smaa_source=smaa_source)
                logger.info(f"策略 {name} 的 df_ind 形狀: {df_ind.shape}, 欄位: {df_ind.columns.tolist()}")
            elif strategy_type == 'RMA':
                df_ind = compute_RMA(df_price, df_factor, params['linlen'], params['factor'], params['smaalen'], params['rma_len'], params['dev_len'], smaa_source=smaa_source)
                logger.info(f"策略 {name} 的 df_ind 形狀: {df_ind.shape}, 欄位: {df_ind.columns.tolist()}")
            elif strategy_type == 'ssma_turn':
                calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
                ssma_params = {k: v for k, v in params.items() if k in calc_keys}
                backtest_params = ssma_params.copy()
                backtest_params['buy_mult'] = params.get('buy_mult', 0.5)
                backtest_params['sell_mult'] = params.get('sell_mult', 0.5)
                backtest_params['stop_loss'] = params.get('stop_loss', 0.0)
                logger.info(f"SSMA_Turn 參數: {ssma_params}")
                df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(df_price, df_factor, **ssma_params, smaa_source=smaa_source)
                logger.info(f"策略 {name} 生成的買入信號數: {len(buy_dates)}, 賣出信號數: {len(sell_dates)}")
                logger.info(f"買入信號: {buy_dates[:5] if buy_dates else '無'}")
                logger.info(f"賣出信號: {sell_dates[:5] if sell_dates else '無'}")
                if df_ind.empty:
                    logger.error(f"策略 {name} 的 df_ind 為空，跳過回測")
                    st.error(f"策略 {name} 的 df_ind 為空，跳過回測")
                    continue
                # 使用與SSSv095b2一致的預設參數設定
                result = backtest_unified(df_ind, strategy_type, params, buy_dates, sell_dates, 
                                         discount=0.30, trade_cooldown_bars=3, bad_holding=False)
                results[name] = result
                continue
            
            required_cols = ['open', 'close', 'smaa', 'base', 'sd']
            if df_ind.empty or not all(col in df_ind.columns for col in required_cols):
                logger.error(f"策略 {name} 的 df_ind 缺少必要欄位: {set(required_cols) - set(df_ind.columns)}，跳過回測")
                st.error(f"策略 {name} 的 df_ind 缺少必要欄位: {set(required_cols) - set(df_ind.columns)}，跳過回測")
                continue
            
            # 對於 single 和 RMA，目前的回測參數是固定的
            # 注意：如果這些策略也需要不同的 discount 或 cooldown，這裡需要修改
            if df_ind.empty:
                logger.error(f"策略 {name} 的 df_ind 為空，跳過計算")
                continue
            
            result = backtest_unified(df_ind, strategy_type, params, discount=0.30, trade_cooldown_bars=3, bad_holding=False)
            results[name] = result

        # 提取權益曲線
        equity_curves = pd.DataFrame({name: result['equity_curve'].reindex(global_index, fill_value=initial_equity) for name, result in results.items()})
        if equity_curves.empty:
            logger.error("未生成任何權益曲線，請檢查回測邏輯或數據")
            st.error("未生成任何權益曲線，請檢查回測邏輯或數據")
        else:
            logger.info(f"權益曲線形狀: {equity_curves.shape}")
            logger.info(equity_curves.head())

            # 使用標籤頁呈現結果
            tabs = st.tabs(["相關性矩陣熱圖", "報酬率相關", "最大回撤相關", "總結"])

            # 標籤頁 1: 相關性矩陣熱圖
            with tabs[0]:
                # 計算相關性矩陣
                corr_matrix = equity_curves.corr()
                corr_matrix.to_csv(RESULT_DIR / 'correlation_matrix.csv')
                st.subheader("相關性矩陣熱圖")
                # 選擇顯示的策略
                selected_strategies = st.multiselect("選擇顯示的策略", options=corr_matrix.columns.tolist(), default=corr_matrix.columns.tolist())
                if not selected_strategies:
                    st.warning("請至少選擇一個策略")
                else:
                    filtered_corr = corr_matrix.loc[selected_strategies, selected_strategies]
                    # 調整刻度範圍
                    zmin = st.slider("最小相關性值", -1.0, 0.0, -1.0, 0.1)
                    zmax = st.slider("最大相關性值", 0.0, 1.0, 1.0, 0.1)
                    fig = px.imshow(
                        filtered_corr,
                        color_continuous_scale='RdBu_r',
                        zmin=zmin,
                        zmax=zmax,
                        text_auto='.2f',
                        title="策略相關性矩陣"
                    )
                    fig.update_layout(
                        width=800,
                        height=600,
                        xaxis_title="策略",
                        yaxis_title="策略",
                        coloraxis_colorbar_title="相關性"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # 標籤頁 2: 報酬率相關
            with tabs[1]:
                st.subheader("走查時段報酬率 (%)")
                period_returns = {}
                period_mdd = {}
                period_pr = {}
                period_mdd_pr = {}
                period_hedge_counts = {}
                periods = WF_PERIODS if walk_forward_mode == "固定 WF_PERIODS" else generate_walk_forward_periods(equity_curves.index, n_splits)

                for i, period in enumerate([(p['test'] if isinstance(p, dict) else p) for p in periods]):
                    start = pd.to_datetime(period[0])
                    end = pd.to_datetime(period[1])
                    valid_dates = equity_curves.index
                    if start < valid_dates[0]:
                        adjusted_start = valid_dates[0]
                        logger.warning(f"起始日期 {start.strftime('%Y-%m-%d')} 早於數據開始，調整為 {adjusted_start.strftime('%Y-%m-%d')}")
                    else:
                        adjusted_start = valid_dates[valid_dates >= start][0]
                    if end > valid_dates[-1]:
                        adjusted_end = valid_dates[-1]
                        logger.warning(f"結束日期 {end.strftime('%Y-%m-%d')} 晚於數據結束，調整為 {adjusted_end.strftime('%Y-%m-%d')}")
                    else:
                        adjusted_end = valid_dates[valid_dates <= end][-1]
                    if (adjusted_end - adjusted_start).days < 30:
                        logger.warning(f"走查區間 {adjusted_start.strftime('%Y-%m-%d')} 至 {adjusted_end.strftime('%Y-%m-%d')} 過短，跳過")
                        continue
                    period_equity = equity_curves.loc[adjusted_start:adjusted_end]
                    for col in period_equity.columns:
                        if pd.isna(period_equity[col].iloc[0]):
                            period_equity[col].iloc[0] = initial_equity
                    # 計算報酬率和 MDD
                    period_return = (period_equity.iloc[-1] / period_equity.iloc[0] - 1) * 100
                    period_mdd_value = period_equity.apply(calculate_max_drawdown) * 100
                    # 計算避險掩碼
                    hedge_mask = (period_return == 0) & (period_mdd_value == 0)
                    # 記錄避險次數
                    if i > 0:  # 非初始時段
                        period_hedge_counts[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = hedge_mask.astype(int)
                    else:  # 初始時段
                        period_hedge_counts[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = pd.Series(0, index=hedge_mask.index)
                    # 計算報酬率 PR 值
                    pr_values_ret = calculate_pr_values(period_return, is_mdd=False, is_initial_period=(i == 0), hedge_mask=hedge_mask)
                    period_pr[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = pr_values_ret
                    # 儲存報酬率
                    period_returns[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = period_return

                period_returns_df = pd.DataFrame(period_returns).T
                period_returns_df.to_csv(RESULT_DIR / 'period_returns.csv')
                st.dataframe(period_returns_df.style.format("{:.2f}%"))
                period_pr_df = pd.DataFrame(period_pr).T
                period_pr_df.to_csv(RESULT_DIR / 'period_pr.csv')
                st.subheader("走查時段報酬率 PR 值 (%)")
                st.dataframe(period_pr_df.style.format("{:.2f}"))
                period_hedge_counts_df = pd.DataFrame(period_hedge_counts).T
                period_hedge_counts_df.to_csv(RESULT_DIR / 'period_hedge_counts.csv')
                st.subheader("走查時段避險次數")
                st.dataframe(period_hedge_counts_df)
                stress_returns = {}
                stress_pr = {}
                stress_hedge_counts = {}
                for i, (start, end) in enumerate(STRESS_PERIODS):
                    start = pd.to_datetime(start)
                    end = pd.to_datetime(end)
                    if start in equity_curves.index and end in equity_curves.index:
                        period_equity = equity_curves.loc[start:end]
                        for col in period_equity.columns:
                            if pd.isna(period_equity[col].iloc[0]):
                                period_equity[col].iloc[0] = initial_equity
                        # 計算報酬率和 MDD
                        period_return = (period_equity.iloc[-1] / period_equity.iloc[0] - 1) * 100
                        period_mdd_value = period_equity.apply(calculate_max_drawdown) * 100
                        # 計算避險掩碼
                        hedge_mask = (period_return == 0) & (period_mdd_value == 0)
                        # 記錄避險次數（壓力時段第一個為初始時段）
                        if i > 0:  # 非初始時段
                            stress_hedge_counts[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = hedge_mask.astype(int)
                        else:  # 初始時段
                            stress_hedge_counts[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = pd.Series(0, index=hedge_mask.index)
                        # 計算報酬率 PR 值
                        pr_values_ret = calculate_pr_values(period_return, is_mdd=False, is_initial_period=(i == 0), hedge_mask=hedge_mask)
                        stress_pr[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = pr_values_ret
                        # 儲存報酬率
                        stress_returns[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = period_return

                stress_returns_df = pd.DataFrame(stress_returns).T
                stress_returns_df.to_csv(RESULT_DIR / 'stress_returns.csv')
                st.subheader("壓力時段報酬率 (%)")
                st.dataframe(stress_returns_df.style.format("{:.2f}%"))
                stress_pr_df = pd.DataFrame(stress_pr).T
                stress_pr_df.to_csv(RESULT_DIR / 'stress_pr.csv')
                st.subheader("壓力時段報酬率 PR 值 (%)")
                st.dataframe(stress_pr_df.style.format("{:.2f}"))
                stress_hedge_counts_df = pd.DataFrame(stress_hedge_counts).T
                stress_hedge_counts_df.to_csv(RESULT_DIR / 'stress_hedge_counts.csv')
                st.subheader("壓力時段避險次數")
                st.dataframe(stress_hedge_counts_df)

            # 標籤頁 3: 最大回撤相關
            with tabs[2]:
                st.subheader("走查時段最大回撤 (%)")
                period_mdd = {}
                period_mdd_pr = {}
                period_mdd_hedge_counts = {}
                periods = WF_PERIODS if walk_forward_mode == "固定 WF_PERIODS" else generate_walk_forward_periods(equity_curves.index, n_splits)
                for i, period in enumerate([(p['test'] if isinstance(p, dict) else p) for p in periods]):
                    start = pd.to_datetime(period[0])
                    end = pd.to_datetime(period[1])
                    valid_dates = equity_curves.index
                    if start < valid_dates[0]:
                        adjusted_start = valid_dates[0]
                    else:
                        adjusted_start = valid_dates[valid_dates >= start][0]
                    if end > valid_dates[-1]:
                        adjusted_end = valid_dates[-1]
                    else:
                        adjusted_end = valid_dates[valid_dates <= end][-1]
                    if (adjusted_end - adjusted_start).days < 30:
                        continue
                    period_equity = equity_curves.loc[adjusted_start:adjusted_end]
                    for col in period_equity.columns:
                        if pd.isna(period_equity[col].iloc[0]):
                            period_equity[col].iloc[0] = initial_equity
                    # 計算報酬率（用於避險掩碼）
                    period_return = (period_equity.iloc[-1] / period_equity.iloc[0] - 1) * 100
                    # 計算 MDD
                    period_mdd_value = period_equity.apply(calculate_max_drawdown) * 100
                    # 計算避險掩碼
                    hedge_mask = (period_return == 0) & (period_mdd_value == 0)
                    # 記錄避險次數
                    if i > 0:  # 非初始時段
                        period_mdd_hedge_counts[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = hedge_mask.astype(int)
                    else:  # 初始時段
                        period_mdd_hedge_counts[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = pd.Series(0, index=hedge_mask.index)
                    # 計算 MDD PR 值
                    pr_values_mdd = calculate_pr_values(period_mdd_value, is_mdd=True, is_initial_period=(i == 0), hedge_mask=hedge_mask)
                    period_mdd_pr[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = pr_values_mdd
                    # 儲存 MDD
                    period_mdd[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = period_mdd_value

                period_mdd_df = pd.DataFrame(period_mdd).T
                period_mdd_df.to_csv(RESULT_DIR / 'period_mdd.csv')
                st.dataframe(period_mdd_df.style.format("{:.2f}%"))
                period_mdd_pr_df = pd.DataFrame(period_mdd_pr).T
                period_mdd_pr_df.to_csv(RESULT_DIR / 'period_mdd_pr.csv')
                st.subheader("走查時段最大回撤 PR 值 (%)")
                st.dataframe(period_mdd_pr_df.style.format("{:.2f}"))
                period_mdd_hedge_counts_df = pd.DataFrame(period_mdd_hedge_counts).T
                period_mdd_hedge_counts_df.to_csv(RESULT_DIR / 'period_mdd_hedge_counts.csv')
                st.subheader("走查時段最大回撤避險次數")
                st.dataframe(period_mdd_hedge_counts_df)

                st.subheader("壓力時段最大回撤 (%)")
                stress_mdd = {}
                stress_mdd_pr = {}
                stress_mdd_hedge_counts = {}
                for i, (start, end) in enumerate(STRESS_PERIODS):
                    start = pd.to_datetime(start)
                    end = pd.to_datetime(end)
                    if start in equity_curves.index and end in equity_curves.index:
                        period_equity = equity_curves.loc[start:end]
                        for col in period_equity.columns:
                            if pd.isna(period_equity[col].iloc[0]):
                                period_equity[col].iloc[0] = initial_equity
                        # 計算報酬率和 MDD
                        period_return = (period_equity.iloc[-1] / period_equity.iloc[0] - 1) * 100
                        period_mdd_value = period_equity.apply(calculate_max_drawdown) * 100
                        # 計算避險掩碼
                        hedge_mask = (period_return == 0) & (period_mdd_value == 0)
                        # 記錄避險次數
                        if i > 0:  # 非初始時段
                            stress_mdd_hedge_counts[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = hedge_mask.astype(int)
                        else:  # 初始時段
                            stress_mdd_hedge_counts[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = pd.Series(0, index=hedge_mask.index)
                        # 計算 MDD PR 值
                        pr_values_mdd = calculate_pr_values(period_mdd_value, is_mdd=True, is_initial_period=(i == 0), hedge_mask=hedge_mask)
                        stress_mdd_pr[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = pr_values_mdd
                        # 儲存 MDD
                        stress_mdd[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = period_mdd_value

                stress_mdd_df = pd.DataFrame(stress_mdd).T
                stress_mdd_df.to_csv(RESULT_DIR / 'stress_mdd.csv')
                st.subheader("壓力時段最大回撤 (%)")
                st.dataframe(stress_mdd_df.style.format("{:.2f}%"))
                stress_mdd_pr_df = pd.DataFrame(stress_mdd_pr).T
                stress_mdd_pr_df.to_csv(RESULT_DIR / 'stress_mdd_pr.csv')
                st.subheader("壓力時段最大回撤 PR 值 (%)")
                st.dataframe(stress_mdd_pr_df.style.format("{:.2f}"))
                stress_mdd_hedge_counts_df = pd.DataFrame(stress_mdd_hedge_counts).T
                stress_mdd_hedge_counts_df.to_csv(RESULT_DIR / 'stress_mdd_hedge_counts.csv')
                st.subheader("壓力時段最大回撤避險次數")
                st.dataframe(stress_mdd_hedge_counts_df)

                st.subheader("整體最大回撤 (%)")
                mdd = {}
                for name, equity in equity_curves.items():
                    mdd[name] = calculate_max_drawdown(equity) * 100
                mdd_df = pd.Series(mdd)
                mdd_df.to_csv(RESULT_DIR / 'mdd.csv')
                st.subheader("整體最大回撤 (%)")
                st.dataframe(mdd_df.to_frame().style.format("{:.2f}%"))

            # 標籤頁 4: 總結
            with tabs[3]:
                st.subheader("各策略平均 PR 值與避險次數")
                avg_pr = pd.concat([period_pr_df.mean(), stress_pr_df.mean(), period_mdd_pr_df.mean(), stress_mdd_pr_df.mean()], axis=1)
                avg_pr.columns = ['走查報酬率 PR', '壓力報酬率 PR', '走查MDD PR', '壓力MDD PR']
                avg_pr['平均 PR'] = avg_pr.mean(axis=1)
                wf_hedge_counts = pd.DataFrame(period_hedge_counts).T.sum()
                stress_hedge_counts = pd.DataFrame(stress_hedge_counts).T.sum()
                total_hedge_counts = wf_hedge_counts + stress_hedge_counts
                summary_df = pd.concat([avg_pr, wf_hedge_counts.to_frame('走查避險次數'), stress_hedge_counts.to_frame('壓力避險次數'), total_hedge_counts.to_frame('總避險次數')], axis=1)
                st.dataframe(summary_df.style.format("{:.2f}", subset=avg_pr.columns).format("{:d}", subset=['走查避險次數', '壓力避險次數', '總避險次數']))

            # 顯示總體回測結果
            st.subheader("策略權益曲線")

            equity_curves_list = []
            for name, result in results.items():
                if 'equity_curve' in result and not result['equity_curve'].empty:
                    equity_curve = result['equity_curve'].rename(name)
                    equity_curves_list.append(equity_curve)

            if not equity_curves_list:
                st.warning("沒有可用的權益曲線數據進行繪製。")
                logger.warning("沒有可用的權益曲線數據進行繪製。")
            else:
                all_equity_curves = pd.concat(equity_curves_list, axis=1)
                all_equity_curves = all_equity_curves.reindex(global_index, method='ffill')
                all_equity_curves.fillna(method='ffill', inplace=True)
                all_equity_curves.fillna(initial_equity, inplace=True)

                df_plot = all_equity_curves.reset_index().melt(id_vars='index', var_name='策略', value_name='權益')
                df_plot.rename(columns={'index': '日期'}, inplace=True)
                
                fig_equity = px.line(df_plot, x='日期', y='權益', color='策略', title="策略權益曲線")
                fig_equity.update_layout(legend_title_text='variable')
                st.plotly_chart(fig_equity, use_container_width=True, key="total_equity_curve")

            # 顯示匯總指標
            st.subheader("策略匯總指標")
            summary_data = []
            for name, result in results.items():
                if 'metrics' in result and result['metrics']:
                    metrics = result['metrics']
                    row = {
                        "策略": name,
                        "總報酬率": metrics.get('total_return', 0),
                        "年化報酬率": metrics.get('annual_return', 0),
                        "最大回撤": metrics.get('max_drawdown', 0),
                        "夏普比率": metrics.get('sharpe_ratio', 0),
                        "卡瑪比率": metrics.get('calmar_ratio', 0),
                        "交易次數": metrics.get('num_trades', 0),
                        "勝率": metrics.get('win_rate', 0)
                    }
                    summary_data.append(row)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data).set_index("策略")
                st.dataframe(summary_df.style.format({
                    "總報酬率": "{:.2%}", "年化報酬率": "{:.2%}", "最大回撤": "{:.2%}",
                    "夏普比率": "{:.2f}", "卡瑪比率": "{:.2f}", "勝率": "{:.2%}"
                }))
            else:
                st.warning("沒有可用的匯總指標數據。")

            # 顯示日誌內容
            st.subheader("日誌內容")
            log_file = Path("logs") / "OS.log"
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    log_content = f.read()
                st.code(log_content, language="text")
            else:
                logger.warning("日誌檔案不存在")

            # 蒙地卡羅測試建議
            st.info("蒙地卡羅測試：請參考 Optuna_12.py 中的 compute_pbo_score 和 compute_simplified_sra 函數實現 PBO 分數與 SRA p 值計算")

