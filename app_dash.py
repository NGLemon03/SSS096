import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
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

# 配置 logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from SSSv096 import (
    param_presets, load_data, compute_single, compute_dual, compute_RMA,
    compute_ssma_turn_combined, backtest_unified, plot_stock_price, plot_equity_cash, calculate_holding_periods
)

# 假設你有 get_version_history_html
try:
    from version_history import get_version_history_html
except ImportError:
    def get_version_history_html() -> str:
        return "<b>無法載入版本歷史記錄</b>"

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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

# --------- Dash Layout ---------
app.layout = html.Div([
    dcc.Store(id='theme-store', data='theme-dark'),
    dcc.Store(id='sidebar-collapsed', data=False),
    html.Div([
        html.Button(id='theme-toggle', n_clicks=0, children='🌑 深色主題', className='btn btn-secondary main-header-bar'),
        html.Button(id='sidebar-toggle', n_clicks=0, children='📋 隱藏側邊欄', className='btn btn-secondary main-header-bar ms-2'),
        html.Button(id='history-btn', n_clicks=0, children='📚 版本沿革', className='btn btn-secondary main-header-bar ms-2'),
    ], className='header-controls'),
    
    # 版本沿革模態框
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("各版本沿革紀錄")),
        dbc.ModalBody([
            dcc.Markdown(get_version_history_html(), dangerously_allow_html=True)
        ], className='version-history-modal-body'),
        dbc.ModalFooter(
            dbc.Button("關閉", id="history-close", className="ms-auto", n_clicks=0)
        ),
    ], id="history-modal", size="lg", is_open=False),
    
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("參數設定"),
                    dbc.Checkbox(id='auto-run', value=True, label="自動運算（參數變動即回測）", className="mb-2"),
                    html.Label("股票代號"),
                    dcc.Dropdown(
                        id='ticker-dropdown',
                        options=[{'label': t, 'value': t} for t in default_tickers],
                        value=default_tickers[0]
                    ),
                    html.Label("數據起始日期"),
                    dcc.Input(id='start-date', type='text', value='2010-01-01'),
                    html.Label("數據結束日期"),
                    dcc.Input(id='end-date', type='text', value=''),
                    html.Label("券商折數(0.7=7折, 0.1=1折)"),
                    dcc.Slider(id='discount-slider', min=0.1, max=0.7, step=0.01, value=0.3, marks={0.1:'0.1',0.3:'0.3',0.5:'0.5',0.7:'0.7'}),
                    html.Label("冷卻期 (bars)"),
                    dcc.Input(id='cooldown-bars', type='number', min=0, max=20, value=3),
                    dbc.Checkbox(id='bad-holding', value=False, label="賣出報酬率<-20%,等待下次賣點"),
                    html.Br(),
                    html.Label("策略選擇"),
                    dcc.Dropdown(
                        id='strategy-dropdown',
                        options=[{'label': s, 'value': s} for s in strategy_names],
                        value=strategy_names[0]
                    ),
                    html.Div(id='strategy-param-area'),  # 動態策略參數
                    html.Br(),
                    html.Button("🚀 一鍵執行所有回測", id='run-btn', n_clicks=0, className="btn btn-primary mb-2"),
                    dcc.Loading(dcc.Store(id='backtest-store'), type="circle"),
                ], id='sidebar-content')
            ], width=3, className='sidebar-panel', id='sidebar-col'),
            dbc.Col([
                dcc.Tabs(
                    id='main-tabs',
                    value='backtest',
                    children=[
                        dcc.Tab(label="策略回測", value="backtest"),
                        dcc.Tab(label="所有策略買賣點比較", value="compare")
                    ],
                    className='main-tabs-bar'
                ),
                html.Div(id='tab-content', className='main-content-panel')
            ], width=9, id='main-content-col')
        ])
    ], fluid=True)
], id='main-bg', className='theme-dark')

# --------- 側邊欄隱藏/顯示切換 ---------
@app.callback(
    Output('sidebar-col', 'width'),
    Output('main-content-col', 'width'),
    Output('sidebar-content', 'style'),
    Output('sidebar-toggle', 'children'),
    Input('sidebar-toggle', 'n_clicks'),
    State('sidebar-collapsed', 'data')
)
def toggle_sidebar(n_clicks, collapsed):
    if n_clicks is None:
        return 3, 9, {}, '📋 隱藏側邊欄'
    
    if collapsed:
        # 顯示側邊欄
        return 3, 9, {}, '📋 隱藏側邊欄'
    else:
        # 隱藏側邊欄
        return 0, 12, {'display': 'none'}, '📋 顯示側邊欄'

# --------- 側邊欄狀態存儲 ---------
@app.callback(
    Output('sidebar-collapsed', 'data'),
    Input('sidebar-toggle', 'n_clicks'),
    State('sidebar-collapsed', 'data')
)
def update_sidebar_state(n_clicks, collapsed):
    if n_clicks is None:
        return False
    return not collapsed

# --------- 動態顯示策略參數 ---------
@app.callback(
    Output('strategy-param-area', 'children'),
    Input('strategy-dropdown', 'value')
)
def update_strategy_params(strategy):
    params = param_presets[strategy]
    
    # 特殊處理 ensemble 策略，顯示參數摘要而不是輸入框
    if params.get('strategy_type') == 'ensemble':
        p = params.get('params', {})
        c = params.get('trade_cost', {})
        return html.Div([
            html.Div(f"method: {params.get('method')}"),
            html.Div(f"floor: {p.get('floor')} | ema_span: {p.get('ema_span')} | "
                     f"delta_cap: {p.get('delta_cap')} | cooldown: {p.get('min_cooldown_days')} | "
                     f"min_trade_dw: {p.get('min_trade_dw')} | majority_k: {p.get('majority_k', '-') }"),
            html.Div(f"cost(bps): buy {c.get('buy_fee_bp')}, sell {c.get('sell_fee_bp')}, tax {c.get('sell_tax_bp')}"),
            html.Small("（Ensemble 參數目前固定於 SSSv096.param_presets，如需調整建議在 SSSv096 內改）")
        ])
    
    # 其他策略照舊處理
    controls = []
    for k, v in params.items():
        if k in ['strategy_type', 'smaa_source']:
            continue
        if isinstance(v, (int, float)):
            controls.append(
                html.Div([
                    html.Label(f"{k}"),
                    dcc.Input(id={'type': 'param-input', 'param': k}, type='number', value=v)
                ])
            )
        else:
            controls.append(
                html.Div([
                    html.Label(f"{k}"),
                    dcc.Input(id={'type': 'param-input', 'param': k}, type='text', value=str(v))
                ])
            )
    return controls

# --------- 執行回測並存到 Store ---------
@app.callback(
    Output('backtest-store', 'data'),
    [
        Input('run-btn', 'n_clicks'),
        Input('auto-run', 'value'),
        Input('ticker-dropdown', 'value'),
        Input('start-date', 'value'),
        Input('end-date', 'value'),
        Input('discount-slider', 'value'),
        Input('cooldown-bars', 'value'),
        Input('bad-holding', 'value'),
        Input('strategy-dropdown', 'value'),
        Input({'type': 'param-input', 'param': ALL}, 'value'),
        Input({'type': 'param-input', 'param': ALL}, 'id'),
    ],
    State('backtest-store', 'data')
)
def run_backtest(n_clicks, auto_run, ticker, start_date, end_date, discount, cooldown, bad_holding, strategy, param_values, param_ids, stored_data):
    # 移除自動快取清理，避免多用户衝突
    # 讓 joblib.Memory 自動管理快取，只在需要時手動清理
    if n_clicks is None and not auto_run:
        return stored_data
    
    # 載入數據
    df_raw, df_factor = load_data(ticker, start_date, end_date, "Self")
    if df_raw.empty:
        return {"error": f"無法載入 {ticker} 的數據"}
    
    ctx_trigger = ctx.triggered_id
    # 只在 auto-run 為 True 或按鈕被點擊時運算
    if not auto_run and ctx_trigger != 'run-btn':
        return stored_data
    
    results = {}
    
    for strat in strategy_names:
        # 只使用 param_presets 中的參數
        strat_params = param_presets[strat].copy()
        strat_type = strat_params["strategy_type"]
        smaa_src = strat_params.get("smaa_source", "Self")
        
        # 為每個策略載入對應的數據
        df_raw, df_factor = load_data(ticker, start_date, end_date if end_date else None, smaa_source=smaa_src)
        
        if strat_type == 'ssma_turn':
            calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'signal_cooldown_days', 'quantile_win']
            ssma_params = {k: v for k, v in strat_params.items() if k in calc_keys}
            backtest_params = ssma_params.copy()
            backtest_params['stop_loss'] = strat_params.get('stop_loss', 0.0)
            df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(df_raw, df_factor, **ssma_params, smaa_source=smaa_src)
            if df_ind.empty:
                continue
            result = backtest_unified(df_ind, strat_type, backtest_params, buy_dates, sell_dates, discount=discount, trade_cooldown_bars=cooldown, bad_holding=bad_holding)
        elif strat_type == 'ensemble':
            # 用 price 當殼，直接調用 SSS_EnsembleTab 避免 streamlit 衝突
            df_ind = df_raw[['open','close']].copy()

            # 把 SSSv096 的巢狀參數攤平：
            flat_params = {}
            flat_params.update(strat_params.get('params', {}))       # floor / ema_span / delta_cap / ...
            flat_params.update(strat_params.get('trade_cost', {}))   # buy_fee_bp / sell_fee_bp / sell_tax_bp
            flat_params['method'] = strat_params.get('method', 'majority')
            flat_params['ticker'] = ticker  # 用 UI 選的 ticker
            
            # === 修復 1：使用比例門檻避免 N 變動時失真 ===
            if 'majority_k' in flat_params and flat_params.get('method') == 'majority':
                # 將固定 K 轉換為比例門檻
                flat_params['majority_k_pct'] = 0.55  # 預設 55% 門檻
                flat_params.pop('majority_k', None)  # 移除固定 K 值
                logger.info(f"[Ensemble] 使用比例門檻 majority_k_pct={flat_params['majority_k_pct']}")
            
            # === 修復 2：直接調用 SSS_EnsembleTab，避免 streamlit 衝突 ===
            try:
                import SSS_EnsembleTab as ens
                logger.info(f"[Ensemble] using {ens.__file__}")
                from SSS_EnsembleTab import run_ensemble, EnsembleParams, CostParams, RunConfig
                
                # 創建配置
                ensemble_params = EnsembleParams(
                    floor=flat_params.get("floor", 0.2),
                    ema_span=flat_params.get("ema_span", 3),
                    delta_cap=flat_params.get("delta_cap", 0.3),
                    majority_k=flat_params.get("majority_k", 6),
                    min_cooldown_days=flat_params.get("min_cooldown_days", 1),  # 與param_presets一致
                    min_trade_dw=flat_params.get("min_trade_dw", 0.01)          # 與param_presets一致
                )
                
                cost_params = CostParams(
                    buy_fee_bp=flat_params.get("buy_fee_bp", 4.27),  # 與param_presets一致
                    sell_fee_bp=flat_params.get("sell_fee_bp", 4.27), # 與param_presets一致
                    sell_tax_bp=flat_params.get("sell_tax_bp", 30.0)  # 與param_presets一致
                )
                
                cfg = RunConfig(
                    ticker=ticker,
                    method=flat_params.get("method", "majority"),
                    params=ensemble_params,
                    cost=cost_params
                )
                
                # === 第3步：統一路徑與preset，確保app與SSS使用相同的trades來源 ===
                # 傳遞比例門檻參數
                if flat_params.get("majority_k_pct"):
                    cfg.majority_k_pct = flat_params.get("majority_k_pct")
                else:
                    # 如果沒有majority_k_pct，強制使用0.55
                    cfg.majority_k_pct = 0.55
                    logger.info(f"[Ensemble] 強制設定 majority_k_pct=0.55")
                
                logger.info(f"[Ensemble] 執行配置: ticker={ticker}, method={flat_params.get('method')}, majority_k_pct={flat_params.get('majority_k_pct', 'N/A')}")
                
                # 運行 ensemble 策略
                open_px, w, trades, stats, method_name, equity, daily_state, trade_ledger = run_ensemble(cfg)
                
                # 直接使用 trade_ledger，不再手工构造简化的 trade_records
                if not trade_ledger.empty:
                    # 将 trade_ledger 转换为 app 需要的格式
                    trade_records = []
                    for _, row in trade_ledger.iterrows():
                        trade_records.append({
                            'signal_date': row.name,  # 使用索引作为日期
                            'trade_date': row.name,
                            'type': row['side'],  # 使用 side 字段
                            'price': row['open'],  # 使用 open 价格
                            'shares': row['units'],  # 使用实际交易单位
                            'return': row.get('return', 0.0),  # 如果有 return 字段
                            'exec_notional': row['exec_notional'],  # 实际交易金额
                            'fees_total': row['fees_total'],  # 总费用
                            'cash_after': row['cash_after'],  # 交易后现金
                            'equity_after': row['equity_after']  # 交易后总资产
                        })
                else:
                    trade_records = []
                
                trade_df = pd.DataFrame(trade_records)
                trades_df = trade_ledger if not trade_ledger.empty else pd.DataFrame()
                signals_df = daily_state if daily_state is not None else pd.DataFrame()
                
                # 构造指标，使用 trade_ledger 的实际交易次数
                metrics = {
                    'total_return': stats.get('total_return', 0.0),
                    'annual_return': stats.get('annual_return', 0.0),
                    'max_drawdown': stats.get('max_drawdown', 0.0),
                    'sharpe_ratio': stats.get('sharpe_ratio', 0.0),
                    'calmar_ratio': stats.get('calmar_ratio', 0.0),
                    'num_trades': len(trade_ledger) if not trade_ledger.empty else 0
                }
                
                # 添加投资组合状态信息
                if daily_state is not None and not daily_state.empty:
                    latest_state = daily_state.iloc[-1]
                    metrics.update({
                        'current_weight': latest_state['w'],
                        'cash_percentage': latest_state['cash_pct'],
                        'invested_percentage': latest_state['invested_pct'],
                        'current_cash': latest_state['cash'],
                        'position_value': latest_state['position_value'],
                        'total_equity': latest_state['equity']
                    })
                
                result = {
                    'trades': [],
                    'trade_df': trade_df,
                    'trades_df': trades_df,
                    'signals_df': signals_df,
                    'metrics': metrics,
                    'equity_curve': equity,
                    'daily_state': daily_state,
                    'trade_ledger': trade_ledger
                }
                
                logger.info(f"[Ensemble] 執行成功: {method_name}, 權益曲線長度={len(equity)}, 交易數={len(trade_ledger) if not trade_ledger.empty else 0}")
                
            except Exception as e:
                logger.error(f"Ensemble 策略執行失敗: {e}")
                # 創建空的結果
                result = {
                    'trades': [],
                    'trade_df': pd.DataFrame(),
                    'trades_df': pd.DataFrame(),
                    'signals_df': pd.DataFrame(),
                    'metrics': {'total_return': 0.0, 'annual_return': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0, 'calmar_ratio': 0.0, 'num_trades': 0},
                    'equity_curve': pd.Series(1.0, index=df_ind.index)
                }
            
            # === 修復 3：添加調試日誌，核對子策略集合是否一致 ===
            logger.info(f"[Ensemble] 執行完成，ticker={ticker}, method={flat_params.get('method')}")
            if 'equity_curve' in result and hasattr(result['equity_curve'], 'shape'):
                logger.info(f"[Ensemble] 權益曲線長度: {len(result['equity_curve'])}")
            if 'trade_df' in result and hasattr(result['trade_df'], 'shape'):
                logger.info(f"[Ensemble] 交易記錄數量: {len(result['trade_df'])}")
        else:
            if strat_type == 'single':
                df_ind = compute_single(df_raw, df_factor, strat_params["linlen"], strat_params["factor"], strat_params["smaalen"], strat_params["devwin"], smaa_source=smaa_src)
            elif strat_type == 'dual':
                df_ind = compute_dual(df_raw, df_factor, strat_params["linlen"], strat_params["factor"], strat_params["smaalen"], strat_params["short_win"], strat_params["long_win"], smaa_source=smaa_src)
            elif strat_type == 'RMA':
                df_ind = compute_RMA(df_raw, df_factor, strat_params["linlen"], strat_params["factor"], strat_params["smaalen"], strat_params["rma_len"], strat_params["dev_len"], smaa_source=smaa_src)
            if df_ind.empty:
                continue
            result = backtest_unified(df_ind, strat_type, strat_params, discount=discount, trade_cooldown_bars=cooldown, bad_holding=bad_holding)
        result['trade_df'] = result['trade_df'].to_json(date_format='iso')
        if 'signals_df' in result and isinstance(result['signals_df'], pd.DataFrame):
            result['signals_df'] = result['signals_df'].to_json(date_format='iso')
        if 'trades_df' in result and isinstance(result['trades_df'], pd.DataFrame):
            result['trades_df'] = result['trades_df'].to_json(date_format='iso')
        if 'equity_curve' in result and isinstance(result['equity_curve'], pd.Series):
            result['equity_curve'] = result['equity_curve'].to_json(date_format='iso')
        if 'trades' in result and isinstance(result['trades'], list):
            result['trades'] = [
                (str(t[0]), t[1], str(t[2])) if isinstance(t, tuple) and len(t) == 3 else t
                for t in result['trades']
            ]
        results[strat] = result
    
    # 使用第一個策略的數據作為主要顯示數據
    first_strat = list(results.keys())[0] if results else strategy_names[0]
    first_smaa_src = param_presets[first_strat].get("smaa_source", "Self")
    df_raw_main, _ = load_data(ticker, start_date, end_date if end_date else None, smaa_source=first_smaa_src)
    df_raw_json = df_raw_main.to_json(date_format='iso')
    return json.dumps({'results': results, 'df_raw': df_raw_json, 'ticker': ticker})

# --------- 主頁籤內容顯示 ---------
@app.callback(
    Output('tab-content', 'children'),
    Input('backtest-store', 'data'),
    Input('main-tabs', 'value'),
    State('strategy-dropdown', 'value'),
    Input('theme-store', 'data')
)
def update_tab(data, tab, selected_strategy, theme):
    if not data:
        return html.Div("請先執行回測")
    data = json.loads(data)
    results = data['results']
    df_raw = pd.read_json(io.StringIO(data['df_raw']))
    ticker = data['ticker']
    strategy_names = list(results.keys())
    # 根據主題決定 plotly template 與顏色
    if theme == 'theme-dark':
        plotly_template = 'plotly_dark'
        font_color = '#fff'
        bg_color = '#181818'
        legend_font_color = '#fff'
        legend_bgcolor = '#232323'
        legend_bordercolor = '#fff'
    elif theme == 'theme-light':
        plotly_template = 'plotly_white'
        font_color = '#111'
        bg_color = '#fff'
        legend_font_color = '#111'
        legend_bgcolor = '#fff'
        legend_bordercolor = '#111'
    else:  # theme-blue
        plotly_template = 'plotly_white'
        font_color = '#ffe066'
        bg_color = '#003366'
        legend_font_color = '#ffe066'
        legend_bgcolor = '#002244'
        legend_bordercolor = '#ffe066'
    
    if tab == "backtest":
        # 創建策略回測的子頁籤
        strategy_tabs = []
        for strategy in strategy_names:
            result = results.get(strategy)
            if not result:
                continue
            
            trade_df = pd.read_json(io.StringIO(result['trade_df']))
            # 型別對齊：保證 trade_date 為 Timestamp，price/shares 為 float
            if 'trade_date' in trade_df.columns:
                trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
            if 'signal_date' in trade_df.columns:
                trade_df['signal_date'] = pd.to_datetime(trade_df['signal_date'])
            if 'price' in trade_df.columns:
                trade_df['price'] = pd.to_numeric(trade_df['price'], errors='coerce')
            if 'shares' in trade_df.columns:
                trade_df['shares'] = pd.to_numeric(trade_df['shares'], errors='coerce')
            
            # 顯示時只顯示日期
            display_df = trade_df.copy()
            if 'trade_date' in display_df.columns:
                display_df['trade_date'] = display_df['trade_date'].dt.date
            if 'signal_date' in display_df.columns:
                display_df['signal_date'] = display_df['signal_date'].dt.date
            if 'price' in display_df.columns:
                display_df['price'] = display_df['price'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            if 'return' in display_df.columns:
                display_df['return'] = display_df['return'].apply(lambda x: "-" if pd.isna(x) else f"{x:.2%}")
            
            metrics = result.get('metrics', {})
            tooltip = f"{strategy} 策略説明"
            param_display = {k: v for k, v in param_presets[strategy].items() if k != "strategy_type"}
            param_str = ", ".join(f"{k}: {v}" for k, v in param_display.items())
            avg_holding = calculate_holding_periods(trade_df)
            metrics['avg_holding_period'] = avg_holding
            
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
            
            metric_cards = []
            for k, v in metrics.items():
                if k in ["total_return", "annual_return", "win_rate", "max_drawdown", "annualized_volatility", "avg_win", "avg_loss"]:
                    txt = f"{v:.2%}" if pd.notna(v) else ""
                elif k in ["calmar_ratio", "sharpe_ratio", "sortino_ratio", "payoff_ratio", "profit_factor"]:
                    txt = f"{v:.2f}" if pd.notna(v) else ""
                elif k in ["max_drawdown_duration", "avg_holding_period"]:
                    txt = f"{v:.2f} 天" if pd.notna(v) else ""
                elif k in ["num_trades", "max_consecutive_wins", "max_consecutive_losses"]:
                    txt = f"{v:.0f}" if pd.notna(v) else ""
                else:
                    txt = f"{v:.2f}" if isinstance(v, (float, int)) and pd.notna(v) else f"{v}"
                metric_cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(label_map.get(k, k), className="card-title-label"),
                                html.Div(txt, style={"font-weight": "bold", "font-size": "20px"})
                            ])
                        ], style={"background": "#1a1a1a", "border": "1px solid #444", "border-radius": "8px", "margin-bottom": "6px"})
                    ], xs=12, sm=6, md=4, lg=2, xl=2, style={"minWidth": "100px", "margin-bottom": "6px", "maxWidth": "12.5%"})
                )
            
            fig1 = plot_stock_price(df_raw, trade_df, ticker)
            fig1.update_layout(
                template=plotly_template, font_color=font_color, plot_bgcolor=bg_color, paper_bgcolor=bg_color,
                legend_font_color=legend_font_color,
                legend=dict(bgcolor=legend_bgcolor, bordercolor=legend_bordercolor, font=dict(color=legend_font_color))
            )
            fig2 = plot_equity_cash(trade_df, df_raw)
            fig2.update_layout(
                template=plotly_template, font_color=font_color, plot_bgcolor=bg_color, paper_bgcolor=bg_color,
                legend_font_color=legend_font_color,
                legend=dict(bgcolor=legend_bgcolor, bordercolor=legend_bordercolor, font=dict(color=legend_font_color))
            )
            
            strategy_content = html.Div([
                html.H4([
                    f"回測策略: {strategy} ",
                    html.Span("ⓘ", title=tooltip, style={"cursor": "help", "color": "#888"})
                ]),
                html.Div(f"參數設定: {param_str}"),
                html.Br(),
                dbc.Row(metric_cards, style={"flex-wrap": "wrap"}, className='metrics-cards-row'),
                html.Br(),
                dcc.Graph(figure=fig1, config={'displayModeBar': True}, className='main-metrics-graph'),
                dcc.Graph(figure=fig2, config={'displayModeBar': True}, className='main-cash-graph'),
                # 將交易明細標題與説明合併為同一行
                html.Div([
                    html.H5("交易明細", style={"marginBottom": 0, "marginRight": "12px"}),
                    html.Div("實際下單日為信號日的隔天（S+1），修改代碼會影響很多層面，暫不修改", 
                             style={"fontWeight": "bold", "fontSize": "16px"})
                ], style={"display": "flex", "alignItems": "center", "marginTop": "16px"}),
                dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in display_df.columns],
                    data=display_df.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center'},
                    id={'type': 'strategy-table', 'strategy': strategy}
                ),
                
                html.Br(),
                html.Button("下載交易紀錄", id={'type': 'download-btn', 'strategy': strategy}),
                dcc.Download(id={'type': 'download-trade', 'strategy': strategy})
            ])
            
            strategy_tabs.append(dcc.Tab(label=strategy, value=f"strategy_{strategy}", children=strategy_content))
        
        # 創建策略回測的子頁籤容器
        return html.Div([
        

            dcc.Tabs(
                id='strategy-tabs',
                value=f"strategy_{strategy_names[0]}" if strategy_names else "no_strategy",
                children=strategy_tabs,
                className='strategy-tabs-bar'
            )
        ])
        
    elif tab == "compare":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['close'], name='Close Price', line=dict(color='dodgerblue')))
        colors = ['green', 'limegreen', 'red', 'orange', 'purple', 'blue', 'pink', 'cyan']
        
        # 定義策略到顏色的映射
        strategy_colors = {strategy: colors[i % len(colors)] for i, strategy in enumerate(strategy_names)}
        
        # 為圖表添加買賣點
        for i, strategy in enumerate(strategy_names):
            result = results.get(strategy)
            if not result:
                continue
            trade_df = pd.read_json(io.StringIO(result['trade_df']))
            if 'trade_date' in trade_df.columns:
                trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
            if trade_df.empty:
                continue
            buys = trade_df[trade_df['type'] == 'buy']
            sells = trade_df[trade_df['type'] == 'sell']
            fig.add_trace(go.Scatter(x=buys['trade_date'], y=buys['price'], mode='markers', name=f'{strategy} Buy',
                                     marker=dict(symbol='cross', size=8, color=colors[i % len(colors)])))
            fig.add_trace(go.Scatter(x=sells['trade_date'], y=sells['price'], mode='markers', name=f'{strategy} Sell',
                                     marker=dict(symbol='x', size=8, color=colors[i % len(colors)])))
        
        # 更新圖表佈局
        fig.update_layout(
            title=f'{ticker} 所有策略買賣點比較',
            xaxis_title='Date', yaxis_title='股價', template=plotly_template,
            font_color=font_color, plot_bgcolor=bg_color, paper_bgcolor=bg_color, legend_font_color=legend_font_color,
            legend=dict(
                x=1.05, y=1, xanchor='left', yanchor='top',
                bordercolor=legend_bordercolor, borderwidth=1, bgcolor=legend_bgcolor,
                itemsizing='constant', orientation='v', font=dict(color=legend_font_color)
            )
        )
        
        # 準備比較表格數據
        comparison_data = []
        for strategy in strategy_names:
            result = results.get(strategy)
            if not result:
                continue
            
            # 讀取交易數據
            trade_df = pd.read_json(io.StringIO(result['trade_df']))
            if 'trade_date' in trade_df.columns:
                trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
            
            # 計算詳細統計信息
            detailed_stats = calculate_strategy_detailed_stats(trade_df, df_raw)
            
            metrics = result['metrics']
            comparison_data.append({
                '策略': strategy,
                '總回報率': f"{metrics.get('total_return', 0):.2%}",
                '年化回報率': f"{metrics.get('annual_return', 0):.2%}",
                '最大回撤': f"{metrics.get('max_drawdown', 0):.2%}",
                '卡瑪比率': f"{metrics.get('calmar_ratio', 0):.2f}",
                '交易次數': metrics.get('num_trades', 0),
                '勝率': f"{metrics.get('win_rate', 0):.2%}",
                '盈虧比': f"{metrics.get('payoff_ratio', 0):.2f}",
                '平均持有天數': f"{detailed_stats['avg_holding_days']:.1f}",
                '賣後買平均天數': f"{detailed_stats['avg_sell_to_buy_days']:.1f}",
                '目前狀態': detailed_stats['current_status'],
                '距離上次操作天數': f"{detailed_stats['days_since_last_action']}"
            })
        
        # 定義顏色調整函數
        def adjust_color_for_theme(color, theme):
            # 預定義顏色到 RGB 的映射
            color_to_rgb = {
                'green': '0, 128, 0',
                'limegreen': '50, 205, 50', 
                'red': '255, 0, 0',
                'orange': '255, 165, 0',
                'purple': '128, 0, 128',
                'blue': '0, 0, 255',
                'pink': '255, 192, 203',
                'cyan': '0, 255, 255'
            }
            
            rgb = color_to_rgb.get(color, '128, 128, 128')  # 默認灰色
            
            if theme == 'theme-dark':
                return f'rgba({rgb}, 0.2)'  # 透明度 0.2
            elif theme == 'theme-light':
                return f'rgba({rgb}, 1)'    # 透明度 1
            else:  # theme-blue
                return f'rgba({rgb}, 0.5)'  # 透明度 0.5
        
        # 創建比較表格並應用條件樣式
        compare_table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in comparison_data[0].keys()] if comparison_data else [],
            data=comparison_data,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_data_conditional=[
                {
                    'if': {'row_index': i},
                    'backgroundColor': adjust_color_for_theme(strategy_colors[row['策略']], theme),
                    'border': f'1px solid {strategy_colors[row['策略']]}'
                } for i, row in enumerate(comparison_data)
            ],
            id='compare-table'
        )
        
        return html.Div([
            dcc.Graph(figure=fig, config={'displayModeBar': True}, className='compare-graph'),
            html.Hr(),
            compare_table
        ])

# --------- 版本沿革模態框控制和主題切換 ---------
@app.callback(
    Output("history-modal", "is_open"),
    Output('main-bg', 'className'),
    Output('theme-toggle', 'children'),
    Output('theme-store', 'data'),
    [Input("history-btn", "n_clicks"), Input("history-close", "n_clicks"), Input('theme-toggle', 'n_clicks')],
    [State("history-modal", "is_open"), State('theme-store', 'data')],
    prevent_initial_call=True
)
def toggle_history_modal_and_theme(history_btn, history_close, theme_btn, is_open, current_theme):
    ctx_trigger = ctx.triggered_id
    
    if ctx_trigger == "history-btn":
        # 開啟版本沿革模態框，強制切換到深色主題
        return True, 'theme-dark', '🌑 深色主題', 'theme-dark'
    elif ctx_trigger == "history-close":
        # 關閉版本沿革模態框，恢復原主題
        return False, current_theme, get_theme_label(current_theme), current_theme
    elif ctx_trigger == "theme-toggle":
        # 正常的主題切換
        if current_theme is None:
            return is_open, 'theme-dark', '🌑 深色主題', 'theme-dark'
        themes = ['theme-dark', 'theme-light', 'theme-blue']
        current_index = themes.index(current_theme)
        next_theme = themes[(current_index + 1) % len(themes)]
        return is_open, next_theme, get_theme_label(next_theme), next_theme
    
    return is_open, current_theme, get_theme_label(current_theme), current_theme

# --------- 下載交易紀錄 ---------
@app.callback(
    Output({'type': 'download-trade', 'strategy': ALL}, 'data'),
    Input({'type': 'download-btn', 'strategy': ALL}, 'n_clicks'),
    State({'type': 'strategy-table', 'strategy': ALL}, 'data'),
    State('backtest-store', 'data'),
    prevent_initial_call=True
)
def download_trade(n_clicks, table_data, backtest_data):
    ctx_trigger = ctx.triggered_id
    if not ctx_trigger or not backtest_data:
        return [None] * len(n_clicks)
    
    # 從觸發的按鈕ID中提取策略名稱
    strategy = ctx_trigger['strategy']
    
    # 從backtest_data中獲取對應策略的交易數據
    data = json.loads(backtest_data)
    results = data['results']
    result = results.get(strategy)
    
    if not result:
        return [None] * len(n_clicks)
    
    trade_df = pd.read_json(io.StringIO(result['trade_df']))
    
    # 創建下載數據
    def to_xlsx(bytes_io):
        with pd.ExcelWriter(bytes_io, engine='openpyxl') as writer:
            trade_df.to_excel(writer, sheet_name='交易紀錄', index=False)
    
    return [dcc.send_bytes(to_xlsx, f"{strategy}_交易紀錄.xlsx") if i > 0 else None for i in n_clicks]

def calculate_strategy_detailed_stats(trade_df, df_raw):
    """計算策略的詳細統計信息"""
    if trade_df.empty:
        return {
            'avg_holding_days': 0,
            'avg_sell_to_buy_days': 0,
            'current_status': '未持有',
            'days_since_last_action': 0
        }
    
    # 確保日期列是 datetime 類型
    if 'trade_date' in trade_df.columns:
        trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
    
    # 按日期排序確保順序正確
    trade_df = trade_df.sort_values('trade_date').reset_index(drop=True)
    
    # 計算平均持有天數（買入到賣出的天數）
    holding_periods = []
    for i in range(len(trade_df) - 1):
        current_type = trade_df.iloc[i]['type']
        next_type = trade_df.iloc[i+1]['type']
        if current_type == 'buy' and next_type in ['sell', 'sell_forced']:
            buy_date = trade_df.iloc[i]['trade_date']
            sell_date = trade_df.iloc[i+1]['trade_date']
            holding_days = (sell_date - buy_date).days
            holding_periods.append(holding_days)
    avg_holding_days = sum(holding_periods) / len(holding_periods) if holding_periods else 0
    
    # 計算賣後買平均天數（賣出到下次買入的天數）
    sell_to_buy_periods = []
    for i in range(len(trade_df) - 1):
        current_type = trade_df.iloc[i]['type']
        next_type = trade_df.iloc[i+1]['type']
        if current_type in ['sell', 'sell_forced'] and next_type == 'buy':
            sell_date = trade_df.iloc[i]['trade_date']
            buy_date = trade_df.iloc[i+1]['trade_date']
            days_between = (buy_date - sell_date).days
            sell_to_buy_periods.append(days_between)
    avg_sell_to_buy_days = sum(sell_to_buy_periods) / len(sell_to_buy_periods) if sell_to_buy_periods else 0
    
    # 取得最後一筆操作
    last_trade = trade_df.iloc[-1] if not trade_df.empty else None
    if not df_raw.empty:
        current_date = df_raw.index[-1]
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
    else:
        current_date = datetime.now()
    
    if last_trade is not None:
        last_type = last_trade['type']
        last_date = last_trade['trade_date']
        if last_type == 'buy':
            current_status = '持有'
            days_since_last_action = (current_date - last_date).days
        elif last_type == 'sell':
            current_status = '未持有'
            days_since_last_action = (current_date - last_date).days
        elif last_type == 'sell_forced':
            # 狀態為持有，天數為目前日期減去最近一筆 buy 日期
            current_status = '持有'
            # 往前找最近一筆 buy
            last_buy = trade_df[trade_df['type'] == 'buy']
            if not last_buy.empty:
                last_buy_date = last_buy.iloc[-1]['trade_date']
                days_since_last_action = (current_date - last_buy_date).days
            else:
                days_since_last_action = 0
        else:
            current_status = '未持有'
            days_since_last_action = (current_date - last_date).days
    else:
        current_status = '未持有'
        days_since_last_action = 0
    
    return {
        'avg_holding_days': round(avg_holding_days, 1),
        'avg_sell_to_buy_days': round(avg_sell_to_buy_days, 1),
        'current_status': current_status,
        'days_since_last_action': days_since_last_action
    }

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

if __name__ == '__main__':
    # 在主線程中執行啟動任務
    safe_startup()
    
    # 設置更安全的服務器配置
    app.run_server(
        debug=True, 
        host='127.0.0.1', 
        port=8050,
        threaded=True,
        use_reloader=False  # 避免重載器造成的線程問題
    ) 