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

# 配置 logger - 使用新的顯式初始化
from analysis.logging_config import init_logging
init_logging()  # 預設只開 console；要落地檔案就設 SSS_CREATE_LOGS=1
logger = logging.getLogger("SSS.App")

# 解包器函數：支援 pack_df/pack_series 和傳統 JSON 字串兩種格式
def df_from_pack(data):
    """從 pack_df 結果或 JSON 字串解包 DataFrame"""
    import io, json
    import pandas as pd
    if data is None or data == "" or data == "[]":
        return pd.DataFrame()
    if isinstance(data, str):
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
    if data is None or data == "" or data == "[]":
        return pd.Series(dtype=float)
    if isinstance(data, str):
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
        'weight_curve'
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
            # 使用新的 ensemble_runner 避免循環依賴
            try:
                from runners.ensemble_runner import run_ensemble_backtest
                from SSS_EnsembleTab import EnsembleParams, CostParams, RunConfig
                
                # 把 SSSv096 的巢狀參數攤平
                flat_params = {}
                flat_params.update(strat_params.get('params', {}))
                flat_params.update(strat_params.get('trade_cost', {}))
                flat_params['method'] = strat_params.get('method', 'majority')
                flat_params['ticker'] = ticker
                
                # 使用比例門檻避免 N 變動時失真
                if 'majority_k' in flat_params and flat_params.get('method') == 'majority':
                    flat_params['majority_k_pct'] = 0.55
                    flat_params.pop('majority_k', None)
                    logger.info(f"[Ensemble] 使用比例門檻 majority_k_pct={flat_params['majority_k_pct']}")
                
                # 創建配置
                ensemble_params = EnsembleParams(
                    floor=flat_params.get("floor", 0.2),
                    ema_span=flat_params.get("ema_span", 3),
                    delta_cap=flat_params.get("delta_cap", 0.3),
                    majority_k=flat_params.get("majority_k", 6),
                    min_cooldown_days=flat_params.get("min_cooldown_days", 1),
                    min_trade_dw=flat_params.get("min_trade_dw", 0.01)
                )
                
                cost_params = CostParams(
                    buy_fee_bp=flat_params.get("buy_fee_bp", 4.27),
                    sell_fee_bp=flat_params.get("sell_fee_bp", 4.27),
                    sell_tax_bp=flat_params.get("sell_tax_bp", 30.0)
                )
                
                cfg = RunConfig(
                    ticker=ticker,
                    method=flat_params.get("method", "majority"),
                    params=ensemble_params,
                    cost=cost_params
                )
                
                # 傳遞比例門檻參數
                if flat_params.get("majority_k_pct"):
                    cfg.majority_k_pct = flat_params.get("majority_k_pct")
                else:
                    cfg.majority_k_pct = 0.55
                    logger.info(f"[Ensemble] 強制設定 majority_k_pct=0.55")
                
                logger.info(f"[Ensemble] 執行配置: ticker={ticker}, method={flat_params.get('method')}, majority_k_pct={flat_params.get('majority_k_pct', 'N/A')}")
                
                # 使用新的 ensemble_runner 執行
                backtest_result = run_ensemble_backtest(cfg)
                
                # 轉換為舊格式以保持相容性
                result = {
                    'trades': [],
                    'trade_df': pack_df(backtest_result.trades),
                    'trades_df': pack_df(backtest_result.trades),
                    'signals_df': pack_df(backtest_result.trades[['trade_date', 'type', 'price']].rename(columns={'type': 'action'}) if not backtest_result.trades.empty else pd.DataFrame(columns=['trade_date', 'action', 'price'])),
                    'metrics': backtest_result.stats,
                    'equity_curve': pack_series(backtest_result.equity_curve),
                    'cash_curve': pack_series(backtest_result.cash_curve) if backtest_result.cash_curve is not None else "",
                    'weight_curve': pack_series(backtest_result.weight_curve) if backtest_result.weight_curve is not None else pack_series(pd.Series(0.0, index=backtest_result.equity_curve.index)),
                    'price_series': pack_series(backtest_result.price_series) if backtest_result.price_series is not None else pack_series(pd.Series(1.0, index=backtest_result.equity_curve.index)),
                    'daily_state': pack_df(backtest_result.daily_state),
                    'trade_ledger': pack_df(backtest_result.ledger),
                    'daily_state_std': pack_df(backtest_result.daily_state),
                    'trade_ledger_std': pack_df(backtest_result.ledger)
                }
                
                logger.info(f"[Ensemble] 執行成功: 權益曲線長度={len(backtest_result.equity_curve)}, 交易數={len(backtest_result.ledger) if backtest_result.ledger is not None and not backtest_result.ledger.empty else 0}")
                
            except Exception as e:
                logger.error(f"Ensemble 策略執行失敗: {e}")
                # 創建空的結果
                result = {
                    'trades': [],
                    'trade_df': pd.DataFrame(),
                    'trades_df': pd.DataFrame(),
                    'signals_df': pd.DataFrame(),
                    'metrics': {'total_return': 0.0, 'annual_return': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0, 'calmar_ratio': 0.0, 'num_trades': 0},
                    'equity_curve': pd.Series(1.0, index=df_raw.index)
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
        # 統一使用 orient="split" 打包，避免重複序列化
        # 注意：Ensemble 策略已經在 pack_df/pack_series 中處理過，這裡只處理單策略
        if strat_type != 'ensemble':
            if hasattr(result.get('trade_df'), 'to_json'):
                result['trade_df'] = result['trade_df'].to_json(date_format='iso', orient='split')
            if 'signals_df' in result and hasattr(result['signals_df'], 'to_json'):
                result['signals_df'] = result['signals_df'].to_json(date_format='iso', orient='split')
            if 'trades_df' in result and hasattr(result['trades_df'], 'to_json'):
                result['trades_df'] = result['trades_df'].to_json(date_format='iso', orient='split')
            if 'equity_curve' in result and hasattr(result['equity_curve'], 'to_json'):
                result['equity_curve'] = result['equity_curve'].to_json(date_format='iso', orient='split')
        if 'trades' in result and isinstance(result['trades'], list):
            result['trades'] = [
                (str(t[0]), t[1], str(t[2])) if isinstance(t, tuple) and len(t) == 3 else t
                for t in result['trades']
            ]
        
        # << 新增：一律做最後保險打包，補上 daily_state / weight_curve 等 >>
        result = _pack_result_for_store(result)
        
        results[strat] = result
    
    # 使用第一個策略的數據作為主要顯示數據
    first_strat = list(results.keys())[0] if results else strategy_names[0]
    first_smaa_src = param_presets[first_strat].get("smaa_source", "Self")
    df_raw_main, _ = load_data(ticker, start_date, end_date if end_date else None, smaa_source=first_smaa_src)
    
    # 統一使用 orient="split" 序列化，確保一致性
    payload = {
        'results': results, 
        'df_raw': df_raw_main.to_json(date_format='iso', orient='split'), 
        'ticker': ticker
    }
    
    # 防守性檢查：如還有漏網的非序列化物件就能提早看出
    try:
        json.dumps(payload)
    except Exception as e:
        logger.exception("[BUG] backtest-store payload 仍含不可序列化物件：%s", e)
        # 如果要強制不噴，可做 fallback：json.dumps(..., default=str) 但通常不建議吞掉
    
    return payload

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
    # data 現在已經是 dict，不需要 json.loads
    results = data['results']
    df_raw = df_from_pack(data['df_raw'])  # 使用 df_from_pack 統一解包
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
            
            # 先解包（放在決定 base_df 之前）
            daily_state_std = df_from_pack(result.get('daily_state_std'))
            trade_ledger_std = df_from_pack(result.get('trade_ledger_std'))
            
            # 使用解包器函數，支援 pack_df 和傳統 JSON 字串兩種格式
            trade_df = df_from_pack(result.get('trade_df'))
            
            # 標準化交易資料，確保有統一的 trade_date/type/price 欄位
            try:
                from sss_core.normalize import normalize_trades_for_ui as norm
                trade_df = norm(trade_df)
                logger.info(f"標準化後 trades_ui 欄位: {list(trade_df.columns)}")
            except Exception as e:
                logger.warning(f"無法使用 sss_core 標準化，使用後備方案: {e}")
                # 後備標準化方案
                if trade_df is not None and len(trade_df) > 0:
                    trade_df = trade_df.copy()
                    trade_df.columns = [str(c).lower() for c in trade_df.columns]
                    
                    # 確保有 trade_date 欄
                    if "trade_date" not in trade_df.columns:
                        if "date" in trade_df.columns:
                            trade_df["trade_date"] = pd.to_datetime(trade_df["date"], errors="coerce")
                        elif isinstance(trade_df.index, pd.DatetimeIndex):
                            trade_df = trade_df.reset_index().rename(columns={"index": "trade_date"})
                        else:
                            trade_df["trade_date"] = pd.NaT
                    else:
                        trade_df["trade_date"] = pd.to_datetime(trade_df["trade_date"], errors="coerce")
                    
                    # 確保有 type 欄
                    if "type" not in trade_df.columns:
                        if "action" in trade_df.columns:
                            trade_df["type"] = trade_df["action"].astype(str).str.lower()
                        elif "side" in trade_df.columns:
                            trade_df["type"] = trade_df["side"].astype(str).str.lower()
                        else:
                            trade_df["type"] = "hold"
                    
                    # 確保有 price 欄
                    if "price" not in trade_df.columns:
                        for c in ["open", "price_open", "exec_price", "px", "close"]:
                            if c in trade_df.columns:
                                trade_df["price"] = trade_df[c]
                                break
                        if "price" not in trade_df.columns:
                            trade_df["price"] = 0.0
            
            # 型別對齊：保證 trade_date 為 Timestamp，price/shares 為 float
            if 'trade_date' in trade_df.columns:
                trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
            if 'signal_date' in trade_df.columns:
                trade_df['signal_date'] = pd.to_datetime(trade_df['signal_date'])
            if 'price' in trade_df.columns:
                trade_df['price'] = pd.to_numeric(trade_df['price'], errors='coerce')
            if 'shares' in trade_df.columns:
                trade_df['shares'] = pd.to_numeric(trade_df['shares'], errors='coerce')
            
            # === 新：若有 trade_ledger，優先顯示更完整的欄位 ===
            ledger_df = df_from_pack(result.get('trade_ledger'))
            try:
                from sss_core.normalize import normalize_trades_for_ui as norm
                ledger_ui = norm(ledger_df) if ledger_df is not None and len(ledger_df)>0 else pd.DataFrame()
            except Exception:
                ledger_ui = ledger_df if ledger_df is not None else pd.DataFrame()
            
            # === 修正：優先使用標準化後的 trade_ledger_std ===
            # 使用 utils_payload 標準化後的結果，確保欄位齊全
            if trade_ledger_std is not None and not trade_ledger_std.empty:
                base_df = trade_ledger_std
                if 'trade_date' in base_df.columns:
                    base_df['trade_date'] = pd.to_datetime(base_df['trade_date'], errors='coerce')
                if 'signal_date' in base_df.columns:
                    base_df['signal_date'] = pd.to_datetime(base_df['signal_date'], errors='coerce')
            elif ledger_ui is not None and not ledger_ui.empty:
                base_df = ledger_ui
                if 'trade_date' in base_df.columns:
                    base_df['trade_date'] = pd.to_datetime(base_df['trade_date'], errors='coerce')
                if 'signal_date' in base_df.columns:
                    base_df['signal_date'] = pd.to_datetime(base_df['signal_date'], errors='coerce')
            else:
                base_df = trade_df
            
            # 記錄原始欄位（偵錯用）
            logger.info("[UI] trade_df 原始欄位：%s", list(base_df.columns) if base_df is not None else None)
            logger.info("[UI] trade_ledger_std 原始欄位：%s", list(trade_ledger_std.columns) if trade_ledger_std is not None else None)

            # 為了 100% 保證 weight_change 出現，先確保權重欄位
            base_df = _ensure_weight_columns(base_df)
            # 使用新的統一格式化函式
            display_df = format_trade_like_df_for_display(base_df)

            # === 交易流水帳(ledger)表格：先準備顯示版 ===
            ledger_src = trade_ledger_std if (trade_ledger_std is not None and not trade_ledger_std.empty) else \
                         (ledger_ui if (ledger_ui is not None and not ledger_ui.empty) else pd.DataFrame())

            if ledger_src is not None and not ledger_src.empty:
                # 為了 100% 保證 weight_change 出現，先確保權重欄位
                ledger_src = _ensure_weight_columns(ledger_src)
                # 使用新的統一格式化函式
                ledger_display = format_trade_like_df_for_display(ledger_src)
                ledger_columns = [{"name": i, "id": i} for i in ledger_display.columns]
                ledger_data = ledger_display.to_dict('records')
            else:
                ledger_columns = []
                ledger_data = []
            
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
            # 檢查是否有 daily_state 可用（ensemble 策略）
            daily_state = df_from_pack(result.get('daily_state'))
            
            # 優先使用標準化後的資料，確保欄位完整
            if daily_state_std is not None and not daily_state_std.empty:
                daily_state_display = daily_state_std
                logger.info(f"[UI] 使用標準化後的 daily_state_std，欄位: {list(daily_state_std.columns)}")
            else:
                daily_state_display = daily_state
                logger.info(f"[UI] 使用原始 daily_state，欄位: {list(daily_state.columns) if daily_state is not None else None}")
            
            # 檢查點（快速自查）
            logger.info(f"[UI] daily_state_display cols={list(daily_state_display.columns) if daily_state_display is not None else None}")
            if daily_state_display is not None:
                logger.info(f"[UI] daily_state_display head=\n{daily_state_display[['equity','cash']].head(3) if 'equity' in daily_state_display.columns and 'cash' in daily_state.columns else 'Missing equity/cash columns'}")
            logger.info(f"[UI] trade_df cols={list(trade_df.columns)} head=\n{trade_df.head(3)}")
            
            # === 修正：實現 fallback 邏輯，讓單一策略也能顯示權益/現金 ===
            if daily_state_display is not None and not daily_state_display.empty and {'equity','cash'}.issubset(daily_state_display.columns):
                # 正常：有 daily_state
                fig2 = plot_equity_cash(daily_state_display, df_raw)
                
                # === 新增：持有權重變化圖（統一背景色） ===
                fig_w = plot_weight_series(daily_state_display, trade_df)
                # 統一背景色為主題一致
                fig_w.update_layout(
                    template=plotly_template,
                    font_color=font_color,
                    plot_bgcolor=bg_color,
                    paper_bgcolor=bg_color,
                    legend=dict(bgcolor=legend_bgcolor, bordercolor=legend_bordercolor, font=dict(color=legend_font_color))
                )
                
                # === 新增：資金權重表格 ===
                # 使用標準化後的 daily_state_display（已經標準化過了）
                # 準備資金權重表格數據
                if not daily_state_display.empty:
                    # 選擇要顯示的欄位（與 Streamlit 一致）
                    display_cols = ['equity', 'cash', 'invested_pct', 'cash_pct', 'w', 'position_value']
                    available_cols = [col for col in display_cols if col in daily_state_display.columns]
                    
                    if available_cols:
                        # 格式化數據用於顯示
                        display_daily_state = daily_state_display[available_cols].copy()
                        display_daily_state.index = display_daily_state.index.strftime('%Y-%m-%d')
                        
                        # 格式化數值
                        for col in ['equity', 'cash', 'position_value']:
                            if col in display_daily_state.columns:
                                display_daily_state[col] = display_daily_state[col].apply(lambda x: f"{int(x):,}" if pd.notnull(x) and not pd.isna(x) else "")
                        
                        for col in ['invested_pct', 'cash_pct']:
                            if col in display_daily_state.columns:
                                display_daily_state[col] = display_daily_state[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
                        
                        for col in ['w']:
                            if col in display_daily_state.columns:
                                display_daily_state[col] = display_daily_state[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
                        
                        # 創建資金權重表格
                        daily_state_table = html.Div([
                            html.H5("資金權重", style={"marginTop": "16px"}),
                            html.Div("每日資產配置狀態，包含權益、現金、投資比例等", 
                                     style={"fontSize": "14px", "color": "#666", "marginBottom": "8px"}),
                            dash_table.DataTable(
                                columns=[{"name": i, "id": i} for i in display_daily_state.columns],
                                data=display_daily_state.head(20).to_dict('records'),  # 只顯示前20筆
                                style_table={'overflowX': 'auto', 'backgroundColor': '#1a1a1a'},
                                style_cell={'textAlign': 'center', 'backgroundColor': '#1a1a1a', 'color': '#fff', 'border': '1px solid #444'},
                                style_header={'backgroundColor': '#2a2a2a', 'color': '#fff', 'border': '1px solid #444'},
                                id={'type': 'daily-state-table', 'strategy': strategy}
                            ),
                            html.Div(f"顯示前20筆記錄，共{len(display_daily_state)}筆", 
                                     style={"fontSize": "12px", "color": "#888", "textAlign": "center", "marginTop": "8px"})
                        ])
                    else:
                        daily_state_table = html.Div("資金權重資料不足", style={"color": "#888", "fontStyle": "italic"})
                else:
                    daily_state_table = html.Div("資金權重資料為空", style={"color": "#888", "fontStyle": "italic"})
            else:
                # 回退：沒有 daily_state，就用交易表重建（單一策略會回來）
                logger.info("[UI] 使用 fallback：從交易表重建權益/現金曲線")
                fig2 = plot_equity_cash(trade_df, df_raw)  # 使用 SSSv096 的 fallback 邏輯
                
                # 權重圖：若 DS 有 w 就畫，否則先給空圖
                if daily_state_display is not None and not daily_state_display.empty and 'w' in daily_state_display.columns:
                    fig_w = plot_weight_series(daily_state_display['w'], title="持有權重變化")
                    fig_w.update_layout(
                        template=plotly_template,
                        font_color=font_color,
                        plot_bgcolor=bg_color,
                        paper_bgcolor=bg_color,
                        legend=dict(bgcolor=legend_bgcolor, bordercolor=legend_bordercolor, font=dict(color=legend_font_color))
                    )
                else:
                    fig_w = go.Figure()  # 先給空圖
                
                daily_state_table = html.Div("使用交易表重建的權益/現金曲線", style={"color": "#888", "fontStyle": "italic"})
            
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
                # === 新增：持有權重變化圖 ===
                dcc.Graph(figure=fig_w, config={'displayModeBar': True}, className='main-weight-graph'),
                # 將交易明細標題與説明合併為同一行
                html.Div([
                    html.H5("交易明細", style={"marginBottom": 0, "marginRight": "12px"}),
                    html.Div("實際下單日為信號日的隔天（S+1），修改代碼會影響很多層面，暫不修改", 
                             style={"fontWeight": "bold", "fontSize": "16px"})
                ], style={"display": "flex", "alignItems": "center", "marginTop": "16px"}),
                
                dash_table.DataTable(
                    columns=[{"name": get_column_display_name(i), "id": i} for i in display_df.columns],
                    data=display_df.to_dict('records'),
                    style_table={'overflowX': 'auto', 'backgroundColor': '#1a1a1a'},
                    style_cell={'textAlign': 'center', 'backgroundColor': '#1a1a1a', 'color': '#fff', 'border': '1px solid #444'},
                    style_header={'backgroundColor': '#2a2a2a', 'color': '#fff', 'border': '1px solid #444'},
                    id={'type': 'strategy-table', 'strategy': strategy}
                ),
                
                # === 新增：交易明細 CSV 下載按鈕 ===
                html.Div([
                    html.Button(
                        "📥 下載交易明細 CSV",
                        id={'type': 'download-trade-details-csv', 'strategy': strategy},
                        style={
                            'backgroundColor': '#28a745',
                            'color': 'white',
                            'border': 'none',
                            'padding': '8px 16px',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'marginTop': '8px',
                            'fontSize': '14px'
                        }
                    ),
                    dcc.Download(id={'type': 'download-trade-details-data', 'strategy': strategy})
                ], style={'textAlign': 'center', 'marginTop': '8px'}),
                

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
            # 使用解包器函數，支援 pack_df 和傳統 JSON 字串兩種格式
            trade_df = df_from_pack(result.get('trade_df'))
            
            # 標準化交易資料
            try:
                from sss_core.normalize import normalize_trades_for_ui as norm
                trade_df = norm(trade_df)
            except Exception:
                # 後備標準化方案
                if trade_df is not None and len(trade_df) > 0:
                    trade_df = trade_df.copy()
                    trade_df.columns = [str(c).lower() for c in trade_df.columns]
                    if "trade_date" not in trade_df.columns and "date" in trade_df.columns:
                        trade_df["trade_date"] = pd.to_datetime(trade_df["date"], errors="coerce")
                    if "type" not in trade_df.columns and "action" in trade_df.columns:
                        trade_df["type"] = trade_df["action"].astype(str).str.lower()
                    if "price" not in trade_df.columns:
                        for c in ["open", "price_open", "exec_price", "px", "close"]:
                            if c in trade_df.columns:
                                trade_df["price"] = trade_df[c]
                                break
            
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
            trade_df = df_from_pack(result.get('trade_df'))
            
            # 標準化交易資料
            try:
                from sss_core.normalize import normalize_trades_for_ui as norm
                trade_df = norm(trade_df)
            except Exception:
                # 後備標準化方案
                if trade_df is not None and len(trade_df) > 0:
                    trade_df = trade_df.copy()
                    trade_df.columns = [str(c).lower() for c in trade_df.columns]
                    if "trade_date" not in trade_df.columns and "date" in trade_df.columns:
                        trade_df["trade_date"] = pd.to_datetime(trade_df["date"], errors="coerce")
                    if "type" not in trade_df.columns and "action" in trade_df.columns:
                        trade_df["type"] = trade_df["action"].astype(str).str.lower()
                    if "price" not in trade_df.columns:
                        for c in ["open", "price_open", "exec_price", "px", "close"]:
                            if c in trade_df.columns:
                                trade_df["price"] = trade_df[c]
                                break
            
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
            style_table={'overflowX': 'auto', 'backgroundColor': '#1a1a1a'},
            style_cell={'textAlign': 'center', 'backgroundColor': '#1a1a1a', 'color': '#fff', 'border': '1px solid #444'},
            style_header={'backgroundColor': '#2a2a2a', 'color': '#fff', 'border': '1px solid #444'},
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
    # backtest_data 現在已經是 dict，不需要 json.loads
    results = backtest_data['results']
    result = results.get(strategy)
    
    if not result:
        return [None] * len(n_clicks)
    
    # 使用解包器函數，支援 pack_df 和傳統 JSON 字串兩種格式
    trade_df = df_from_pack(result.get('trade_df'))
    
    # 標準化交易資料
    try:
        from sss_core.normalize import normalize_trades_for_ui as norm
        trade_df = norm(trade_df)
    except Exception:
        # 後備標準化方案
        if trade_df is not None and len(trade_df) > 0:
            trade_df = trade_df.copy()
            trade_df.columns = [str(c).lower() for c in trade_df.columns]
            if "trade_date" not in trade_df.columns and "date" in trade_df.columns:
                trade_df["trade_date"] = pd.to_datetime(trade_df["date"], errors="coerce")
            if "type" not in trade_df.columns and "action" in trade_df.columns:
                trade_df["type"] = trade_df["action"].astype(str).str.lower()
            if "price" not in trade_df.columns:
                for c in ["open", "price_open", "exec_price", "px", "close"]:
                    if c in trade_df.columns:
                        trade_df["price"] = trade_df[c]
                        break
    
    # 創建下載數據
    def to_xlsx(bytes_io):
        with pd.ExcelWriter(bytes_io, engine='openpyxl') as writer:
            trade_df.to_excel(writer, sheet_name='交易紀錄', index=False)
    
    return [dcc.send_bytes(to_xlsx, f"{strategy}_交易紀錄.xlsx") if i and i > 0 else None for i in n_clicks]

# --------- 下載交易明細 CSV ---------
@app.callback(
    Output({'type': 'download-trade-details-data', 'strategy': ALL}, 'data'),
    Input({'type': 'download-trade-details-csv', 'strategy': ALL}, 'n_clicks'),
    State({'type': 'strategy-table', 'strategy': ALL}, 'data'),
    State('backtest-store', 'data'),
    prevent_initial_call=True
)
def download_trade_details_csv(n_clicks, table_data, backtest_data):
    """下載交易明細為 CSV 格式"""
    ctx_trigger = ctx.triggered_id
    if not ctx_trigger or not backtest_data:
        return [None] * len(n_clicks)
    
    # 從觸發的按鈕ID中提取策略名稱
    strategy = ctx_trigger['strategy']
    
    # 從backtest_data中獲取對應策略的交易數據
    results = backtest_data['results']
    result = results.get(strategy)
    
    if not result:
        return [None] * len(n_clicks)
    
    # 使用解包器函數，支援 pack_df 和傳統 JSON 字串兩種格式
    trade_df = df_from_pack(result.get('trade_df'))
    
    # 標準化交易資料
    try:
        from sss_core.normalize import normalize_trades_for_ui as norm
        trade_df = norm(trade_df)
    except Exception:
        # 後備標準化方案
        if trade_df is not None and len(trade_df) > 0:
            trade_df = trade_df.copy()
            trade_df.columns = [str(c).lower() for c in trade_df.columns]
            if "trade_date" not in trade_df.columns and "date" in trade_df.columns:
                trade_df["trade_date"] = pd.to_datetime(trade_df["date"], errors="coerce")
            if "type" not in trade_df.columns and "action" in trade_df.columns:
                trade_df["type"] = trade_df["action"].astype(str).str.lower()
            if "price" not in trade_df.columns:
                for c in ["open", "price_open", "exec_price", "px", "close"]:
                    if c in trade_df.columns:
                        trade_df["price"] = trade_df[c]
                        break
    
    # 創建 CSV 下載數據
    def to_csv(bytes_io):
        # 使用 UTF-8 BOM 確保 Excel 能正確顯示中文
        bytes_io.write('\ufeff'.encode('utf-8'))
        trade_df.to_csv(bytes_io, index=False, encoding='utf-8-sig')
    
    return [dcc.send_bytes(to_csv, f"{strategy}_交易明細.csv") if i and i > 0 else None for i in n_clicks]

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