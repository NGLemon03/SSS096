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
        # 支援多種欄位名稱格式
        high_col = None
        low_col = None
        close_col = None
        
        # 檢查英文欄位名稱
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            high_col = 'high'
            low_col = 'low'
            close_col = 'close'
        # 檢查中文欄位名稱
        elif '最高價' in df.columns and '最低價' in df.columns and '收盤價' in df.columns:
            high_col = '最高價'
            low_col = '最低價'
            close_col = '收盤價'
        # 檢查其他可能的欄位名稱
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
            else:
                logger.warning("找不到可用的價格欄位來計算 ATR")
                return pd.Series(index=df.index, dtype=float)
            
            price_change = close.diff().abs()
            atr = price_change.rolling(window=window).mean()
        
        # 檢查計算結果
        if atr is None or atr.empty or atr.isna().all():
            logger.warning(f"ATR 計算結果無效，window={window}")
            return pd.Series(index=df.index, dtype=float)
        
        return atr
    except Exception as e:
        logger.warning(f"ATR 計算失敗: {e}")
        return pd.Series(index=df.index, dtype=float)


def _build_benchmark_df(df_raw):
    """建立基準資料 DataFrame，統一處理欄位名稱和數據轉換"""
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
        'daily_state_base', 'trade_ledger_base', 'weight_curve_base'
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
                    
                    # === 全局開關套用區塊 ===
                    html.H6("🔧 全局開關套用", style={"marginTop":"16px","marginBottom":"8px","color":"#28a745"}),
                    dbc.Checkbox(id='global-apply-switch', value=False, label="啟用全局參數套用", style={"marginBottom":"8px"}),
                    html.Div([
                        html.Label("風險閥門 CAP", style={"fontSize":"12px","color":"#888"}),
                        dcc.Input(id='risk-cap-input', type='number', min=0.1, max=1.0, step=0.1, value=0.3, 
                                 style={"width":"80px","marginBottom":"8px"})
                    ], style={"marginBottom":"8px"}),
                    html.Div([
                        html.Label("ATR(20)/ATR(60) 比值門檻", style={"fontSize":"12px","color":"#888"}),
                        dcc.Input(id='atr-ratio-threshold', type='number', min=0.5, max=2.0, step=0.1, value=1.0, 
                                 style={"width":"80px","marginBottom":"8px"})
                    ], style={"marginBottom":"8px"}),
                    html.Div([
                        dbc.Checkbox(id='force-valve-trigger', value=False, label="強制觸發風險閥門（測試用）", style={"fontSize":"11px","color":"#dc3545"}),
                        html.Small("💡 勾選後將強制觸發風險閥門，用於測試功能", style={"color":"#dc3545","fontSize":"10px"})
                    ], style={"marginBottom":"8px"}),
                    html.Small("💡 啟用後，這些參數將套用到所有策略中，並重新計算策略信號", style={"color":"#666","fontSize":"11px"}),
                    
                    # === 風險閥門狀態顯示區域 ===
                    html.Div(id='risk-valve-status', style={"marginTop":"8px","padding":"8px","backgroundColor":"#f8f9fa","borderRadius":"4px","border":"1px solid #dee2e6"}),
                    
                    html.Div([
                        html.Small("🔒 風險閥門說明:", style={"color":"#28a745","fontWeight":"bold","fontSize":"11px"}),
                        html.Small("• CAP: 控制最大風險暴露 (0.1=10%, 0.3=30%)", style={"color":"#666","fontSize":"10px"}),
                        html.Small("• ATR比值: 當短期波動>長期波動時，自動降低風險", style={"color":"#666","fontSize":"10px"}),
                        html.Small("• 適用於: SSMA策略的delta_cap、Ensemble策略的floor/delta_cap", style={"color":"#666","fontSize":"10px"})
                    ], style={"marginTop":"4px","padding":"8px","backgroundColor":"#f8f9fa","borderRadius":"4px"}),
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
                        dcc.Tab(label="所有策略買賣點比較", value="compare"),
                        dcc.Tab(label="🔍 增強分析", value="enhanced")
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

# --------- 風險閥門狀態更新 ---------
@app.callback(
    Output('risk-valve-status', 'children'),
    [
        Input('global-apply-switch', 'value'),
        Input('risk-cap-input', 'value'),
        Input('atr-ratio-threshold', 'value'),
        Input('force-valve-trigger', 'value'),
        Input('ticker-dropdown', 'value'),
        Input('start-date', 'value'),
        Input('end-date', 'value')
    ]
)
def update_risk_valve_status(global_apply, risk_cap, atr_ratio, force_trigger, ticker, start_date, end_date):
    """動態更新風險閥門狀態顯示"""
    logger.info(f"=== 風險閥門狀態更新 ===")
    logger.info(f"global_apply: {global_apply}")
    logger.info(f"risk_cap: {risk_cap}")
    logger.info(f"atr_ratio: {atr_ratio}")
    logger.info(f"force_trigger: {force_trigger}")
    logger.info(f"ticker: {ticker}")
    logger.info(f"start_date: {start_date}")
    logger.info(f"end_date: {end_date}")
    
    if not global_apply:
        logger.info("風險閥門未啟用")
        return html.Div([
            html.Small("🔴 風險閥門未啟用", style={"color":"#dc3545","fontWeight":"bold"}),
            html.Br(),
            html.Small("點擊上方複選框啟用全局風險控制", style={"color":"#666","fontSize":"10px"})
        ])
    
    # 如果啟用，嘗試載入數據並計算 ATR 比值
    try:
        if ticker and start_date:
            logger.info(f"開始載入數據: ticker={ticker}, start_date={start_date}, end_date={end_date}")
            df_raw, _ = load_data(ticker, start_date, end_date if end_date else None, "Self")
            logger.info(f"數據載入結果: 空={df_raw.empty}, 形狀={df_raw.shape if not df_raw.empty else 'N/A'}")
            
            if not df_raw.empty:
                # 計算 ATR 比值
                logger.info("開始計算 ATR 比值")
                atr_20 = calculate_atr(df_raw, 20)
                atr_60 = calculate_atr(df_raw, 60)
                logger.info(f"ATR 計算完成: atr_20={type(atr_20)}, atr_60={type(atr_60)}")
                
                # 加入除錯資訊
                debug_info = []
                debug_info.append(f"數據欄位: {list(df_raw.columns)}")
                debug_info.append(f"數據行數: {len(df_raw)}")
                debug_info.append(f"ATR(20) 類型: {type(atr_20)}")
                debug_info.append(f"ATR(60) 類型: {type(atr_60)}")
                
                if atr_20 is not None:
                    debug_info.append(f"ATR(20) 長度: {len(atr_20) if hasattr(atr_20, '__len__') else 'N/A'}")
                    debug_info.append(f"ATR(20) 非空值: {atr_20.notna().sum() if hasattr(atr_20, 'notna') else 'N/A'}")
                
                if atr_60 is not None:
                    debug_info.append(f"ATR(60) 長度: {len(atr_60) if hasattr(atr_60, '__len__') else 'N/A'}")
                    debug_info.append(f"ATR(60) 非空值: {atr_60.notna().sum() if hasattr(atr_60, 'notna') else 'N/A'}")
                
                # 確保 ATR 數據有效
                if (atr_20 is not None and atr_60 is not None and 
                    hasattr(atr_20, 'empty') and hasattr(atr_60, 'empty') and
                    not atr_20.empty and not atr_60.empty):
                    
                    # 檢查是否有足夠的非空值
                    atr_20_valid = atr_20.dropna()
                    atr_60_valid = atr_60.dropna()
                    
                    if len(atr_20_valid) > 0 and len(atr_60_valid) > 0:
                        # 取最新的 ATR 值進行比較
                        atr_20_latest = atr_20_valid.iloc[-1]
                        atr_60_latest = atr_60_valid.iloc[-1]
                        
                        debug_info.append(f"ATR(20) 最新值: {atr_20_latest:.6f}")
                        debug_info.append(f"ATR(60) 最新值: {atr_60_latest:.6f}")
                        
                        if atr_60_latest > 0:
                            atr_ratio_current = atr_20_latest / atr_60_latest
                            debug_info.append(f"ATR 比值: {atr_ratio_current:.4f}")
                            
                            # 判斷是否需要觸發風險閥門
                            valve_triggered = atr_ratio_current > atr_ratio
                            
                            # 如果啟用強制觸發，則強制觸發風險閥門
                            if force_trigger:
                                valve_triggered = True
                                logger.info(f"強制觸發風險閥門啟用")
                            
                            # 記錄風險閥門狀態到日誌
                            logger.info(f"ATR 比值計算: {atr_20_latest:.6f} / {atr_60_latest:.6f} = {atr_ratio_current:.4f}")
                            logger.info(f"風險閥門門檻: {atr_ratio}, 當前比值: {atr_ratio_current:.4f}")
                            logger.info(f"風險閥門觸發: {'是' if valve_triggered else '否'}")
                            logger.info(f"風險閥門狀態: {'🔴 觸發' if valve_triggered else '🟢 正常'}")
                            
                            status_color = "#dc3545" if valve_triggered else "#28a745"
                            status_icon = "🔴" if valve_triggered else "🟢"
                            status_text = "觸發" if valve_triggered else "正常"
                            
                            # 加入強制觸發的狀態顯示
                            force_status = ""
                            if force_trigger:
                                force_status = html.Br() + html.Small("🔴 強制觸發已啟用", style={"color":"#dc3545","fontWeight":"bold","fontSize":"10px"})
                            
                            return html.Div([
                                html.Div([
                                    html.Small(f"{status_icon} 風險閥門狀態: {status_text}", 
                                              style={"color":status_color,"fontWeight":"bold","fontSize":"12px"}),
                                    force_status,
                                    html.Br(),
                                    html.Small(f"ATR(20)/ATR(60) = {atr_ratio_current:.2f}", style={"color":"#666","fontSize":"11px"}),
                                    html.Br(),
                                    html.Small(f"門檻值: {atr_ratio}", style={"color":"#666","fontSize":"11px"}),
                                    html.Br(),
                                    html.Small(f"風險CAP: {risk_cap*100:.0f}%", style={"color":"#666","fontSize":"11px"}),
                                    html.Br(),
                                    html.Small(f"現金保留下限: {(1-risk_cap)*100:.0f}%", style={"color":"#666","fontSize":"11px"}),
                                    html.Br(),
                                    html.Small("--- 除錯資訊 ---", style={"color":"#999","fontSize":"10px","fontStyle":"italic"}),
                                    html.Small([html.Div(info) for info in debug_info], style={"color":"#999","fontSize":"9px"})
                                ])
                            ])
                        else:
                            logger.warning(f"ATR(60) 值為 0，無法計算比值: {atr_60_latest:.6f}")
                            return html.Div([
                                html.Small("🟡 ATR 計算異常", style={"color":"#ffc107","fontWeight":"bold"}),
                                html.Br(),
                                html.Small(f"ATR(60) 值為 {atr_60_latest:.6f}，無法計算比值", style={"color":"#666","fontSize":"10px"}),
                                html.Br(),
                                html.Small("--- 除錯資訊 ---", style={"color":"#999","fontSize":"10px","fontStyle":"italic"}),
                                html.Small([html.Div(info) for info in debug_info], style={"color":"#999","fontSize":"9px"})
                            ])
                    else:
                        logger.warning(f"ATR 數據不足: ATR(20) 有效值={len(atr_20_valid)}, ATR(60) 有效值={len(atr_60_valid)}")
                        return html.Div([
                            html.Small("🟡 ATR 數據不足", style={"color":"#ffc107","fontWeight":"bold"}),
                            html.Br(),
                            html.Small(f"ATR(20) 有效值: {len(atr_20_valid)}, ATR(60) 有效值: {len(atr_60_valid)}", style={"color":"#666","fontSize":"10px"}),
                            html.Br(),
                            html.Small("--- 除錯資訊 ---", style={"color":"#999","fontSize":"10px","fontStyle":"italic"}),
                            html.Small([html.Div(info) for info in debug_info], style={"color":"#999","fontSize":"9px"})
                        ])
                else:
                    logger.warning("ATR 數據無效，無法計算比值")
                    return html.Div([
                        html.Small("🟡 ATR 數據無效", style={"color":"#ffc107","fontWeight":"bold"}),
                        html.Br(),
                        html.Small("無法計算 ATR 比值", style={"color":"#666","fontSize":"10px"}),
                        html.Br(),
                        html.Small("--- 除錯資訊 ---", style={"color":"#999","fontSize":"10px","fontStyle":"italic"}),
                        html.Small([html.Div(info) for info in debug_info], style={"color":"#666","fontSize":"9px"})
                    ])

            else:
                logger.warning(f"無法載入數據: ticker={ticker}, start_date={start_date}")
                return html.Div([
                    html.Small("🟡 無法載入數據", style={"color":"#ffc107","fontWeight":"bold"}),
                    html.Br(),
                    html.Small("請先選擇股票代號和日期", style={"color":"#666","fontSize":"10px"})
                ])
        else:
            logger.info("等待數據載入：未選擇股票代號或日期")
            return html.Div([
                html.Small("🟡 等待數據載入", style={"color":"#ffc107","fontWeight":"bold"}),
                html.Br(),
                html.Small("請選擇股票代號和日期", style={"color":"#666","fontSize":"10px"})
            ])
    except Exception as e:
        logger.error(f"風險閥門狀態更新失敗: {e}")
        return html.Div([
            html.Small("🟡 計算中...", style={"color":"#ffc107","fontWeight":"bold"}),
            html.Br(),
            html.Small(f"錯誤: {str(e)}", style={"color":"#666","fontSize":"10px"})
        ])

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
        Input('global-apply-switch', 'value'),
        Input('risk-cap-input', 'value'),
        Input('atr-ratio-threshold', 'value'),
        Input('force-valve-trigger', 'value'),
        Input('strategy-dropdown', 'value'),
        Input({'type': 'param-input', 'param': ALL}, 'value'),
        Input({'type': 'param-input', 'param': ALL}, 'id'),
    ],
    State('backtest-store', 'data')
)
def run_backtest(n_clicks, auto_run, ticker, start_date, end_date, discount, cooldown, bad_holding, global_apply, risk_cap, atr_ratio, force_trigger, strategy, param_values, param_ids, stored_data):
    # === 調試日誌（僅在 DEBUG 級別時顯示）===
    logger.debug(f"run_backtest 被調用 - n_clicks: {n_clicks}, auto_run: {auto_run}, trigger: {ctx.triggered_id}")
    
    # 移除自動快取清理，避免多用户衝突
    # 讓 joblib.Memory 自動管理快取，只在需要時手動清理
    if n_clicks is None and not auto_run:
        logger.debug(f"早期返回：n_clicks={n_clicks}, auto_run={auto_run}")
        return stored_data
    
    # 載入數據
    df_raw, df_factor = load_data(ticker, start_date, end_date, "Self")
    if df_raw.empty:
        logger.warning(f"無法載入 {ticker} 的數據")
        return {"error": f"無法載入 {ticker} 的數據"}
    
    ctx_trigger = ctx.triggered_id
    
    # 只在 auto-run 為 True 或按鈕被點擊時運算
    if not auto_run and ctx_trigger != 'run-btn':
        logger.debug(f"跳過回測：auto_run={auto_run}, ctx_trigger={ctx_trigger}")
        return stored_data
    
    logger.info(f"開始執行回測 - ticker: {ticker}, 策略數: {len(strategy_names)}")
    results = {}
    
    # === 新增：全局風險閥門觸發狀態追蹤 ===
    valve_triggered = False
    atr_ratio_current = None
    
    for strat in strategy_names:
        # 只使用 param_presets 中的參數
        strat_params = param_presets[strat].copy()
        strat_type = strat_params["strategy_type"]
        smaa_src = strat_params.get("smaa_source", "Self")
        
        # 為每個策略載入對應的數據
        df_raw, df_factor = load_data(ticker, start_date, end_date if end_date else None, smaa_source=smaa_src)
        
        # 應用全局風險閥門設定（如果啟用）
        logger.info(f"[{strat}] 風險閥門開關狀態: global_apply={global_apply}, 類型={type(global_apply)}")
        if global_apply:
            logger.info(f"[{strat}] 應用全局風險閥門: CAP={risk_cap}, ATR比值門檻={atr_ratio}")
            
            # 計算 ATR 比值（使用最新數據，僅用於日誌顯示）
            try:
                atr_20 = calculate_atr(df_raw, 20)
                atr_60 = calculate_atr(df_raw, 60)
                
                # 確保 ATR 數據有效
                if not atr_20.empty and not atr_60.empty:
                    atr_20_valid = atr_20.dropna()
                    atr_60_valid = atr_60.dropna()
                    
                    # 檢查樣本數量是否足夠
                    min_samples_20, min_samples_60 = 30, 60  # 至少需要 30 和 60 個樣本
                    if len(atr_20_valid) < min_samples_20 or len(atr_60_valid) < min_samples_60:
                        logger.warning(f"[{strat}] ATR 樣本不足，20期:{len(atr_20_valid)}/{min_samples_20}, 60期:{len(atr_60_valid)}/{min_samples_60}")
                        continue
                    
                    atr_20_latest = atr_20_valid.iloc[-1]
                    atr_60_latest = atr_60_valid.iloc[-1]
                    
                    # 檢查 ATR 值是否合理
                    if atr_60_latest <= 0 or not np.isfinite(atr_60_latest):
                        logger.warning(f"[{strat}] ATR(60) 值異常: {atr_60_latest}，跳過風險閥門")
                        continue
                    
                    if atr_20_latest <= 0 or not np.isfinite(atr_20_latest):
                        logger.warning(f"[{strat}] ATR(20) 值異常: {atr_20_latest}，跳過風險閥門")
                        continue
                    
                    atr_ratio_current = atr_20_latest / atr_60_latest
                    logger.info(f"[{strat}] 最新ATR比值: {atr_ratio_current:.4f} (20期:{atr_20_latest:.4f}, 60期:{atr_60_latest:.4f})")
                else:
                    logger.warning(f"[{strat}] ATR 計算結果為空")
                    
                # 強制觸發時設置標記
                if force_trigger:
                    valve_triggered = True
                    logger.info(f"[{strat}] 🔴 強制觸發風險閥門啟用")
                    
            except Exception as e:
                logger.warning(f"[{strat}] ATR 計算失敗: {e}")
        else:
            logger.info(f"[{strat}] 未啟用全局風險閥門")
        
        if strat_type == 'ssma_turn':
            calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'signal_cooldown_days', 'quantile_win']
            ssma_params = {k: v for k, v in strat_params.items() if k in calc_keys}
            backtest_params = ssma_params.copy()
            backtest_params['stop_loss'] = strat_params.get('stop_loss', 0.0)
            
            # 重新計算策略信號（因為參數可能已經被風險閥門調整）
            df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(df_raw, df_factor, **ssma_params, smaa_source=smaa_src)
            if df_ind.empty:
                continue
            result = backtest_unified(df_ind, strat_type, backtest_params, buy_dates, sell_dates, discount=discount, trade_cooldown_bars=cooldown, bad_holding=bad_holding)
            
            # === 在 ssma_turn 也套用風險閥門（和 Ensemble 一致的後置覆寫） ===
            if global_apply:
                # 判斷是否要觸發（與你的 ATR 檢查或強制觸發一致）
                valve_triggered_local = False
                ratio_local = None
                try:
                    atr_20 = calculate_atr(df_raw, 20)
                    atr_60 = calculate_atr(df_raw, 60)
                    if not atr_20.empty and not atr_60.empty:
                        a20 = atr_20.dropna().iloc[-1]
                        a60 = atr_60.dropna().iloc[-1]
                        if a60 > 0:
                            ratio_local = float(a20 / a60)
                            valve_triggered_local = (ratio_local > atr_ratio)  # 與進階分析一致：使用 ">"
                except Exception:
                    pass

                if force_trigger:
                    valve_triggered_local = True
                    if ratio_local is None:
                        ratio_local = 1.5

                if valve_triggered_local:
                    from SSS_EnsembleTab import risk_valve_backtest, CostParams
                    # 取得 open 價；df_raw 欄位名稱是小寫
                    open_px = df_raw['open'] if 'open' in df_raw.columns else df_raw['close']
                    # 從回測輸出抓 w（先用標準化 daily_state，如果沒有就用原 daily_state）
                    w_series = None
                    try:
                        ds_std = df_from_pack(result.get('daily_state_std'))
                        if ds_std is not None and not ds_std.empty and 'w' in ds_std.columns:
                            w_series = ds_std['w']
                    except Exception:
                        pass
                    if w_series is None:
                        ds = df_from_pack(result.get('daily_state'))
                        if ds is not None and not ds.empty and 'w' in ds.columns:
                            w_series = ds['w']

                    if w_series is not None:
                        # 交易成本（與 Ensemble 分支一致）
                        trade_cost = strat_params.get('trade_cost', {})
                        cost_params = CostParams(
                            buy_fee_bp=float(trade_cost.get("buy_fee_bp", 4.27)),
                            sell_fee_bp=float(trade_cost.get("sell_fee_bp", 4.27)),
                            sell_tax_bp=float(trade_cost.get("sell_tax_bp", 30.0))
                        )

                        # === 全局套用風險閥門：確保參數一致性 (2025/08/20) ===
                        global_valve_params = {
                            "open_px": open_px,
                            "w": w_series,
                            "cost": cost_params,
                            "benchmark_df": df_raw,
                            "mode": "cap",
                            "cap_level": float(risk_cap),
                            "slope20_thresh": 0.0, 
                            "slope60_thresh": 0.0,
                            "atr_win": 20, 
                            "atr_ref_win": 60,
                            "atr_ratio_mult": float(ratio_local if ratio_local is not None else atr_ratio),   # 若你有 local ratio，就用 local；否則全局 atr_ratio
                            "use_slopes": True,
                            "slope_method": "polyfit",
                            "atr_cmp": "gt"
                        }
                        
                        # 記錄全局風險閥門配置
                        logger.info(f"[Global] 風險閥門配置: cap_level={global_valve_params['cap_level']}, atr_ratio_mult={global_valve_params['atr_ratio_mult']}")
                        
                        rv = risk_valve_backtest(**global_valve_params)

                        # 覆寫結果，確保 UI 與輸出一致（和 Ensemble 分支對齊）
                        result['equity_curve']     = pack_series(rv["daily_state_valve"]["equity"])
                        result['daily_state']      = pack_df(rv["daily_state_valve"])
                        result['daily_state_std']  = pack_df(rv["daily_state_valve"])
                        result['trade_ledger']     = pack_df(rv["trade_ledger_valve"])
                        result['trade_ledger_std'] = pack_df(rv["trade_ledger_valve"])
                        result['weight_curve']     = pack_series(rv["weights_valve"])
                        # 給 UI 的標記（下個小節會用到）
                        result['valve'] = {
                            "applied": True,
                            "cap": float(risk_cap),
                            "atr_ratio": ratio_local
                        }
                        
                        logger.info(f"[{strat}] SSMA 風險閥門已套用（cap={risk_cap}, ratio={ratio_local:.4f}）")
                    else:
                        logger.warning(f"[{strat}] SSMA 無法取得權重序列，跳過風險閥門套用")
                else:
                    logger.info(f"[{strat}] SSMA 風險閥門未觸發，使用原始結果")
                    # 給 UI 的標記（未觸發）
                    result['valve'] = {
                        "applied": False,
                        "cap": float(risk_cap),
                        "atr_ratio": ratio_local if ratio_local is not None else "N/A"
                    }
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
                
                # 註解掉原本的無條件風險閥門調整（會造成 floor 方向錯誤）
                # if global_apply:
                #     logger.info(f"[{strat}] Ensemble 策略應用風險閥門: 原始 delta_cap={ensemble_params.delta_cap}, floor={ensemble_params.floor}")
                #     
                #     # 調整 delta_cap（最大風險暴露）
                #     if ensemble_params.delta_cap > risk_cap:
                #         ensemble_params.delta_cap = risk_cap
                #         logger.info(f"[{strat}] 調整 delta_cap 為 {ensemble_params.delta_cap}")
                #     
                #     # 調整 floor（最小現金保留）
                #     min_floor = 1 - risk_cap
                #     if ensemble_params.floor < min_floor:
                #         ensemble_params.floor = min_floor
                #         logger.info(f"[{strat}] 調整 floor 為 {ensemble_params.floor}")
                #     
                #     logger.info(f"[{strat}] Ensemble 策略最終參數: delta_cap={ensemble_params.delta_cap}, floor={ensemble_params.floor}")
                
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
                
                # --- 新增：只在 ATR 觸發時啟用風險閥門 ---
                valve_triggered = False
                ratio = None
                try:
                    atr_20 = calculate_atr(df_raw, 20)
                    atr_60 = calculate_atr(df_raw, 60)
                    
                    # 增加詳細的調試資訊
                    logger.info(f"[{strat}] Ensemble ATR 計算: atr_20={type(atr_20)}, atr_60={type(atr_60)}")
                    
                    if not atr_20.empty and not atr_60.empty:
                        atr_20_valid = atr_20.dropna()
                        atr_60_valid = atr_60.dropna()
                        
                        logger.info(f"[{strat}] Ensemble ATR 有效值: atr_20={len(atr_20_valid)}, atr_60={len(atr_60_valid)}")
                        
                        if len(atr_20_valid) > 0 and len(atr_60_valid) > 0:
                            a20 = atr_20_valid.iloc[-1]
                            a60 = atr_60_valid.iloc[-1]
                            
                            logger.info(f"[{strat}] Ensemble ATR 最新值: a20={a20:.6f}, a60={a60:.6f}")
                            
                            if a60 > 0:
                                ratio = float(a20 / a60)
                                valve_triggered = (ratio > atr_ratio)  # 與進階分析一致：使用 ">"
                                logger.info(f"[{strat}] Ensemble ATR 比值: {ratio:.4f} (門檻={atr_ratio}) -> 觸發={valve_triggered}")
                                
                                # 增加風險閥門觸發的詳細資訊
                                if valve_triggered:
                                    logger.info(f"[{strat}] 🔴 風險閥門觸發！ATR比值({ratio:.4f}) > 門檻({atr_ratio})")
                                else:
                                    logger.info(f"[{strat}] 🟢 風險閥門未觸發，ATR比值({ratio:.4f}) <= 門檻({atr_ratio})")
                            else:
                                logger.warning(f"[{strat}] Ensemble ATR(60) 值為 0，無法計算比值")
                        else:
                            logger.warning(f"[{strat}] Ensemble ATR 數據不足")
                    else:
                        logger.warning(f"[{strat}] Ensemble ATR 計算結果為空")
                        
                except Exception as e:
                    logger.warning(f"[{strat}] 無法計算 Ensemble ATR 比值: {e}")
                    logger.warning(f"[{strat}] 錯誤詳情: {type(e).__name__}: {str(e)}")

                # 如果啟用強制觸發，則強制觸發風險閥門
                if force_trigger:
                    valve_triggered = True
                    logger.info(f"[{strat}] 🔴 強制觸發風險閥門啟用")
                    if ratio is None:
                        ratio = 1.5  # 設定一個預設值用於顯示

                # 使用新的 ensemble_runner 執行
                backtest_result = run_ensemble_backtest(cfg)

                # 若全局開關開啟且達觸發條件，才在權重序列上套用 CAP
                if global_apply and valve_triggered:
                    from SSS_EnsembleTab import risk_valve_backtest
                    bench = df_raw  # 已含 open/high/low/close/volume
                    
                    logger.info(f"[{strat}] 🔴 開始套用風險閥門: cap={risk_cap}, ratio={ratio:.4f}")
                    
                    rv = risk_valve_backtest(
                        open_px=backtest_result.price_series,
                        w=backtest_result.weight_curve,
                        cost=cost_params,
                        benchmark_df=bench,
                        mode="cap",
                        cap_level=float(risk_cap),
                        slope20_thresh=0.0, slope60_thresh=0.0,
                        atr_win=20, atr_ref_win=60,
                        atr_ratio_mult=float(atr_ratio),   # ← UI 的 ATR 門檻
                        use_slopes=True,                   # ← 跟增強分析一致
                        slope_method="polyfit",            # ← 跟增強分析一致
                        atr_cmp="gt"                       # ← 跟增強分析一致（用 >）
                    )
                    # 覆寫結果，確保 UI 與輸出一致
                    backtest_result.daily_state = rv["daily_state_valve"]
                    backtest_result.ledger = rv["trade_ledger_valve"]
                    backtest_result.weight_curve = rv["weights_valve"]
                    backtest_result.equity_curve = rv["daily_state_valve"]["equity"]
                    logger.info(f"[{strat}] 風險閥門已套用（cap={risk_cap}, ratio={ratio:.4f}）")
                    
                    # 增加風險閥門效果的詳細資訊
                    if "metrics" in rv:
                        logger.info(f"[{strat}] 風險閥門效果: PF原始={rv['metrics'].get('pf_orig', 'N/A'):.2f}, PF閥門={rv['metrics'].get('pf_valve', 'N/A'):.2f}")
                        logger.info(f"[{strat}] 風險閥門效果: MDD原始={rv['metrics'].get('mdd_orig', 'N/A'):.2f}%, MDD閥門={rv['metrics'].get('mdd_valve', 'N/A'):.2f}%")
                    
                    # 給 UI 的標記（與 SSMA 分支對齊）
                    result['valve'] = {
                        "applied": True,
                        "cap": float(risk_cap),
                        "atr_ratio": ratio
                    }
                    
                    # 新增：讓全局區段知道已套用過
                    result['_risk_valve_applied'] = True
                else:
                    if global_apply:
                        logger.info(f"[{strat}] 🟢 風險閥門未觸發，使用原始參數")
                        # 給 UI 的標記（未觸發）
                        result['valve'] = {
                            "applied": False,
                            "cap": float(risk_cap),
                            "atr_ratio": ratio if ratio is not None else "N/A"
                        }
                    else:
                        logger.info(f"[{strat}] ⚪ 全局風險閥門未啟用")
                        # 給 UI 的標記（未啟用）
                        result['valve'] = {
                            "applied": False,
                            "cap": "N/A",
                            "atr_ratio": "N/A"
                        }
                
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
            
            # 為其他策略類型添加 valve 標記
            if global_apply:
                result['valve'] = {
                    "applied": False,  # 其他策略類型暫時不支援風險閥門
                    "cap": float(risk_cap),
                    "atr_ratio": "N/A"
                }
            else:
                result['valve'] = {
                    "applied": False,
                    "cap": "N/A",
                    "atr_ratio": "N/A"
                }
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
        
        # === 全局風險閥門：逐日動態套用（與增強分析一致） ===
        if global_apply:
            # 新增：若策略分支已經套用，就不要再來一次
            if result.get('_risk_valve_applied'):
                logger.info(f"[{strat}] 已由策略分支套用風險閥門，跳過全局再次套用")
            else:
                # 原本區塊從這裡開始
                # 1) 取 ds（daily_state），並解包
                ds_raw = result.get("daily_state_std") or result.get("daily_state")
                ds = df_from_pack(ds_raw)
                if ds is None or ds.empty or "w" not in ds.columns:
                    logger.warning(f"[{strat}] daily_state 不含 'w'，跳過全局風險閥門")
                else:
                    # 2) 使用與進階分析一致的風險閥門判斷邏輯
                    try:
                        from SSS_EnsembleTab import compute_risk_valve_signals
                        
                        # 建立基準資料（有高低價就帶上）
                        bench = _build_benchmark_df(df_raw)
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
                        
                        # 使用進階分析的預設參數：斜率門檻=0，ATR比值=1.5，比較符號=">"
                        risk_signals = compute_risk_valve_signals(
                            benchmark_df=bench,
                            slope20_thresh=0.0,      # 20日斜率門檻
                            slope60_thresh=0.0,      # 60日斜率門檻
                            atr_win=20,              # ATR計算窗口
                            atr_ref_win=60,          # ATR參考窗口
                            atr_ratio_mult=float(atr_ratio),  # ATR比值門檻
                            use_slopes=True,         # 啟用斜率條件
                            slope_method="polyfit",   # 使用多項式擬合斜率
                            atr_cmp="gt"             # 使用 ">" 比較符號
                        )
                        
                        mask = risk_signals["risk_trigger"].reindex(ds.index).fillna(False)
                        logger.info(f"[{strat}] 進階分析風險閥門：斜率條件啟用，ATR比值門檻={atr_ratio}")
                        
                    except Exception as e:
                        logger.warning(f"[{strat}] 無法使用進階分析風險閥門，回退到 ATR-only: {e}")
                        # 回退到原本的 ATR-only 邏輯
                        atr20 = calculate_atr(df_raw, 20)
                        atr60 = calculate_atr(df_raw, 60)
                        if atr20 is None or atr60 is None:
                            logger.warning(f"[{strat}] 無法計算 ATR20/60，跳過全局風險閥門")
                            continue
                        ratio = (atr20 / atr60).replace([np.inf, -np.inf], np.nan)
                        mask = (ratio > float(atr_ratio))  # 與進階分析一致：使用 ">" 比較
                    
                    if force_trigger:
                        mask[:] = True  # 強制全部日子套 CAP

                    # 在全局壓 w 之前加：保存未套閥門的 baseline
                    if "daily_state_base" not in result and ds_raw is not None:
                        result["daily_state_base"] = ds_raw  # 保存未套閥門的 baseline
                    
                    # ➋ 追加以下兩行（放在同一段、覆寫 w 之前）
                    if "trade_ledger_base" not in result and result.get("trade_ledger") is not None:
                        result["trade_ledger_base"] = result["trade_ledger"]
                    if "weight_curve_base" not in result and result.get("weight_curve") is not None:
                        result["weight_curve_base"] = result["weight_curve"]
                    
                    # 3) 對齊到 ds.index，逐日壓 w 至 CAP
                    mask_aligned = mask.reindex(ds.index).fillna(False).to_numpy()
                    w = ds["w"].astype(float).to_numpy()
                    w_new = w.copy()
                    w_new[mask_aligned] = np.minimum(w_new[mask_aligned], float(risk_cap))
                    ds["w"] = w_new

                    # 4) 回寫 ds，並重算交易/權益
                    result["daily_state_std"] = pack_df(ds)

                    # open 價（沒有 open 就退而求其次用收盤價）
                    open_px = (df_raw["open"] if "open" in df_raw.columns else df_raw.get("收盤價")).astype(float)
                    open_px = open_px.reindex(ds.index).dropna()

                    # 若你沿用現有的 risk_valve_backtest，給 cap_level=1.0 表示「w 已經是目標序列」
                    try:
                        from SSS_EnsembleTab import (
                            risk_valve_backtest,
                            CostParams,
                            _mdd_from_daily_equity,
                            _sell_returns_pct_from_ledger,
                        )
                        
                        # 成本參數
                        trade_cost = (strat_params.get("trade_cost", {}) 
                                      if isinstance(strat_params, dict) else {})
                        cost = CostParams(
                            buy_fee_bp=float(trade_cost.get("buy_fee_bp", 4.27)),
                            sell_fee_bp=float(trade_cost.get("sell_fee_bp", 4.27)),
                            sell_tax_bp=float(trade_cost.get("sell_tax_bp", 30.0)),
                        )
                        
                        # 基準（有高低價就帶上）
                        bench = _build_benchmark_df(df_raw)
                        
                        # === 風險閥門回測：確保參數一致性 (2025/08/20) ===
                        valve_params = {
                            "open_px": open_px,
                            "w": ds["w"].astype(float).reindex(open_px.index).fillna(0.0),
                            "cost": cost,
                            "benchmark_df": bench,
                            "mode": "cap",
                            "cap_level": float(risk_cap),  # 使用實際的風險上限值
                            "slope20_thresh": 0.0,         # 👈 與進階分析一致：20日斜率門檻
                            "slope60_thresh": 0.0,         # 👈 與進階分析一致：60日斜率門檻
                            "atr_win": 20, 
                            "atr_ref_win": 60,
                            "atr_ratio_mult": float(atr_ratio),   # 👈 與全局一致
                            "use_slopes": True,            # 👈 與進階分析一致：啟用斜率條件
                            "slope_method": "polyfit",     # 👈 與進階分析一致：使用多項式擬合
                            "atr_cmp": "gt"               # 👈 與進階分析一致：使用 ">" 比較符號
                        }
                        
                        # 記錄風險閥門配置用於診斷
                        logger.info(f"[{strat}] 風險閥門配置: cap_level={valve_params['cap_level']}, atr_ratio_mult={valve_params['atr_ratio_mult']}")
                        
                        result_cap = risk_valve_backtest(**valve_params)
                    except Exception as e:
                        logger.warning(f"[{strat}] 無法導入 risk_valve_backtest: {e}")
                        result_cap = None

                    if result_cap:
                        # === 安全覆寫：清掉舊鍵並補齊新鍵 ===
                        logger.info(f"[UI_CHECK] 即將覆寫：new_trades={len(result_cap.get('trade_ledger_valve', pd.DataFrame()))} rows, new_ds={len(result_cap.get('daily_state_valve', pd.DataFrame()))} rows")
                        
                        # 1) 覆寫結果 —— 一律用 pack_df/pack_series
                        if 'trade_ledger_valve' in result_cap:
                            result['trades'] = pack_df(result_cap['trade_ledger_valve'])
                            result['trade_ledger'] = pack_df(result_cap['trade_ledger_valve'])
                            result['trade_ledger_std'] = pack_df(result_cap['trade_ledger_valve'])
                        
                        if 'daily_state_valve' in result_cap:
                            result['daily_state'] = pack_df(result_cap['daily_state_valve'])
                            result['daily_state_std'] = pack_df(result_cap['daily_state_valve'])
                        
                        if 'weights_valve' in result_cap:
                            result['weight_curve'] = pack_series(result_cap['weights_valve'])
                        
                        # 權益曲線：若是 Series
                        if 'daily_state_valve' in result_cap and 'equity' in result_cap['daily_state_valve']:
                            try:
                                result['equity_curve'] = pack_series(result_cap['daily_state_valve']['equity'])
                            except Exception:
                                # 若你存的是 DataFrame
                                result['equity_curve'] = pack_df(result_cap['daily_state_valve']['equity'].to_frame('equity'))
                        
                        # 2) **關鍵**：把 UI 可能拿來用的舊快取清掉，強迫 UI 走新資料
                        for k in ['trades_ui', 'trade_df', 'trade_ledger_std', 'metrics']:
                            if k in result:
                                result.pop(k, None)
                        

                        
                        # 新增：標記 valve 狀態供後續快取判斷
                        result['valve'] = {
                            'applied': True,
                            'cap': float(risk_cap),
                            'atr_ratio_mult': float(atr_ratio),
                        }
                        
                        # 新增：存入 ensemble 參數（若可取得）
                        # 在全局風險閥門區塊中，我們沒有 cfg 物件，直接使用預設值
                        result["ensemble_params"] = {"majority_k_pct": 0.55}  # 預設值

                        # 2025-08-20 重算指標以保留績效資訊 #app_dash.py
                        ledger_valve = result_cap.get('trade_ledger_valve', pd.DataFrame())
                        ds_valve = result_cap.get('daily_state_valve', pd.DataFrame())
                        if not ledger_valve.empty and not ds_valve.empty and 'equity' in ds_valve:
                            r = _sell_returns_pct_from_ledger(ledger_valve)
                            eq = ds_valve['equity']
                            total_ret = eq.iloc[-1] / eq.iloc[0] - 1
                            years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1)
                            ann_ret = (1 + total_ret) ** (1 / years) - 1
                            mdd = _mdd_from_daily_equity(eq)
                            dd = eq / eq.cummax() - 1
                            blocks = (~(dd < 0)).cumsum()
                            dd_dur = int((dd.groupby(blocks).cumcount() + 1).where(dd < 0).max() or 0)
                            num_trades = len(r)
                            win_rate = (r > 0).sum() / num_trades if num_trades > 0 else 0
                            avg_win = r[r > 0].mean() if win_rate > 0 else np.nan
                            avg_loss = r[r < 0].mean() if win_rate < 1 else np.nan
                            payoff = abs(avg_win / avg_loss) if avg_loss != 0 and not np.isnan(avg_win) else np.nan
                            daily_r = eq.pct_change().dropna()
                            sharpe = (daily_r.mean() * np.sqrt(252)) / daily_r.std() if daily_r.std() != 0 else np.nan
                            downside = daily_r[daily_r < 0]
                            sortino = (daily_r.mean() * np.sqrt(252)) / downside.std() if downside.std() != 0 else np.nan
                            ann_vol = daily_r.std() * np.sqrt(252) if len(daily_r) > 0 else np.nan
                            prof = r[r > 0].sum()
                            loss = abs(r[r < 0].sum())
                            pf = prof / loss if loss != 0 else np.nan
                            win_flag = r > 0
                            grp = (win_flag != win_flag.shift()).cumsum()
                            consec = win_flag.groupby(grp).cumcount() + 1
                            max_wins = int(consec[win_flag].max() if True in win_flag.values else 0)
                            max_losses = int(consec[~win_flag].max() if False in win_flag.values else 0)
                            result['metrics'] = {
                                'total_return': float(total_ret),
                                'annual_return': float(ann_ret),
                                'max_drawdown': float(mdd),
                                'max_drawdown_duration': dd_dur,
                                'calmar_ratio': float(ann_ret / abs(mdd)) if mdd < 0 else np.nan,
                                'num_trades': int(num_trades),
                                'win_rate': float(win_rate),
                                'avg_win': float(avg_win) if not np.isnan(avg_win) else np.nan,
                                'avg_loss': float(avg_loss) if not np.isnan(avg_loss) else np.nan,
                                'payoff_ratio': float(payoff) if not np.isnan(payoff) else np.nan,
                                'sharpe_ratio': float(sharpe) if not np.isnan(sharpe) else np.nan,
                                'sortino_ratio': float(sortino) if not np.isnan(sortino) else np.nan,
                                'max_consecutive_wins': max_wins,
                                'max_consecutive_losses': max_losses,
                                'annualized_volatility': float(ann_vol) if not np.isnan(ann_vol) else np.nan,
                                'profit_factor': float(pf) if not np.isnan(pf) else np.nan,
                            }
                        
                        # 3) 給 UI 一個旗標與參數，便於顯示「已套用」
                        result['_risk_valve_applied'] = True
                        result['_risk_valve_params'] = {
                            'cap': float(risk_cap),
                            'atr_ratio': float(atr_ratio),
                            'atr20_last': float(atr_20_valid.iloc[-1]) if len(atr_20_valid) > 0 else None,
                            'atr60_last': float(atr_60_valid.iloc[-1]) if len(atr_60_valid) > 0 else None,
                        }
                        
                        true_days = int(mask_aligned.sum())
                        logger.info(f"[{strat}] 全局風險閥門已套用（逐日），風險天數={true_days}, CAP={risk_cap:.2f}")
                    else:
                        logger.warning(f"[{strat}] 風險閥門重算沒有返回結果")

        
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
    
    # === 回測完成日誌 ===
    logger.info(f"回測完成 - 策略數: {len(results)}, ticker: {ticker}, 數據行數: {len(df_raw_main)}")
    logger.debug(f"策略列表: {list(results.keys())}")
    
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
    # === 調試日誌（僅在 DEBUG 級別時顯示）===
    logger.debug(f"update_tab 被調用 - tab: {tab}, strategy: {selected_strategy}")
    
    if not data:
        logger.warning("沒有回測數據，顯示提示訊息")
        return html.Div("請先執行回測")
    
    # data 現在已經是 dict，不需要 json.loads
    results = data['results']
    df_raw = df_from_pack(data['df_raw'])  # 使用 df_from_pack 統一解包
    ticker = data['ticker']
    strategy_names = list(results.keys())
    
    logger.debug(f"數據解析完成 - 策略數: {len(strategy_names)}, ticker: {ticker}, 數據行數: {len(df_raw) if df_raw is not None else 0}")
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
            
            # === 統一入口：讀取交易表、日狀態、權益曲線 ===
            # 讀交易表的統一入口：先用標準鍵，再 fallback
            trade_df = None
            candidates = [
                result.get('trades'),      # 全局覆寫後標準鍵
                result.get('trades_ui'),   # 舊格式（若還存在）
                result.get('trade_df'),    # 某些策略自帶
            ]
            
            for cand in candidates:
                if cand is None:
                    continue
                # cand 可能已是 DataFrame 或打包字串
                df = df_from_pack(cand) if isinstance(cand, str) else cand
                if df is not None and getattr(df, 'empty', True) is False:
                    trade_df = df.copy()
                    break
            
            if trade_df is None:
                # 建立空表避免後續崩
                trade_df = pd.DataFrame(columns=['trade_date','type','price','shares','return'])
            
            # 日狀態與權益曲線也類似處理
            daily_state_std = df_from_pack(result.get('daily_state_std'))
            if daily_state_std is None or daily_state_std.empty:
                daily_state_std = df_from_pack(result.get('daily_state'))
            if daily_state_std is None:
                daily_state_std = pd.DataFrame()
            
            trade_ledger_std = df_from_pack(result.get('trade_ledger_std'))
            if trade_ledger_std is None or trade_ledger_std.empty:
                trade_ledger_std = df_from_pack(result.get('trade_ledger'))
            if trade_ledger_std is None:
                trade_ledger_std = pd.DataFrame()
            
            # 記錄來源選擇結果
            logger.info(f"[UI] {strategy} trades 來源優先序：trades -> trades_ui -> trade_df；實際使用={'trades' if 'trades' in result else ('trades_ui' if 'trades_ui' in result else 'trade_df')}")
            logger.info(f"[UI] {strategy} 讀取後前 3 列 w: {daily_state_std['w'].head(3).tolist() if 'w' in daily_state_std.columns else 'N/A'}")
            
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
                logger.info(f"[UI] daily_state_display head=\n{daily_state_display[['equity','cash']].head(3) if 'equity' in daily_state_display.columns and 'cash' in daily_state_display.columns else 'Missing equity/cash columns'}")
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
            
            # === 計算風險閥門徽章內容 ===
            valve = results.get(strategy, {}).get('valve', {}) or {}
            valve_badge_text = ("已套用" if valve.get("applied") else "未套用")
            valve_badge_extra = []
            if isinstance(valve.get("cap"), (int, float)):
                valve_badge_extra.append(f"CAP={valve['cap']:.2f}")
            if isinstance(valve.get("atr_ratio"), (int, float)):
                valve_badge_extra.append(f"ATR比值={valve['atr_ratio']:.2f}")
            elif valve.get("atr_ratio") == "forced":
                valve_badge_extra.append("強制觸發")
            
            valve_badge = html.Span(
                "🛡️ 風險閥門：" + valve_badge_text + ((" | " + " | ".join(valve_badge_extra)) if valve_badge_extra else ""),
                style={
                    "marginLeft": "8px",
                    "color": ("#dc3545" if valve.get("applied") else "#6c757d"),
                    "fontWeight": "bold"
                }
            ) if valve else html.Span("")

            strategy_content = html.Div([
                html.H4([
                    f"回測策略: {strategy} ",
                    html.Span("ⓘ", title=tooltip, style={"cursor": "help", "color": "#888"}),
                    valve_badge
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
        
    elif tab == "enhanced":
        # === 增強分析頁面 ===
        enhanced_controls = html.Div([
            html.H4("🔍 增強分析"),
            
            # === 新增：全局參數套用狀態提示 ===
            html.Div([
                html.Div(id="enhanced-global-status", style={
                    "padding": "12px",
                    "marginBottom": "16px",
                    "borderRadius": "8px",
                    "border": "1px solid #dee2e6",
                    "backgroundColor": "#f8f9fa"
                })
            ]),
            
            # === 新增：從回測結果載入區塊 ===
            html.Details([
                html.Summary("🧠 從回測結果載入"),
                html.Div([
                    html.Div("選擇策略（自動評分：ledger_std > ledger > trade_df）", 
                             style={"marginBottom":"8px","fontSize":"14px","color":"#666"}),
                    dcc.Dropdown(
                        id="enhanced-strategy-selector",
                        placeholder="請先執行回測...",
                        style={"width":"100%","marginBottom":"8px"}
                    ),
                    html.Button("載入選定策略", id="load-enhanced-strategy", n_clicks=0, 
                               style={"width":"100%","marginBottom":"8px"}),
                    html.Div(id="enhanced-load-status", style={"fontSize":"12px","color":"#888"}),
                    html.Div("💡 回測完成後會自動快取最佳策略", 
                             style={"fontSize":"11px","color":"#666","fontStyle":"italic","marginTop":"4px"})
                ])
            ], style={"marginBottom":"16px"}),
            
            # === 隱藏的 cache store ===
            dcc.Store(id="enhanced-trades-cache"),
            
            html.Details([
                html.Summary("風險閥門回測"),
                html.Div([
                    dcc.Dropdown(
                        id="rv-mode", options=[
                            {"label":"降低上限 (cap)","value":"cap"},
                            {"label":"禁止加碼 (ban_add)","value":"ban_add"},
                        ], value="cap", clearable=False, style={"width":"240px"}
                    ),
                    dcc.Slider(id="rv-cap", min=0.1, max=1.0, step=0.05, value=0.5,
                               tooltip={"placement":"bottom","always_visible":True}),
                    html.Div("ATR(20)/ATR(60) 比值門檻", style={"marginTop":"8px"}),
                    dcc.Slider(id="rv-atr-mult", min=1.0, max=2.0, step=0.05, value=1.3,
                               tooltip={"placement":"bottom","always_visible":True}),
                    html.Button("執行風險閥門回測", id="run-rv", n_clicks=0, style={"marginTop":"8px"})
                ])
            ]),
            
            html.Div(id="rv-summary", style={"marginTop":"12px"}),
            dcc.Graph(id="rv-equity-chart"),
            dcc.Graph(id="rv-dd-chart"),
            
            # === 新增：數據比對功能 ===
            html.Details([
                html.Summary("🔍 數據比對與診斷"),
                html.Div([
                    html.Div("直接輸出實際數據進行比對，診斷全局套用與強化分析結果不同的問題", 
                             style={"marginBottom":"8px","fontSize":"14px","color":"#666"}),
                    html.Button("輸出數據比對報告", id="export-data-comparison", n_clicks=0, 
                               style={"width":"100%","marginBottom":"8px","backgroundColor":"#17a2b8","color":"white"}),
                    html.Div(id="data-comparison-output", style={"fontSize":"12px","color":"#666","marginTop":"8px"}),
                    dcc.Download(id="data-comparison-csv")
                ])
            ], style={"border":"1px solid #17a2b8","borderRadius":"8px","padding":"12px","marginTop":"12px"}),
            
            # === 新增：風險-報酬地圖（Pareto Map）區塊 ===
            html.Details([
                html.Summary("📊 風險-報酬地圖（Pareto Map）"),
                html.Div([
                    html.Div("生成策略的風險-報酬分析圖表", 
                             style={"marginBottom":"8px","fontSize":"14px","color":"#666"}),
                    html.Button("生成 Pareto Map", id="generate-pareto-map", n_clicks=0, 
                               style={"width":"100%","marginBottom":"8px"}),
                    html.Div(id="pareto-map-status", style={"fontSize":"12px","color":"#888","marginBottom":"8px"}),
                    dcc.Graph(id="pareto-map-graph", style={"height":"600px"}),
                    html.Div([
                        html.Button("📥 下載 Pareto Map 數據 (CSV)", id="download-pareto-csv", n_clicks=0,
                                   style={"width":"100%","marginBottom":"8px"}),
                        dcc.Download(id="pareto-csv-download"),
                        html.H6("圖表說明：", style={"marginTop":"16px","marginBottom":"8px"}),
                        html.Ul([
                            html.Li("橫軸：最大回撤（愈左愈好）"),
                            html.Li("縱軸：PF 獲利因子（愈上愈好）"),
                            html.Li("顏色：右尾調整幅度（紅色=削減右尾，藍色=放大右尾，0為中線）"),
                            html.Li("點大小：風險觸發天數（越大＝管得越勤）"),
                            html.Li("理想區域：綠色虛線框內（又上又左、顏色接近中線、點不要大到誇張）")
                        ], style={"fontSize":"12px","color":"#666"})
                    ])
                ])
            ], style={"border":"1px solid #333","borderRadius":"8px","padding":"12px","marginTop":"12px"}),
            # === 交易貢獻拆解區塊 ===
            html.Details([
                html.Summary("🔍 交易貢獻拆解"),
                html.Div([
                    html.Div("拆解交易貢獻，分析不同加碼/減碼階段的績效表現", 
                             style={"marginBottom":"8px","fontSize":"14px","color":"#666"}),
                    html.Div([
                        html.Div([
                            html.Label("最小間距 (天)", style={"fontSize":"12px","color":"#888"}),
                            dcc.Input(id="phase-min-gap", type="number", value=5, min=0, max=30, step=1,
                                     style={"width":"80px","marginRight":"16px"})
                        ], style={"display":"inline-block","marginRight":"16px"}),
                                            html.Div([
                        html.Label("冷卻期 (天)", style={"fontSize":"12px","color":"#888"}),
                        dcc.Input(id="phase-cooldown", type="number", value=10, min=0, max=30, step=1,
                                 style={"width":"80px"})
                    ], style={"display":"inline-block"})
                ], style={"marginBottom":"8px"}),
                html.Div([
                    html.Button("執行交易貢獻拆解", id="run-phase", n_clicks=0, 
                               style={"width":"48%","marginBottom":"8px","marginRight":"2%"}),
                    html.Button("批量測試參數範圍", id="run-batch-phase", n_clicks=0,
                               style={"width":"48%","marginBottom":"8px","marginLeft":"2%","backgroundColor":"#28a745","color":"white"})
                ], style={"display":"flex","justifyContent":"space-between"}),
                    html.Div([
                        html.H6("參數說明：", style={"marginTop":"16px","marginBottom":"8px"}),
                        html.Ul([
                            html.Li("最小間距：兩次加碼至少要間隔幾天，才算獨立訊號（過濾短期噪音）"),
                            html.Li("冷卻期：每次加碼後，必須過多久才允許下一筆加碼（避免過度曝險）"),
                            html.Li("用途：讓拆解聚焦在比較有意義的加碼波段，避免被短期小單稀釋")
                        ], style={"fontSize":"12px","color":"#666","marginBottom":"16px"}),
                        html.Div(id="phase-table"),
                        html.Div([
                            html.H6("批量測試結果", style={"marginTop":"16px","marginBottom":"8px","color":"#28a745"}),
                            html.Div(id="batch-phase-results", style={"fontSize":"12px"})
                        ])
                    ])
                ])
            ], style={"border":"1px solid #333","borderRadius":"8px","padding":"12px","marginTop":"12px"})
        ])
        
        return enhanced_controls

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

# --------- 增強分析 Callback：全局參數狀態更新 ---------
@app.callback(
    Output("enhanced-global-status", "children"),
    [
        Input("global-apply-switch", "value"),
        Input("risk-cap-input", "value"),
        Input("atr-ratio-threshold", "value"),
        Input("force-valve-trigger", "value")
    ]
)
def update_enhanced_global_status(global_apply, risk_cap, atr_ratio, force_trigger):
    """更新增強分析頁面的全局參數套用狀態"""
    if not global_apply:
        return html.Div([
            html.Small("🔴 全局參數套用未啟用", style={"color":"#dc3545","fontWeight":"bold","fontSize":"14px"}),
            html.Br(),
            html.Small("增強分析將使用頁面內建的參數設定", style={"color":"#666","fontSize":"12px"}),
            html.Br(),
            html.Small("💡 如需使用全局設定，請在側邊欄啟用「啟用全局參數套用」", style={"color":"#666","fontSize":"11px","fontStyle":"italic"})
        ])
    
    # 如果啟用全局參數套用
    status_color = "#28a745" if not force_trigger else "#dc3545"
    status_icon = "🟢" if not force_trigger else "🔴"
    status_text = "正常" if not force_trigger else "強制觸發"
    
    return html.Div([
        html.Small(f"{status_icon} 全局參數套用已啟用", style={"color":status_color,"fontWeight":"bold","fontSize":"14px"}),
        html.Br(),
        html.Small(f"風險閥門 CAP: {risk_cap}", style={"color":"#666","fontSize":"12px"}),
        html.Br(),
        html.Small(f"ATR比值門檻: {atr_ratio}", style={"color":"#666","fontSize":"12px"}),
        html.Br(),
        html.Small(f"狀態: {status_text}", style={"color":status_color,"fontSize":"12px"}),
        html.Br(),
        html.Small("💡 增強分析的風險閥門回測將優先使用這些全局設定", style={"color":"#28a745","fontSize":"11px","fontStyle":"italic"})
    ])

# --------- 增強分析 Callback：風險閥門回測（整合版） ---------
@app.callback(
    Output("rv-summary","children"),
    Output("rv-equity-chart","figure"),
    Output("rv-dd-chart","figure"),
    Input("run-rv","n_clicks"),
    State("rv-mode","value"),
    State("rv-cap","value"),
    State("rv-atr-mult","value"),
    State("enhanced-trades-cache","data"),
    # === 新增：讀取全局參數設定 ===
    State("global-apply-switch","value"),
    State("risk-cap-input","value"),
    State("atr-ratio-threshold","value"),
    # === 新增：讀取全局套用數據源 ===
    State("backtest-store","data"),
    prevent_initial_call=True
)
def _run_rv(n_clicks, mode, cap_level, atr_mult, cache, global_apply, global_risk_cap, global_atr_ratio, backtest_data=None):
    if not n_clicks or not cache:
        return "請先載入策略資料", no_update, no_update

    # === 修正：優先使用全局參數設定 ===
    if global_apply:
        # 如果啟用全局參數套用，優先使用全局設定
        effective_cap = global_risk_cap if global_risk_cap is not None else cap_level
        effective_atr_ratio = global_atr_ratio if global_atr_ratio is not None else atr_mult
        logger.info(f"增強分析使用全局參數：CAP={effective_cap}, ATR比值門檻={effective_atr_ratio}")
        
        # === 新增：詳細的參數對比日誌 ===
        logger.info(f"=== 增強分析參數對比 ===")
        logger.info(f"全局設定：CAP={global_risk_cap}, ATR比值門檻={global_atr_ratio}")
        logger.info(f"頁面設定：CAP={cap_level}, ATR比值門檻={atr_mult}")
        logger.info(f"最終使用：CAP={effective_cap}, ATR比值門檻={effective_atr_ratio}")
        
    else:
        # 否則使用增強分析頁面的設定
        effective_cap = cap_level
        effective_atr_ratio = atr_mult
        logger.info(f"增強分析使用頁面參數：CAP={effective_cap}, ATR比值門檻={effective_atr_ratio}")
    
    # === 整合：使用與全局套用相同的數據源 ===
    logger.info(f"=== 數據驗證 ===")
    
    # 優先使用全局套用的數據源，確保一致性
    if global_apply and backtest_data:
        # 從 backtest-store 獲取數據，與全局套用保持一致
        results = backtest_data.get("results", {})
        if results:
            # 找到對應的策略結果
            strategy_name = cache.get("strategy") if cache else None
            if strategy_name and strategy_name in results:
                result = results[strategy_name]
                df_raw = df_from_pack(backtest_data.get("df_raw"))
                daily_state = df_from_pack(result.get("daily_state_std") or result.get("daily_state"))
                logger.info(f"使用全局套用數據源: {strategy_name}")
            else:
                # 回退到快取數據
                df_raw = df_from_pack(cache.get("df_raw"))
                daily_state = df_from_pack(cache.get("daily_state"))
                logger.info("回退到快取數據源")
        else:
            # 回退到快取數據
            df_raw = df_from_pack(cache.get("df_raw"))
            daily_state = df_from_pack(cache.get("daily_state"))
            logger.info("回退到快取數據源")
    else:
        # 使用快取數據
        df_raw = df_from_pack(cache.get("df_raw"))
        daily_state = df_from_pack(cache.get("daily_state"))
        logger.info("使用快取數據源")
    
    # === 新增：數據一致性檢查 ===
    if global_apply and backtest_data:
        logger.info("=== 數據一致性檢查 ===")
        # 檢查與全局套用數據的一致性
        global_df_raw = df_from_pack(backtest_data.get("df_raw"))
        if global_df_raw is not None and df_raw is not None:
            if len(global_df_raw) == len(df_raw):
                logger.info(f"✅ 數據長度一致: {len(df_raw)}")
            else:
                logger.warning(f"⚠️  數據長度不一致: 全局={len(global_df_raw)}, 增強分析={len(df_raw)}")
        
        if daily_state is not None:
            logger.info(f"✅ daily_state 載入成功: {len(daily_state)} 行")
        else:
            logger.warning("⚠️  daily_state 載入失敗")
    
    # === 原有數據驗證日誌 ===
    logger.info(f"df_raw 形狀: {df_raw.shape if df_raw is not None else 'None'}")
    logger.info(f"daily_state 形狀: {daily_state.shape if daily_state is not None else 'None'}")
    if daily_state is not None:
        logger.info(f"daily_state 欄位: {list(daily_state.columns)}")
        logger.info(f"daily_state 索引範圍: {daily_state.index.min()} 到 {daily_state.index.max()}")
        if "w" in daily_state.columns:
            logger.info(f"權重欄位統計: 最小值={daily_state['w'].min():.4f}, 最大值={daily_state['w'].max():.4f}, 平均值={daily_state['w'].mean():.4f}")
    
    if df_raw is None or df_raw.empty:
        return "找不到股價資料", no_update, no_update
    
    if daily_state is None or daily_state.empty:
        return "找不到 daily_state（每日資產/權重）", no_update, no_update

    # 欄名對齊
    c_open = "open" if "open" in df_raw.columns else _first_col(df_raw, ["Open","開盤價"])
    c_close = "close" if "close" in df_raw.columns else _first_col(df_raw, ["Close","收盤價"])
    c_high  = "high" if "high" in df_raw.columns else _first_col(df_raw, ["High","最高價"])
    c_low   = "low"  if "low"  in df_raw.columns else _first_col(df_raw, ["Low","最低價"])

    if c_open is None or c_close is None:
        return "股價資料缺少 open/close 欄位", no_update, no_update

    open_px = pd.to_numeric(df_raw[c_open], errors="coerce").dropna()
    open_px.index = pd.to_datetime(df_raw.index)

    # 權重取自 daily_state
    if "w" not in daily_state.columns:
        return "daily_state 缺少權重欄位 'w'", no_update, no_update
    
    w = daily_state["w"].astype(float).reindex(open_px.index).ffill().fillna(0.0)

    # 成本參數（使用 SSS_EnsembleTab 預設）
    cost = None

    # 基準：用 df_raw 當基準（即可），函式能在無高低價時回退
    bench = pd.DataFrame({
        "收盤價": pd.to_numeric(df_raw[c_close], errors="coerce"),
    }, index=pd.to_datetime(df_raw.index))
    if c_high and c_low:
        bench["最高價"] = pd.to_numeric(df_raw[c_high], errors="coerce")
        bench["最低價"] = pd.to_numeric(df_raw[c_low], errors="coerce")

    # 需要用到 SSS_EnsembleTab 內新加的函式
    try:
        from SSS_EnsembleTab import risk_valve_backtest
        # === 增強分析風險閥門：確保參數一致性 (2025/08/20) ===
        enhanced_valve_params = {
            "open_px": open_px, 
            "w": w, 
            "cost": cost, 
            "benchmark_df": bench,
            "mode": mode, 
            "cap_level": float(effective_cap),  # === 修正：使用有效參數 ===
            "slope20_thresh": 0.0, 
            "slope60_thresh": 0.0,
            "atr_win": 20, 
            "atr_ref_win": 60, 
            "atr_ratio_mult": float(effective_atr_ratio),  # === 修正：使用有效參數 ===
            "use_slopes": True, 
            "slope_method": "polyfit", 
            "atr_cmp": "gt"
        }
        
        # 記錄增強分析風險閥門配置
        logger.info(f"[Enhanced] 風險閥門配置: cap_level={enhanced_valve_params['cap_level']}, atr_ratio_mult={enhanced_valve_params['atr_ratio_mult']}")
        
        out = risk_valve_backtest(**enhanced_valve_params)
    except Exception as e:
        return f"風險閥門回測執行失敗: {e}", no_update, no_update

    m = out["metrics"]
    
    # 計算風險觸發天數
    sig = out["signals"]["risk_trigger"]
    trigger_days = int(sig.fillna(False).sum())
    
    # === 修正：顯示實際使用的參數 ===
    summary = html.Div([
        html.Code(f"PF: 原始 {m['pf_orig']:.2f} → 閥門 {m['pf_valve']:.2f}"), html.Br(),
        html.Code(f"MDD: 原始 {m['mdd_orig']:.2%} → 閥門 {m['mdd_valve']:.2%}"), html.Br(),
        html.Code(f"右尾總和(>P90 正報酬): 原始 {m['right_tail_sum_orig']:.2f} → 閥門 {m['right_tail_sum_valve']:.2f} (↓{m['right_tail_reduction']:.2f})"), html.Br(),
        html.Code(f"風險觸發天數：{trigger_days} 天"), html.Br(),
        html.Code(f"使用參數：CAP={effective_cap}, ATR比值門檻={effective_atr_ratio}"), html.Br(),
        html.Code(f"參數來源：{'全局設定' if global_apply else '頁面設定'}", style={"color": "#28a745" if global_apply else "#ffc107"})
    ])

    # 繪圖：兩版權益與回撤
    import plotly.graph_objects as go
    eq1 = out["daily_state_orig"]["equity"]
    eq2 = out["daily_state_valve"]["equity"]
    dd1 = eq1/eq1.cummax()-1
    dd2 = eq2/eq2.cummax()-1

    palette = {
        "orig":  {"color": "#1f77b4", "dash": "solid"},
        "valve": {"color": "#ff7f0e", "dash": "dot"},
    }

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=eq1.index, y=eq1, name="原始",
        mode="lines", line=dict(color=palette["orig"]["color"], width=2, dash=palette["orig"]["dash"]),
        legendgroup="equity"
    ))
    fig_eq.add_trace(go.Scatter(
        x=eq2.index, y=eq2, name="閥門",
        mode="lines", line=dict(color=palette["valve"]["color"], width=2, dash=palette["valve"]["dash"]),
        legendgroup="equity"
    ))
    fig_eq.update_layout(title="權益曲線（Open→Open）", legend_orientation="h")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd1.index, y=dd1, name="原始",
        mode="lines", line=dict(color=palette["orig"]["color"], width=2, dash=palette["orig"]["dash"]),
        legendgroup="dd"
    ))
    fig_dd.add_trace(go.Scatter(
        x=dd2.index, y=dd2, name="閥門",
        mode="lines", line=dict(color=palette["valve"]["color"], width=2, dash=palette["valve"]["dash"]),
        legendgroup="dd"
    ))
    fig_dd.update_layout(title="回撤曲線", legend_orientation="h", yaxis_tickformat=".0%")

    return summary, fig_eq, fig_dd

# --------- 增強分析 Callback：數據比對報告 ---------
@app.callback(
    Output("data-comparison-output", "children"),
    Output("data-comparison-csv", "data"),
    Input("export-data-comparison", "n_clicks"),
    State("enhanced-trades-cache", "data"),
    State("backtest-store", "data"),
    State("global-apply-switch", "value"),
    State("risk-cap-input", "value"),
    State("atr-ratio-threshold", "value"),
    State("rv-cap", "value"),
    State("rv-atr-mult", "value"),
    prevent_initial_call=True
)
def generate_data_comparison_report(n_clicks, cache, backtest_data, global_apply, global_cap, global_atr, page_cap, page_atr):
    """生成數據比對報告，診斷全局套用與強化分析結果不同的問題 - 增強版 (2025/08/20)"""
    if not n_clicks:
        return "請點擊按鈕生成報告", no_update
    
    logger.info(f"=== 生成增強數據比對報告 ===")
    
    # 收集參數資訊
    param_info = {
        "全局參數套用": "啟用" if global_apply else "未啟用",
        "全局風險閥門CAP": global_cap,
        "全局ATR比值門檻": global_atr,
        "頁面風險閥門CAP": page_cap,
        "頁面ATR比值門檻": page_atr,
        "最終使用CAP": global_cap if global_apply else page_cap,
        "最終使用ATR比值門檻": global_atr if global_apply else page_atr,
        "參數差異分析": "CAP差異={}, ATR差異={}".format(
            abs((global_cap or 0) - (page_cap or 0)), 
            abs((global_atr or 0) - (page_atr or 0))
        )
    }
    
    # 收集數據資訊
    data_info = {}
    
    if cache:
        df_raw = df_from_pack(cache.get("df_raw"))
        daily_state = df_from_pack(cache.get("daily_state"))
        trade_data = df_from_pack(cache.get("trade_data"))
        weight_curve = df_from_pack(cache.get("weight_curve"))
        
        data_info["enhanced_cache"] = {
            "df_raw_shape": df_raw.shape if df_raw is not None else None,
            "daily_state_shape": daily_state.shape if daily_state is not None else None,
            "trade_data_shape": trade_data.shape if trade_data is not None else None,
            "weight_curve_shape": weight_curve.shape if weight_curve is not None else None,
            "daily_state_columns": list(daily_state.columns) if daily_state is not None else None,
            "daily_state_index_range": f"{daily_state.index.min()} 到 {daily_state.index.max()}" if daily_state is not None and not daily_state.empty else None
        }
        
        if daily_state is not None and "w" in daily_state.columns:
            data_info["enhanced_cache"]["weight_stats"] = {
                "min": float(daily_state["w"].min()),
                "max": float(daily_state["w"].max()),
                "mean": float(daily_state["w"].mean()),
                "std": float(daily_state["w"].std())
            }
    
    if backtest_data and backtest_data.get("results"):
        results = backtest_data["results"]
        data_info["backtest_store"] = {
            "available_strategies": list(results.keys()),
            "results_count": len(results)
        }
        
        # 選擇第一個策略進行詳細分析
        if results:
            first_strategy = list(results.keys())[0]
            result = results[first_strategy]
            
            data_info["backtest_store"]["first_strategy"] = {
                "name": first_strategy,
                "has_daily_state": result.get("daily_state") is not None,
                "has_daily_state_std": result.get("daily_state_std") is not None,
                "has_weight_curve": result.get("weight_curve") is not None,
                "valve_info": result.get("valve", {})
            }
    
    # 生成報告
    report_lines = []
    report_lines.append("=== 數據比對報告 ===")
    report_lines.append("")
    
    # 參數部分
    report_lines.append("📊 參數設定:")
    for key, value in param_info.items():
        report_lines.append(f"  {key}: {value}")
    report_lines.append("")
    
    # 數據部分
    report_lines.append("📈 數據狀態:")
    if "enhanced_cache" in data_info:
        report_lines.append("  Enhanced Cache:")
        for key, value in data_info["enhanced_cache"].items():
            report_lines.append(f"    {key}: {value}")
        report_lines.append("")
    
    if "backtest_store" in data_info:
        report_lines.append("  Backtest Store:")
        for key, value in data_info["backtest_store"].items():
            report_lines.append(f"    {key}: {value}")
        report_lines.append("")
    
    # 增強診斷建議 (2025/08/20)
    report_lines.append("🔍 詳細診斷建議:")
    
    # 參數一致性檢查
    if global_apply:
        cap_diff = abs((global_cap or 0) - (page_cap or 0))
        atr_diff = abs((global_atr or 0) - (page_atr or 0))
        if cap_diff > 0.001 or atr_diff > 0.001:
            report_lines.append(f"  ⚠️  全局與頁面參數差異: CAP差異={cap_diff:.4f}, ATR差異={atr_diff:.4f}")
            report_lines.append("      → 建議檢查 UI 介面的參數同步機制")
        else:
            report_lines.append("  ✅ 全局參數與頁面參數一致")
    else:
        report_lines.append("  ℹ️  未啟用全局參數套用，使用頁面參數")
        report_lines.append("      → 確認是否需要啟用全局套用以保持一致性")
    
    # 數據完整性檢查
    enhanced_has_data = "enhanced_cache" in data_info and data_info["enhanced_cache"]["daily_state_shape"]
    backtest_has_data = "backtest_store" in data_info and data_info["backtest_store"]["results_count"] > 0
    
    if enhanced_has_data:
        report_lines.append("  ✅ Enhanced Cache 有數據")
        if "weight_stats" in data_info["enhanced_cache"]:
            ws = data_info["enhanced_cache"]["weight_stats"]
            report_lines.append(f"      權重範圍: {ws['min']:.4f} ~ {ws['max']:.4f}, 均值: {ws['mean']:.4f}")
    else:
        report_lines.append("  ❌ Enhanced Cache 無數據")
        report_lines.append("      → 可能需要重新執行增強分析")
    
    if backtest_has_data:
        report_lines.append("  ✅ Backtest Store 有結果")
    else:
        report_lines.append("  ❌ Backtest Store 無結果")
        report_lines.append("      → 可能需要重新執行回測分析")
    
    # 風險閥門邏輯檢查
    effective_cap = global_cap if global_apply else page_cap
    effective_atr = global_atr if global_apply else page_atr
    
    report_lines.append("  🔧 風險閥門配置:")
    report_lines.append(f"      有效CAP值: {effective_cap}")
    report_lines.append(f"      有效ATR門檻: {effective_atr}")
    
    if effective_cap and effective_cap < 0.1:
        report_lines.append("      ⚠️  CAP值過低，可能造成過度保守")
    if effective_atr and effective_atr > 3.0:
        report_lines.append("      ⚠️  ATR門檻過高，可能很少觸發")
    
    # 一致性檢查總結
    consistency_issues = []
    if global_apply and (cap_diff > 0.001 or atr_diff > 0.001):
        consistency_issues.append("參數不一致")
    if not enhanced_has_data:
        consistency_issues.append("Enhanced Cache缺失")
    if not backtest_has_data:
        consistency_issues.append("Backtest Store缺失")
    
    if consistency_issues:
        report_lines.append(f"  🚨 發現一致性問題: {', '.join(consistency_issues)}")
        report_lines.append("      建議優先解決這些問題以確保分析結果一致性")
    else:
        report_lines.append("  ✅ 未發現明顯一致性問題")
    
    # 生成 CSV 數據
    csv_data = []
    for key, value in param_info.items():
        csv_data.append({"項目": key, "數值": str(value)})
    
    csv_data.append({"項目": "", "數值": ""})
    csv_data.append({"項目": "=== 數據狀態 ===", "數值": ""})
    
    if "enhanced_cache" in data_info:
        for key, value in data_info["enhanced_cache"].items():
            csv_data.append({"項目": f"Enhanced_{key}", "數值": str(value)})
    
    if "backtest_store" in data_info:
        for key, value in data_info["backtest_store"].items():
            csv_data.append({"項目": f"Backtest_{key}", "數值": str(value)})
    
    # 返回報告和 CSV 下載
    report_text = "\n".join(report_lines)
    csv_df = pd.DataFrame(csv_data)
    
    return report_text, dcc.send_data_frame(csv_df.to_csv, "data_comparison_report.csv", index=False)

def _first_col(df, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    return None

# --------- 增強分析 Callback：交易貢獻拆解（修正版） ---------
@app.callback(
    Output("phase-table", "children"),
    Input("run-phase", "n_clicks"),
    State("phase-min-gap", "value"),
    State("phase-cooldown", "value"),
    State("enhanced-trades-cache", "data"),
    State("theme-store", "data"),   # 若沒有 theme-store，這行與下方 theme 相關可移除
    prevent_initial_call=True
)
def _run_phase(n_clicks, min_gap, cooldown, cache, theme):
    import numpy as np
    from urllib.parse import quote as urlparse
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not cache:
        return html.Div("尚未載入回測結果", style={"color": "#ffb703"})

    # 從快取還原資料
    trade_df = df_from_pack(cache.get("trade_data"))
    daily_state = df_from_pack(cache.get("daily_state"))
    
    if trade_df is None or trade_df.empty:
        return "找不到交易資料"
    
    if daily_state is None or daily_state.empty:
        return "找不到 daily_state（每日資產/權重）"
    
    if "equity" not in daily_state.columns:
        return "daily_state 缺少權益欄位 'equity'"

    equity = daily_state["equity"]

    # 呼叫你已寫好的分析函數
    try:
        from SSS_EnsembleTab import trade_contribution_by_phase
        table = trade_contribution_by_phase(trade_df, equity, min_gap, cooldown).copy()
    except Exception as e:
        return f"交易貢獻拆解執行失敗: {e}"

    if table.empty:
        return "無資料"

    # 數字欄位轉型
    num_cols = ["交易筆數","賣出報酬總和(%)","階段內MDD(%)","階段淨貢獻(%)"]
    for c in num_cols:
        if c in table.columns:
            table[c] = pd.to_numeric(table[c], errors="coerce")

    # ====== 總體 KPI ======
    avg_net = table["階段淨貢獻(%)"].mean() if "階段淨貢獻(%)" in table else np.nan
    avg_mdd = table["階段內MDD(%)"].mean() if "階段內MDD(%)" in table else np.nan
    succ_all = (table["階段淨貢獻(%)"] > 0).mean() if "階段淨貢獻(%)" in table else np.nan
    succ_acc = np.nan
    if "階段" in table.columns and "階段淨貢獻(%)" in table.columns:
        mask_acc = table["階段"].astype(str).str.contains("加碼", na=False)
        if mask_acc.any():
            succ_acc = (table.loc[mask_acc, "階段淨貢獻(%)"] > 0).mean()
    risk_eff = np.nan
    if pd.notna(avg_net) and pd.notna(avg_mdd) and avg_mdd != 0:
        risk_eff = avg_net / abs(avg_mdd)

    # ====== CSV 文字（給複製用；DataTable 另有內建下載）======
    csv_text = table.to_csv(index=False)
    csv_data_url = "data:text/csv;charset=utf-8," + urlparse(csv_text)

    # ====== 主題樣式（避免白底白字）======
    theme = theme or "theme-dark"
    if theme == "theme-dark":
        table_bg = "#1a1a1a"; cell_color = "#ffffff"
        header_bg = "#2a2a2a"; header_color = "#ffffff"; border = "#444444"
        accent_bg = "#243447"; accent_color = "#ffffff"
    elif theme == "theme-light":
        table_bg = "#ffffff"; cell_color = "#111111"
        header_bg = "#f2f2f2"; header_color = "#111111"; border = "#cccccc"
        accent_bg = "#eef2ff"; accent_color = "#111111"
    else:  # theme-blue
        table_bg = "#0b1e3a"; cell_color = "#ffe066"
        header_bg = "#12345b"; header_color = "#ffe066"; border = "#335577"
        accent_bg = "#12345b"; accent_color = "#ffe066"

    style_table = {
        "overflowX": "auto",
        "overflowY": "auto",
        "maxHeight": "70vh",
        "fontSize": "12px",
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": table_bg,
        "border": f"1px solid {border}",
        # 允許選取→可複製
        "userSelect": "text", "-webkit-user-select": "text",
        "-moz-user-select": "text", "-ms-user-select": "text",
    }
    style_cell = {
        "textAlign": "center",
        "padding": "8px",
        "minWidth": "80px",
        "backgroundColor": table_bg,
        "color": cell_color,
        "border": f"1px solid {border}",
        "whiteSpace": "normal",
        "height": "auto",
    }
    style_header = {
        "backgroundColor": header_bg,
        "color": header_color,
        "fontWeight": "bold",
        "textAlign": "center",
        "borderBottom": f"2px solid {border}",
    }

    # ====== 完整表格 ======
    # 注意：full_table 將在 ordered 變數定義後重新定義

    # ====== 易讀版（KPI + Top3 / Worst3）======
    def kpi(label, value):
        return html.Div([
            html.Div(label, style={"fontSize": "12px", "opacity": 0.8}),
            html.Div(value, style={"fontSize": "18px", "fontWeight": "bold"})
        ], style={
            "backgroundColor": accent_bg, "color": accent_color,
            "padding": "10px 14px", "borderRadius": "12px", "minWidth": "160px"
        })

    kpi_bar = html.Div([
        kpi("平均每段淨貢獻(%)", f"{avg_net:.2f}" if pd.notna(avg_net) else "—"),
        kpi("平均每段 MDD(%)", f"{avg_mdd:.2f}" if pd.notna(avg_mdd) else "—"),
        kpi("成功率(全部)", f"{succ_all*100:.1f}%" if pd.notna(succ_all) else "—"),
        kpi("成功率(加碼)", f"{succ_acc*100:.1f}%" if pd.notna(succ_acc) else "—"),
        kpi("風險效率", f"{risk_eff:.3f}" if pd.notna(risk_eff) else "—"),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"10px"})

    # ====== 分組 KPI：加碼 vs 減碼 ======
    def _group_metrics(mask):
        if {"階段淨貢獻(%)","階段內MDD(%)"}.issubset(table.columns):
            sub = table.loc[mask]
            if sub.empty:
                return None
            a_net = sub["階段淨貢獻(%)"].mean()
            a_mdd = sub["階段內MDD(%)"].mean()
            succ  = (sub["階段淨貢獻(%)"] > 0).mean()
            eff   = (a_net / abs(a_mdd)) if pd.notna(a_net) and pd.notna(a_mdd) and a_mdd != 0 else np.nan
            return {"count": int(len(sub)), "avg_net": a_net, "avg_mdd": a_mdd, "succ": succ, "eff": eff}
        return None

    def _fmt(val, pct=False, dec=2):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "—"
        return f"{val*100:.1f}%" if pct else f"{val:.{dec}f}"

    def group_row(title, m):
        return html.Div([
            html.Div(title, style={"fontWeight":"bold","marginRight":"12px","minWidth":"72px","alignSelf":"center"}),
            kpi("段數", f"{m['count']}" if m else "—"),
            kpi("平均淨貢獻(%)", _fmt(m['avg_net']) if m else "—"),
            kpi("平均MDD(%)",   _fmt(m['avg_mdd']) if m else "—"),
            kpi("成功率",        _fmt(m['succ'], pct=True) if m else "—"),
            kpi("風險效率",      _fmt(m['eff'],  dec=3) if m else "—"),
        ], style={"display":"flex","gap":"10px","flexWrap":"wrap","marginBottom":"8px"})

    acc_metrics = dis_metrics = None
    if "階段" in table.columns:
        mask_acc = table["階段"].astype(str).str.contains("加碼", na=False)
        mask_dis = table["階段"].astype(str).str.contains("減碼", na=False)
        acc_metrics = _group_metrics(mask_acc)
        dis_metrics = _group_metrics(mask_dis)

    group_section = html.Div([
        html.H6("分組 KPI（加碼 vs 減碼）", style={"margin":"8px 0 6px 0"}),
        group_row("加碼段", acc_metrics),
        group_row("減碼段", dis_metrics),
    ], style={"marginTop":"4px"})

    # ====== Top/Worst 來源切換（全部 / 只加碼 / 只減碼） ======
    source_selector = html.Div([
        html.Div("Top/Worst 來源", style={"marginRight":"8px", "alignSelf":"center"}),
        dcc.RadioItems(
            id="phase-source",
            options=[
                {"label": "全部",   "value": "all"},
                {"label": "加碼段", "value": "acc"},
                {"label": "減碼段", "value": "dis"},
            ],
            value="all",
            inline=True,
            inputStyle={"marginRight":"4px"},
            labelStyle={"marginRight":"12px"}
        )
    ], style={"display":"flex","gap":"6px","alignItems":"center","margin":"6px 0 8px 0"})

    # 欄位順序（完整表 & Top/Worst 共用）
    ordered = [c for c in ["階段","開始日期","結束日期","交易筆數",
                           "階段淨貢獻(%)","賣出報酬總和(%)","階段內MDD(%)","是否成功"] if c in table.columns]
    basis_col = "階段淨貢獻(%)" if "階段淨貢獻(%)" in table.columns else "賣出報酬總和(%)"

    # ====== 完整表格 ======
    full_table = dash_table.DataTable(
        id="phase-datatable",
        columns=[{"name": c, "id": c, "type": ("numeric" if c in num_cols else "text")} for c in ordered],
        data=table[ordered].to_dict("records"),
        # 分頁
        page_action="native",
        page_current=0,
        page_size=100,            # 預設每頁 100，若要改可在這裡
        # 互動
        sort_action="native",
        filter_action="native",
        # 下載
        export_format="csv",
        export_headers="display",
        # 複製
        cell_selectable=True,
        virtualization=False,     # 關閉虛擬化，避免複製時只複到可視區
        fixed_rows={"headers": True},
        style_table=style_table,
        style_cell=style_cell,
        style_header=style_header,
        css=[{
            "selector": ".dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner *",
            "rule": "user-select: text; -webkit-user-select: text; -moz-user-select: text; -ms-user-select: text;"
        }],
    )

    # ====== dcc.Store：提供 Top/Worst 動態 callback 使用 ======
    store = dcc.Store(id="phase-table-store", data={
        "records": table[ordered].to_dict("records"),
        "ordered": ordered,
        "basis": basis_col,
        "has_stage": "階段" in table.columns
    })

    # 預設（全部來源）先算一次，避免空畫面
    def _subset(src):
        df = table
        if "階段" not in df.columns:
            return df
        if src == "acc":
            return df[df["階段"].astype(str).str.contains("加碼", na=False)]
        if src == "dis":
            return df[df["階段"].astype(str).str.contains("減碼", na=False)]
        return df
    base = _subset("all")
    top3   = base.nlargest(3, basis_col) if basis_col in base else base.head(3)
    worst3 = base.nsmallest(3, basis_col) if basis_col in base else base.tail(3)

    def simple_table(df, tbl_id):
        return dash_table.DataTable(
            id=tbl_id,
            columns=[{"name": c, "id": c} for c in ordered],
            data=df[ordered].to_dict("records"),
            page_action="none",
            style_table=style_table, style_cell=style_cell, style_header=style_header
        )

    top3_table = simple_table(top3, "phase-top-table")
    worst3_table = simple_table(worst3, "phase-worst-table")

    # ====== Copy / Download 工具列 ======
    tools = html.Div([
        html.Button("複製全部（CSV）", id="phase-copy-btn",
                    style={"padding": "6px 10px", "borderRadius": "8px", "cursor": "pointer"}),
        dcc.Clipboard(target_id="phase-csv-text", title="Copy", style={"marginLeft": "6px"}),
        html.A("下載 CSV", href=csv_data_url, download="trade_contribution.csv",
               style={"marginLeft": "12px", "textDecoration": "none"})
    ], style={"display": "flex", "alignItems": "center", "gap": "4px", "marginBottom": "8px"})

    # 隱藏的 CSV 文字來源（給 Clipboard 用）
    csv_hidden = html.Pre(id="phase-csv-text", children=csv_text, style={"display": "none"})

    # ====== Tabs：易讀版 / 完整表格 ======
    tabs = dcc.Tabs(id="phase-tabs", value="summary", children=[
        dcc.Tab(label="易讀版", value="summary", children=[
            kpi_bar,
            group_section,
            source_selector,
            html.H6("最賺的 3 段（依來源與排序欄）", style={"marginTop":"8px"}),
            top3_table,
            html.H6("最虧的 3 段（依來源與排序欄）", style={"marginTop":"16px"}),
            worst3_table
        ]),
        dcc.Tab(label="完整表格", value="full", children=[full_table]),
    ])

    return html.Div([tools, csv_hidden, store, tabs], style={"marginTop": "8px"})

# --------- 批量測試參數範圍 Callback ---------
@app.callback(
    Output("batch-phase-results", "children"),
    Input("run-batch-phase", "n_clicks"),
    State("enhanced-trades-cache", "data"),
    prevent_initial_call=True
)
def _run_batch_phase_test(n_clicks, cache):
    """批量測試1-24範圍的最小間距和冷卻期參數"""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not cache:
        return html.Div("尚未載入回測結果", style={"color": "#ffb703"})

    # 從快取還原資料
    trade_df = df_from_pack(cache.get("trade_data"))
    daily_state = df_from_pack(cache.get("daily_state"))
    
    if trade_df is None or trade_df.empty:
        return "找不到交易資料"
    
    if daily_state is None or daily_state.empty:
        return "找不到 daily_state（每日資產/權重）"
    
    if "equity" not in daily_state.columns:
        return "daily_state 缺少權益欄位 'equity'"

    equity = daily_state["equity"]
    
    # 檢查並準備交易資料格式
    debug_info = []
    debug_info.append(f"原始交易資料欄位: {list(trade_df.columns)}")
    debug_info.append(f"交易資料行數: {len(trade_df)}")
    debug_info.append(f"權益資料行數: {len(equity)}")
    
    # 檢查必要欄位並進行轉換
    required_mappings = {
        "date": ["date", "trade_date", "交易日期", "Date"],
        "type": ["type", "交易類型", "action", "side", "Type"],
        "w_before": ["w_before", "交易前權重", "weight_before", "weight_prev"],
        "w_after": ["w_after", "交易後權重", "weight_after", "weight_next"]
    }
    
    # 尋找對應的欄位
    found_columns = {}
    for target, possible_names in required_mappings.items():
        for name in possible_names:
            if name in trade_df.columns:
                found_columns[target] = name
                break
    
    debug_info.append(f"找到的欄位對應: {found_columns}")
    
    # 如果缺少必要欄位，嘗試創建
    if len(found_columns) < 4:
        debug_info.append("缺少必要欄位，嘗試創建...")
        
        # 嘗試從現有欄位推導
        if "weight_change" in trade_df.columns and "w_before" not in found_columns:
            # 如果有權重變化，嘗試重建前後權重
            trade_df = trade_df.copy()
            trade_df["w_before"] = 0.0
            trade_df["w_after"] = trade_df["weight_change"]
            found_columns["w_before"] = "w_before"
            found_columns["w_after"] = "w_after"
            debug_info.append("從 weight_change 創建 w_before 和 w_after")
        
        if "price" in trade_df.columns and "type" not in found_columns:
            # 如果有價格，假設為買入
            trade_df["type"] = "buy"
            found_columns["type"] = "type"
            debug_info.append("創建 type 欄位，預設為 buy")
    
    # 批量測試參數範圍 1-24
    results = []
    total_combinations = 24 * 24  # 576種組合
    
    try:
        from SSS_EnsembleTab import trade_contribution_by_phase
        
        # 進度顯示
        progress_div = html.Div([
            html.H6("正在執行批量測試...", style={"color": "#28a745"}),
            html.Div(f"測試範圍：最小間距 1-24 天，冷卻期 1-24 天", style={"fontSize": "12px", "color": "#666"}),
            html.Div(f"總組合數：{total_combinations}", style={"fontSize": "12px", "color": "#666"}),
            html.Div(id="batch-progress", children="開始測試...")
        ])
        
        # 執行批量測試
        batch_results = []
        debug_info = []
        
        # 先測試一個簡單的案例
        test_min_gap, test_cooldown = 1, 1
        try:
            debug_info.append(f"開始測試單一案例: min_gap={test_min_gap}, cooldown={test_cooldown}")
            
            # 檢查交易資料的權重欄位
            if "weight_change" in trade_df.columns:
                debug_info.append(f"找到 weight_change 欄位，範圍: {trade_df['weight_change'].min():.4f} ~ {trade_df['weight_change'].max():.4f}")
            
            # 檢查權益資料
            if len(equity) > 0:
                debug_info.append(f"權益資料範圍: {equity.min():.2f} ~ {equity.max():.2f}")
            
            table = trade_contribution_by_phase(trade_df, equity, test_min_gap, test_cooldown)
            debug_info.append(f"函數執行成功，返回表格大小: {table.shape}")
            debug_info.append(f"表格欄位: {list(table.columns)}")
            
            if not table.empty:
                debug_info.append(f"第一行資料: {table.iloc[0].to_dict()}")
                
                # 檢查是否有階段淨貢獻欄位
                if "階段淨貢獻(%)" in table.columns:
                    debug_info.append(f"階段淨貢獻欄位存在，非空值數量: {table['階段淨貢獻(%)'].notna().sum()}")
                    debug_info.append(f"階段淨貢獻範圍: {table['階段淨貢獻(%)'].min():.2f} ~ {table['階段淨貢獻(%)'].max():.2f}")
                else:
                    debug_info.append("缺少階段淨貢獻欄位")
                
                if "階段內MDD(%)" in table.columns:
                    debug_info.append(f"階段內MDD欄位存在，非空值數量: {table['階段內MDD(%)'].notna().sum()}")
                    debug_info.append(f"階段內MDD範圍: {table['階段內MDD(%)'].min():.2f} ~ {table['階段內MDD(%)'].max():.2f}")
                else:
                    debug_info.append("缺少階段內MDD欄位")
            else:
                debug_info.append("函數返回空表格")
                
        except Exception as e:
            import traceback
            debug_info.append(f"函數執行錯誤: {str(e)}")
            debug_info.append(f"錯誤詳情: {traceback.format_exc()}")
        
        # 如果單一測試成功，繼續批量測試
        if not table.empty and "階段淨貢獻(%)" in table.columns and "階段內MDD(%)" in table.columns:
            debug_info.append("單一測試成功，開始批量測試...")
            
            for min_gap in range(1, 25):
                for cooldown in range(1, 25):
                    try:
                        table = trade_contribution_by_phase(trade_df, equity, min_gap, cooldown)
                        
                        if not table.empty:
                            # 過濾掉摘要行（通常包含"統計摘要"字樣）
                            data_rows = table[~table["階段"].astype(str).str.contains("統計摘要", na=False)]
                            
                            if len(data_rows) == 0:
                                continue
                            
                            # 計算關鍵指標
                            avg_net = data_rows["階段淨貢獻(%)"].mean()
                            avg_mdd = data_rows["階段內MDD(%)"].mean()
                            succ_rate = (data_rows["階段淨貢獻(%)"] > 0).mean()
                            risk_eff = avg_net / abs(avg_mdd) if avg_mdd != 0 else 0
                            
                            batch_results.append({
                                "最小間距": min_gap,
                                "冷卻期": cooldown,
                                "平均淨貢獻(%)": round(avg_net, 2),
                                "平均MDD(%)": round(avg_mdd, 2),
                                "成功率(%)": round(succ_rate * 100, 1),
                                "風險效率": round(risk_eff, 3),
                                "階段數": len(data_rows)
                            })
                    except Exception as e:
                        # 記錄錯誤但繼續執行
                        continue
        else:
            debug_info.append("單一測試失敗，跳過批量測試")
        
        if not batch_results:
            # 顯示除錯資訊
            debug_html = html.Div([
                html.H6("除錯資訊", style={"color": "#dc3545", "marginTop": "16px"}),
                html.Div([html.Pre(info) for info in debug_info], style={"backgroundColor": "#f8f9fa", "padding": "10px", "borderRadius": "4px", "fontSize": "11px"})
            ])
            
            return html.Div([
                html.Div("批量測試完成，但無有效結果", style={"color": "#ffb703"}),
                html.Div("可能原因：", style={"marginTop": "8px", "color": "#666"}),
                html.Ul([
                    html.Li("交易資料格式不正確"),
                    html.Li("缺少必要的欄位（階段淨貢獻(%)、階段內MDD(%)）"),
                    html.Li("所有參數組合都無法產生有效階段"),
                    html.Li("函數執行時發生錯誤")
                ], style={"fontSize": "12px", "color": "#666"}),
                debug_html
            ])
        
        # 轉換為DataFrame並排序
        results_df = pd.DataFrame(batch_results)
        
        # 按風險效率排序（降序）
        results_df = results_df.sort_values("風險效率", ascending=False)
        
        # 生成CSV下載連結
        csv_text = results_df.to_csv(index=False)
        csv_data_url = "data:text/csv;charset=utf-8," + urlparse(csv_text)
        
        # 顯示前10名結果
        top10 = results_df.head(10)
        
        # 生成結果表格
        results_table = dash_table.DataTable(
            id="batch-results-table",
            columns=[{"name": c, "id": c} for c in results_df.columns],
            data=top10.to_dict("records"),
            page_action="none",
            style_table={"overflowX": "auto", "fontSize": "11px"},
            style_cell={"textAlign": "center", "padding": "4px", "minWidth": "60px"},
            style_header={"backgroundColor": "#28a745", "color": "white", "fontWeight": "bold"}
        )
        
        # 統計摘要
        summary_stats = html.Div([
            html.H6("批量測試摘要", style={"marginTop": "16px", "marginBottom": "8px", "color": "#28a745"}),
            html.Div(f"有效組合數：{len(results_df)} / {total_combinations}", style={"fontSize": "12px"}),
            html.Div(f"最佳風險效率：{results_df['風險效率'].max():.3f}", style={"fontSize": "12px"}),
            html.Div(f"最佳平均淨貢獻：{results_df['平均淨貢獻(%)'].max():.2f}%", style={"fontSize": "12px"}),
            html.Div(f"最佳成功率：{results_df['成功率(%)'].max():.1f}%", style={"fontSize": "12px"}),
            html.Div([
                html.Button("下載完整結果CSV", id="download-batch-csv", 
                           style={"backgroundColor": "#28a745", "color": "white", "border": "none", "padding": "8px 16px", "borderRadius": "4px", "cursor": "pointer"}),
                html.A("直接下載", href=csv_data_url, download="batch_phase_test_results.csv",
                       style={"marginLeft": "12px", "textDecoration": "none", "color": "#28a745"})
            ], style={"marginTop": "8px"})
        ])
        
        return html.Div([
            summary_stats,
            html.H6("前10名最佳參數組合（按風險效率排序）", style={"marginTop": "16px", "marginBottom": "8px"}),
            results_table
        ])
        
    except Exception as e:
        return html.Div(f"批量測試執行失敗: {str(e)}", style={"color": "#dc3545"})

# --- Gate analysis buttons until cache is ready ---
@app.callback(
    Output("run-rv", "disabled"),
    Output("run-phase", "disabled"),
    Output("run-batch-phase", "disabled"),
    Input("enhanced-trades-cache", "data"),
    prevent_initial_call=False
)
def _gate_analyze_buttons(cache):
    ready = bool(cache) and (
        (cache.get("trade_data") or cache.get("trade_df") or cache.get("trade_ledger") or cache.get("trade_ledger_std"))
        and (cache.get("daily_state") or cache.get("daily_state_std"))
    )
    disabled = not ready
    return disabled, disabled, disabled

# --------- 增強分析 Callback A：依 backtest-store 填滿策略選單 ---------
@app.callback(
    Output("enhanced-strategy-selector", "options"),
    Output("enhanced-strategy-selector", "value"),
    Input("backtest-store", "data"),
    prevent_initial_call=False
)
def _populate_enhanced_strategy_selector(bstore):
    """依 backtest-store 填滿策略選單，並自動選擇最佳策略"""
    if not bstore:
        return [], None
    
    results = bstore.get("results", {})
    if not results:
        return [], None
    
    # 策略評分：ledger_std > ledger > trade_df
    strategy_scores = []
    for strategy_name, result in results.items():
        score = 0
        if result.get("trade_ledger_std"):
            score += 100  # 最高分：標準化交易流水帳
        elif result.get("trade_ledger"):
            score += 50   # 中分：原始交易流水帳
        elif result.get("trade_df"):
            score += 10   # 低分：交易明細
        
        # 額外加分：有 daily_state
        if result.get("daily_state") or result.get("daily_state_std"):
            score += 20
        
        strategy_scores.append((strategy_name, score))
    
    # 按分數排序
    strategy_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 生成選單選項
    options = [{"label": f"{name} (分數: {score})", "value": name} 
               for name, score in strategy_scores]
    
    # 自動選擇最高分策略
    auto_select = strategy_scores[0][0] if strategy_scores else None
    
    return options, auto_select

# --------- 增強分析 Callback B：載入選定策略到 enhanced-trades-cache ---------
@app.callback(
    Output("enhanced-trades-cache", "data"),
    Output("enhanced-load-status", "children"),
    Input("load-enhanced-strategy", "n_clicks"),
    State("enhanced-strategy-selector", "value"),
    State("backtest-store", "data"),
    prevent_initial_call=True
)
def _load_enhanced_strategy_to_cache(n_clicks, selected_strategy, bstore):
    """載入選定策略的回測結果到 enhanced-trades-cache"""
    if not n_clicks or not selected_strategy or not bstore:
        return no_update, "請選擇策略並點擊載入"
    
    results = bstore.get("results", {})
    if selected_strategy not in results:
        return no_update, f"找不到策略：{selected_strategy}"
    
    result = results[selected_strategy]
    
    # 優先順序：ledger_std > ledger > trade_df
    trade_data = None
    data_source = ""
    
    if result.get("trade_ledger_std"):
        trade_data = df_from_pack(result["trade_ledger_std"])
        data_source = "trade_ledger_std (標準化)"
    elif result.get("trade_ledger"):
        trade_data = df_from_pack(result["trade_ledger"])
        data_source = "trade_ledger (原始)"
    elif result.get("trade_df"):
        trade_data = df_from_pack(result["trade_df"])
        data_source = "trade_df (交易明細)"
    else:
        return no_update, "該策略無交易資料"
    
    # 標準化交易資料
    try:
        from sss_core.normalize import normalize_trades_for_ui as norm
        trade_data = norm(trade_data)
    except Exception:
        # 後備標準化方案
        if trade_data is not None and len(trade_data) > 0:
            trade_data = trade_data.copy()
            trade_data.columns = [str(c).lower() for c in trade_data.columns]
            
            # 確保有 trade_date 欄
            if "trade_date" not in trade_data.columns:
                if "date" in trade_data.columns:
                    trade_data["trade_date"] = pd.to_datetime(trade_data["date"], errors="coerce")
                elif isinstance(trade_data.index, pd.DatetimeIndex):
                    trade_data = trade_data.reset_index().rename(columns={"index": "trade_date"})
                else:
                    trade_data["trade_date"] = pd.to_datetime(trade_data["date"], errors="coerce")
            
            # 確保有 type 欄
            if "type" not in trade_data.columns:
                if "action" in trade_data.columns:
                    trade_data["type"] = trade_data["action"].astype(str).str.lower()
                elif "side" in trade_data.columns:
                    trade_data["type"] = trade_data["side"].astype(str).str.lower()
                else:
                    trade_data["type"] = "hold"
            
            # 確保有 price 欄
            if "price" not in trade_data.columns:
                for c in ["open", "price_open", "exec_price", "px", "close"]:
                    if c in trade_data.columns:
                        trade_data["price"] = trade_data[c]
                        break
                if "price" not in trade_data.columns:
                    trade_data["price"] = 0.0
    
    # 準備 daily_state - 若已套用閥門則優先使用調整後資料
    daily_state = None
    valve_info = result.get("valve", {})
    valve_on = bool(valve_info.get("applied", False))
    
    # 先用閥門後的日線（若有）
    if valve_on and result.get("daily_state"):
        daily_state = df_from_pack(result["daily_state"])
    elif result.get("daily_state_std"):
        daily_state = df_from_pack(result["daily_state_std"])
    elif result.get("daily_state"):
        daily_state = df_from_pack(result["daily_state"])
    elif result.get("daily_state_base"):
        daily_state = df_from_pack(result["daily_state_base"])
    else:
        daily_state = None
    
    # 準備 weight_curve 和閥門資訊
    weight_curve = None
    if result.get("weight_curve"):
        weight_curve = df_from_pack(result["weight_curve"])
    
    # 獲取閥門狀態資訊
    valve_info = result.get("valve", {})  # {"applied": bool, "cap": float, "atr_ratio": float or "N/A"}
    valve_on = bool(valve_info.get("applied", False))
    
    # 若閥門生效，保證分析端覆寫 w_series
    if valve_on and weight_curve is not None and daily_state is not None:
        ds = daily_state.copy()
        wc = weight_curve.copy()
        # 對齊時間索引；若 ds 有 'trade_date' 欄就 merge，否則以索引對齊
        if "trade_date" in ds.columns:
            ds["trade_date"] = pd.to_datetime(ds["trade_date"])
            wc = wc.rename("w").to_frame().reset_index().rename(columns={"index": "trade_date"})
            ds = ds.merge(wc, on="trade_date", how="left")
        else:
            # 以索引對齊
            ds.index = pd.to_datetime(ds.index)
            wc.index = pd.to_datetime(wc.index)
            # 修正：確保 wc 是 Series 並且正確對齊
            if isinstance(wc, pd.DataFrame):
                if "w" in wc.columns:
                    wc_series = wc["w"]
                else:
                    wc_series = wc.iloc[:, 0]  # 取第一列
            else:
                wc_series = wc
            ds["w"] = wc_series.reindex(ds.index).ffill().bfill()
        daily_state = ds
    
    # 準備 df_raw
    df_raw = None
    if bstore.get("df_raw"):
        try:
            df_raw = pd.read_json(bstore["df_raw"], orient="split")
        except Exception:
            df_raw = pd.DataFrame()
    
    # ---- pack valve flags into cache ----
    cache_data = {
        "strategy": selected_strategy,
        "trade_data": pack_df(trade_data) if trade_data is not None else None,
        "daily_state": pack_df(daily_state) if daily_state is not None else None,
        "weight_curve": pack_df(weight_curve) if weight_curve is not None else None,
        "df_raw": pack_df(df_raw) if df_raw is not None else None,
        "valve": valve_info,
        "valve_applied": valve_on,
        "ensemble_params": result.get("ensemble_params", {}),
        "data_source": data_source,
        "timestamp": datetime.now().isoformat(),
        # ➌ 新增：baseline 版本一併放進快取
        "daily_state_base": result.get("daily_state_base"),
        "weight_curve_base": result.get("weight_curve_base"),
        "trade_ledger_base": result.get("trade_ledger_base"),
    }
    
    status_msg = f"✅ 已載入 {selected_strategy} ({data_source})"
    if daily_state is not None:
        status_msg += f"，包含 {len(daily_state)} 筆日線資料"
    if trade_data is not None:
        status_msg += f"，包含 {len(trade_data)} 筆交易"
    
    return cache_data, status_msg

# --------- 增強分析 Callback C：自動快取最佳策略 ---------
@app.callback(
    Output("enhanced-trades-cache", "data", allow_duplicate=True),
    Output("enhanced-load-status", "children", allow_duplicate=True),
    Input("backtest-store", "data"),
    State("enhanced-strategy-selector", "value"),
    prevent_initial_call='initial_duplicate'
)
def _auto_cache_best_strategy(bstore, current_selection):
    """回測完成後自動快取最佳策略"""
    if not bstore:
        return no_update, no_update
    
    results = bstore.get("results", {})
    if not results:
        return no_update, no_update
    
    # 如果已經有手動選擇，不覆蓋
    if current_selection:
        return no_update, no_update
    
    # 策略評分：ledger_std > ledger > trade_df
    strategy_scores = []
    for strategy_name, result in results.items():
        score = 0
        if result.get("trade_ledger_std"):
            score += 100  # 最高分：標準化交易流水帳
        elif result.get("trade_ledger"):
            score += 50   # 中分：原始交易流水帳
        elif result.get("trade_df"):
            score += 10   # 低分：交易明細
        
        # 額外加分：有 daily_state
        if result.get("daily_state") or result.get("daily_state_std"):
            score += 20
        
        strategy_scores.append((strategy_name, score))
    
    # 按分數排序，選擇最佳策略
    if not strategy_scores:
        return no_update, no_update
    
    strategy_scores.sort(key=lambda x: x[1], reverse=True)
    best_strategy = strategy_scores[0][0]
    best_result = results[best_strategy]
    
    # 準備交易資料（優先順序：ledger_std > ledger > trade_df）
    trade_data = None
    data_source = ""
    
    if best_result.get("trade_ledger_std"):
        trade_data = df_from_pack(best_result["trade_ledger_std"])
        data_source = "trade_ledger_std (標準化)"
    elif best_result.get("trade_ledger"):
        trade_data = df_from_pack(best_result["trade_ledger"])
        data_source = "trade_ledger (原始)"
    elif best_result.get("trade_df"):
        trade_data = df_from_pack(best_result["trade_df"])
        data_source = "trade_df (交易明細)"
    else:
        return no_update, no_update
    
    # 標準化交易資料
    try:
        from sss_core.normalize import normalize_trades_for_ui as norm
        trade_data = norm(trade_data)
    except Exception:
        # 後備標準化方案
        if trade_data is not None and len(trade_data) > 0:
            trade_data = trade_data.copy()
            trade_data.columns = [str(c).lower() for c in trade_data.columns]
            
            # 確保有 trade_date 欄
            if "trade_date" not in trade_data.columns:
                if "date" in trade_data.columns:
                    trade_data["trade_date"] = pd.to_datetime(trade_data["date"], errors="coerce")
                elif isinstance(trade_data.index, pd.DatetimeIndex):
                    trade_data = trade_data.reset_index().rename(columns={"index": "trade_date"})
                else:
                    trade_data["trade_date"] = pd.to_datetime(trade_data["date"], errors="coerce")
            
            # 確保有 type 欄
            if "type" not in trade_data.columns:
                if "action" in trade_data.columns:
                    trade_data["type"] = trade_data["action"].astype(str).str.lower()
                elif "side" in trade_data.columns:
                    trade_data["type"] = trade_data["side"].astype(str).str.lower()
                else:
                    trade_data["type"] = "hold"
            
            # 確保有 price 欄
            if "price" not in trade_data.columns:
                for c in ["open", "price_open", "exec_price", "px", "close"]:
                    if c in trade_data.columns:
                        trade_data["price"] = trade_data[c]
                        break
                if "price" not in trade_data.columns:
                    trade_data["price"] = 0.0
    
    # ---- choose daily_state consistently ----
    valve_info = best_result.get("valve", {}) or {}
    valve_on = bool(valve_info.get("applied", False))
    
    daily_state = None
    if valve_on and best_result.get("daily_state"):                 # adjusted first if valve is on
        daily_state = df_from_pack(best_result["daily_state"])
    elif best_result.get("daily_state_std"):
        daily_state = df_from_pack(best_result["daily_state_std"])
    elif best_result.get("daily_state"):
        daily_state = df_from_pack(best_result["daily_state"])
    elif best_result.get("daily_state_base"):
        daily_state = df_from_pack(best_result["daily_state_base"])
    
    # 準備 weight_curve 和閥門資訊
    weight_curve = None
    if best_result.get("weight_curve"):
        weight_curve = df_from_pack(best_result["weight_curve"])
    
    # 若閥門生效，保證分析端覆寫 w_series
    if valve_on and weight_curve is not None and daily_state is not None:
        ds = daily_state.copy()
        wc = weight_curve.copy()
        # 對齊時間索引；若 ds 有 'trade_date' 欄就 merge，否則以索引對齊
        if "trade_date" in ds.columns:
            ds["trade_date"] = pd.to_datetime(ds["trade_date"])
            wc = wc.rename("w").to_frame().reset_index().rename(columns={"index": "trade_date"})
            ds = ds.merge(wc, on="trade_date", how="left")
        else:
            # 以索引對齊
            ds.index = pd.to_datetime(ds.index)
            wc.index = pd.to_datetime(wc.index)
            # 修正：確保 wc 是 Series 並且正確對齊
            if isinstance(wc, pd.DataFrame):
                if "w" in wc.columns:
                    wc_series = wc["w"]
                else:
                    wc_series = wc.iloc[:, 0]  # 取第一列
            else:
                wc_series = wc
            ds["w"] = wc_series.reindex(ds.index).ffill().bfill()
        daily_state = ds
    
    # 準備 df_raw
    df_raw = None
    if bstore.get("df_raw"):
        try:
            df_raw = pd.read_json(bstore["df_raw"], orient="split")
        except Exception:
            df_raw = pd.DataFrame()
    
    # ---- pack valve flags into cache ----
    cache_data = {
        "strategy": best_strategy,
        "trade_data": pack_df(trade_data) if trade_data is not None else None,
        "daily_state": pack_df(daily_state) if daily_state is not None else None,
        "weight_curve": pack_df(weight_curve) if weight_curve is not None else None,
        "df_raw": pack_df(df_raw) if df_raw is not None else None,
        "valve": valve_info,
        "valve_applied": valve_on,
        "ensemble_params": best_result.get("ensemble_params", {}),
        "data_source": data_source,
        "timestamp": datetime.now().isoformat(),
        "auto_cached": True,
        # ➌ 新增：baseline 版本一併放進快取
        "daily_state_base": best_result.get("daily_state_base"),
        "weight_curve_base": best_result.get("weight_curve_base"),
        "trade_ledger_base": best_result.get("trade_ledger_base"),
    }
    
    status_msg = f"🔄 自動快取最佳策略：{best_strategy} ({data_source})"
    if daily_state is not None:
        status_msg += f"，包含 {len(daily_state)} 筆日線資料"
    if trade_data is not None:
        status_msg += f"，包含 {len(trade_data)} 筆交易"
    
    return cache_data, status_msg

# --------- 新增：風險-報酬地圖（Pareto Map）Callback ---------
@app.callback(
    Output("pareto-map-graph", "figure"),
    Output("pareto-map-status", "children"),
    Input("generate-pareto-map", "n_clicks"),
    State("enhanced-trades-cache", "data"),
    State("backtest-store", "data"),
    State("rv-mode", "value"),
    State("risk-cap-input", "value"),
    State("atr-ratio-threshold", "value"),
    prevent_initial_call=True
)
def generate_pareto_map(n_clicks, cache, backtest_data, rv_mode, risk_cap_value, atr_ratio_value):
    """生成風險-報酬地圖（Pareto Map）：掃描 cap 與 ATR(20)/ATR(60) 比值全組合"""
    logger.info(f"=== Pareto Map 生成開始 ===")
    logger.info(f"n_clicks: {n_clicks}")
    logger.info(f"cache 存在: {cache is not None}")
    logger.info(f"backtest_data 存在: {backtest_data is not None}")
    
    if not n_clicks:
        logger.warning("沒有點擊事件")
        return go.Figure(), "❌ 請點擊生成按鈕"
    
    # 優先使用 enhanced-trades-cache，如果沒有則嘗試從 backtest-store 生成
    if cache:
        logger.info("使用 enhanced-trades-cache 資料")
        df_raw = df_from_pack(cache.get("df_raw"))
        daily_state = df_from_pack(cache.get("daily_state"))
        data_source = "enhanced-trades-cache"
        logger.info(f"df_raw 形狀: {df_raw.shape if df_raw is not None else 'None'}")
        logger.info(f"daily_state 形狀: {daily_state.shape if daily_state is not None else 'None'}")
    elif backtest_data and backtest_data.get("results"):
        logger.info("使用 backtest-store 資料")
        results = backtest_data["results"]
        logger.info(f"可用策略: {list(results.keys())}")
        
        # 從 backtest-store 選擇第一個有 daily_state 的策略
        selected_strategy = None
        for strategy_name, result in results.items():
            logger.info(f"檢查策略 {strategy_name}: daily_state={result.get('daily_state') is not None}, daily_state_std={result.get('daily_state_std') is not None}")
            if result.get("daily_state") or result.get("daily_state_std"):
                selected_strategy = strategy_name
                logger.info(f"選擇策略: {selected_strategy}")
                break
        
        if not selected_strategy:
            logger.error("沒有找到包含 daily_state 的策略")
            return go.Figure(), "❌ 回測結果中沒有找到包含 daily_state 的策略"
        
        result = results[selected_strategy]
        daily_state = df_from_pack(result.get("daily_state") or result.get("daily_state_std"))
        df_raw = df_from_pack(backtest_data.get("df_raw"))
        data_source = f"backtest-store ({selected_strategy})"
        logger.info(f"df_raw 形狀: {df_raw.shape if df_raw is not None else 'None'}")
        logger.info(f"daily_state 形狀: {daily_state.shape if daily_state is not None else 'None'}")
    else:
        logger.error("沒有可用的資料來源")
        return go.Figure(), "❌ 請先執行回測，或於『🧠 從回測結果載入』載入策略"

    # 資料驗證
    logger.info("=== 資料驗證 ===")
    if df_raw is None or df_raw.empty:
        logger.error("df_raw 為空")
        return go.Figure(), "❌ 找不到股價資料 (df_raw)"
    if daily_state is None or daily_state.empty:
        logger.error("daily_state 為空")
        return go.Figure(), "❌ 找不到 daily_state（每日資產/權重）"
    
    # 資料不足時的行為對齊
    if len(daily_state) < 60:
        logger.warning("資料不足（<60天），已略過掃描")
        return go.Figure(), "⚠️ 資料不足（<60天），已略過掃描"
    
    logger.info(f"df_raw 欄位: {list(df_raw.columns)}")
    logger.info(f"daily_state 欄位: {list(daily_state.columns)}")

    # 欄名對齊
    c_open = "open" if "open" in df_raw.columns else _first_col(df_raw, ["Open","開盤價"])
    c_close = "close" if "close" in df_raw.columns else _first_col(df_raw, ["Close","收盤價"])
    c_high  = "high" if "high" in df_raw.columns else _first_col(df_raw, ["High","最高價"])
    c_low   = "low"  if "low"  in df_raw.columns else _first_col(df_raw, ["Low","最低價"])
    
    logger.info(f"欄名對齊結果: open={c_open}, close={c_close}, high={c_high}, low={c_low}")
    
    if c_open is None or c_close is None:
        logger.error("缺少必要的價格欄位")
        return go.Figure(), "❌ 股價資料缺少 open/close 欄位"

    # 準備輸入序列
    open_px = pd.to_numeric(df_raw[c_open], errors="coerce").dropna()
    open_px.index = pd.to_datetime(df_raw.index)
    
    # 取 open_px 後，準備 w（baseline 優先）
    ds_base = df_from_pack(cache.get("daily_state_base")) if cache else None
    wc_base = series_from_pack(cache.get("weight_curve_base")) if cache else None
    
    # 從 backtest-store 來的情況
    if ds_base is None and (not cache) and backtest_data and "results" in backtest_data:
        ds_base = df_from_pack(result.get("daily_state_base"))
        # 注意：weight_curve_base 也可能存在於 result
        try:
            wc_base = series_from_pack(result.get("weight_curve_base"))
        except Exception:
            wc_base = None
    
    # 以 baseline w 為優先；沒有再退回現行 daily_state['w']
    if ds_base is not None and (not ds_base.empty) and ("w" in ds_base.columns):
        w = pd.to_numeric(ds_base["w"], errors="coerce").reindex(open_px.index).ffill().fillna(0.0)
    elif wc_base is not None and (not wc_base.empty):
        w = pd.to_numeric(wc_base, errors="coerce").reindex(open_px.index).ffill().fillna(0.0)
    else:
        # 後備：沿用現行 daily_state（可能已被閥門壓過）
        if "w" not in daily_state.columns:
            return go.Figure(), "❌ daily_state 缺少權重欄位 'w'"
        w = pd.to_numeric(daily_state["w"], errors="coerce").reindex(open_px.index).ffill().fillna(0.0)

    bench = pd.DataFrame({
        "收盤價": pd.to_numeric(df_raw[c_close], errors="coerce"),
    }, index=pd.to_datetime(df_raw.index))
    if c_high and c_low:
        bench["最高價"] = pd.to_numeric(df_raw[c_high], errors="coerce")
        bench["最低價"] = pd.to_numeric(df_raw[c_low], errors="coerce")

    # ATR 樣本檢查（與狀態面板一致）
    logger.info("=== ATR 樣本檢查 ===")
    a20, a60 = calculate_atr(df_raw, 20), calculate_atr(df_raw, 60)
    if a20 is None or a60 is None or a20.dropna().size < 60 or a60.dropna().size < 60:
        logger.warning("ATR 樣本不足，回傳警示")
        return go.Figure(), "🟡 ATR 樣本不足（請拉長期間或改用更長資料）"
    
    # 掃描參數格點 - 把全局門檻置入格點
    logger.info("=== 開始掃描參數格點 ===")
    import numpy as np
    
    # 讀取當前設定
    cap_now = float(risk_cap_value) if risk_cap_value else 0.8
    atr_now = float(atr_ratio_value) if atr_ratio_value else 1.2
    
    # 基本格點
    caps = np.round(np.linspace(0.10, 1.00, 19), 2)
    atr_mults = np.round(np.linspace(1.00, 2.00, 21), 2)
    
    # 將全局設定植入格點（避免被內插忽略）
    if risk_cap_value is not None:
        caps = np.unique(np.r_[caps, float(risk_cap_value)])
    if atr_ratio_value is not None:
        atr_mults = np.unique(np.r_[atr_mults, float(atr_ratio_value)])
    
    logger.info(f"當前設定: cap={cap_now:.2f}, atr={atr_now:.2f}")
    logger.info(f"cap 範圍: {len(caps)} 個值，從 {caps[0]} 到 {caps[-1]}")
    logger.info(f"ATR 比值範圍: {len(atr_mults)} 個值，從 {atr_mults[0]} 到 {atr_mults[-1]}")
    logger.info(f"總組合數: {len(caps) * len(atr_mults)}")

    pareto_rows = []
    tried = 0
    succeeded = 0
    
    # 檢查是否可以匯入 risk_valve_backtest
    try:
        from SSS_EnsembleTab import risk_valve_backtest
        logger.info("成功匯入 risk_valve_backtest")
    except Exception as e:
        logger.error(f"匯入 risk_valve_backtest 失敗: {e}")
        return go.Figure(), f"❌ 無法匯入 risk_valve_backtest: {e}"
    
    logger.info("開始執行參數掃描...")
    for cap_level in caps:
        for atr_mult in atr_mults:
            tried += 1
            if tried % 50 == 0:  # 每50次記錄一次進度
                logger.info(f"進度: {tried}/{len(caps) * len(atr_mults)} (cap={cap_level:.2f}, atr={atr_mult:.2f})")
            
            try:
                out = risk_valve_backtest(
                    open_px=open_px, w=w, cost=None, benchmark_df=bench,
                    mode=(rv_mode or "cap"), cap_level=float(cap_level),
                    slope20_thresh=0.0, slope60_thresh=0.0,
                    atr_win=20, atr_ref_win=60, atr_ratio_mult=float(atr_mult),
                    use_slopes=True, slope_method="polyfit", atr_cmp="gt"
                )
                
                if not isinstance(out, dict) or "metrics" not in out:
                    logger.warning(f"cap={cap_level:.2f}, atr={atr_mult:.2f}: 回傳格式異常")
                    continue
                
                m = out["metrics"]
                sig = out["signals"]["risk_trigger"]
                trigger_days = int(sig.fillna(False).sum())

                # 取用『閥門』版本作為此組合的點位
                pf = float(m.get("pf_valve", np.nan))
                mdd = float(m.get("mdd_valve", np.nan))
                rt_sum_valve = float(m.get("right_tail_sum_valve", np.nan))
                rt_sum_orig = float(m.get("right_tail_sum_orig", np.nan)) if m.get("right_tail_sum_orig") is not None else np.nan
                rt_reduction = float(m.get("right_tail_reduction", np.nan)) if m.get("right_tail_reduction") is not None else (rt_sum_orig - rt_sum_valve if np.isfinite(rt_sum_orig) and np.isfinite(rt_sum_valve) else np.nan)

                # 收集一筆點資料
                pareto_rows.append({
                    "cap": cap_level,
                    "atr": atr_mult,
                    "pf": pf,
                    "max_drawdown": abs(mdd) if pd.notna(mdd) else np.nan,
                    "right_tail_sum_valve": rt_sum_valve,
                    "right_tail_sum_orig": rt_sum_orig,
                    "right_tail_reduction": rt_reduction,
                    "risk_trigger_days": trigger_days,
                    "label": f"cap={cap_level:.2f}, atr={atr_mult:.2f}"
                })
                succeeded += 1
                
                if succeeded % 20 == 0:  # 每20次成功記錄一次
                    logger.info(f"成功: {succeeded} 組 (cap={cap_level:.2f}, atr={atr_mult:.2f})")
                    
            except Exception as e:
                logger.warning(f"cap={cap_level:.2f}, atr={atr_mult:.2f} 執行失敗: {e}")
                continue

    logger.info(f"=== 掃描完成 ===")
    logger.info(f"嘗試: {tried} 組，成功: {succeeded} 組")
    
    if not pareto_rows:
        logger.error("沒有成功生成任何資料點")
        return go.Figure(), "❌ 無法從風險閥門回測的參數組合中取得資料"

    # 用 reduction 當顏色（越大=削越多右尾→越紅），符合『顏色越紅＝削太多右尾』
    logger.info("開始處理結果資料...")
    dfp = pd.DataFrame(pareto_rows).dropna(subset=["pf","max_drawdown","right_tail_reduction"]).reset_index(drop=True)
    logger.info(f"處理後資料點數: {len(dfp)}")
    logger.info(f"dfp 欄位: {list(dfp.columns)}")
    
    if dfp.empty:
        logger.error("處理後資料為空")
        return go.Figure(), "❌ 資料處理後為空，請檢查原始資料"
    
    logger.info("開始繪製圖表...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfp['max_drawdown'],
        y=dfp['pf'],
        mode='markers',
        marker=dict(
            size=np.clip(dfp['risk_trigger_days'] / 5.0, 6, 30),
            color=dfp['right_tail_reduction'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="右尾削減幅度")
        ),
        text=dfp['label'],
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "MDD: %{x:.2%}<br>" +
            "PF: %{y:.2f}<br>" +
            "右尾總和(閥門): %{customdata[0]:.2f}<br>" +
            "右尾總和(原始): %{customdata[1]:.2f}<br>" +
            "右尾削減: %{marker.color:.2f}<br>" +
            "風險觸發天數: %{marker.size:.0f} 天<br>" +
            "<extra></extra>"
        ),
        customdata=dfp[["right_tail_sum_valve","right_tail_sum_orig"]].values,
        name="cap-atr grid"
    ))
    
    # 加入「Current」標記點（當前全局設定）
    if cap_now in caps and atr_now in atr_mults:
        # 找到當前設定對應的點位
        current_point = dfp[(dfp['cap'] == cap_now) & (dfp['atr'] == atr_now)]
        if not current_point.empty:
            fig.add_trace(go.Scatter(
                x=current_point['max_drawdown'],
                y=current_point['pf'],
                mode='markers',
                marker=dict(
                    size=20,
                    symbol='star',
                    color='gold',
                    line=dict(color='black', width=2)
                ),
                text=f"Current: cap={cap_now:.2f}, atr={atr_now:.2f}",
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "MDD: %{x:.2%}<br>" +
                    "PF: %{y:.2f}<br>" +
                    "<extra></extra>"
                ),
                name="Current Settings"
            ))
    
    # 加入「Global」標記點（全局門檻設定）
    if risk_cap_value is not None and atr_ratio_value is not None:
        # 嘗試找到對應的掃描結果點位
        global_cap = float(risk_cap_value)
        global_atr = float(atr_ratio_value)
        global_point = dfp[(dfp['cap'] == global_cap) & (dfp['atr'] == global_atr)]
        
        if not global_point.empty:
            fig.add_trace(go.Scatter(
                x=global_point['max_drawdown'],
                y=global_point['pf'],
                mode='markers',
                marker=dict(
                    size=25,
                    symbol='diamond',
                    color='blue',
                    line=dict(color='white', width=2)
                ),
                text=f"Global: cap={global_cap:.2f}, atr={global_atr:.2f}",
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "MDD: %{x:.2%}<br>" +
                    "PF: %{y:.2f}<br>" +
                    "<extra></extra>"
                ),
                name="Global Setting"
            ))

    fig.update_layout(
        title={
            'text': f'風險-報酬地圖（Pareto Map）- {succeeded}/{tried} 組',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="最大回撤（愈左愈好）",
        yaxis_title="PF 獲利因子（愈上愈好）",
        xaxis=dict(tickformat=".1%", gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(r=120)
    )

    status_msg = f"✅ 成功生成：掃描 cap×ATR 比值 {succeeded}/{tried} 組。顏色=右尾調整幅度（紅=削減，藍=放大），大小=風險觸發天數。目前全局設定：cap={cap_now:.2f}, atr={atr_now:.2f}。資料來源：{data_source}"
    return fig, status_msg

def calculate_pareto_metrics(equity_curve, trade_df):
    """計算 Pareto Map 所需的指標"""
    try:
        # 初始化指標
        max_drawdown = 0.0
        pf = 1.0
        right_tail_loss = 0.0  # 預設：無調整
        risk_trigger_days = 50
        
        # 處理權益曲線數據
        if equity_curve is not None and not equity_curve.empty:
            # 確保權益曲線是 Series
            if isinstance(equity_curve, pd.DataFrame):
                if len(equity_curve.columns) == 1:
                    equity_curve = equity_curve.iloc[:, 0]
                else:
                    # 如果有多列，使用第一列
                    equity_curve = equity_curve.iloc[:, 0]
            
            # 計算日報酬率
            daily_returns = equity_curve.pct_change().dropna()
            
            if len(daily_returns) > 0:
                # 1. 最大回撤（愈左愈好）
                peak = equity_curve.expanding().max()
                drawdown = (equity_curve - peak) / peak
                max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0.0
                
                # 2. PF（獲利因子，愈上愈好）
                if trade_df is not None and not trade_df.empty and 'return' in trade_df.columns:
                    profits = trade_df[trade_df['return'] > 0]['return'].sum()
                    losses = abs(trade_df[trade_df['return'] < 0]['return'].sum())
                    pf = profits / losses if losses > 0 else (profits if profits > 0 else 1.0)
                else:
                    # 如果沒有交易數據，用年化報酬率代替
                    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
                    annual_return = total_return * (252 / len(daily_returns))
                    pf = 1 + annual_return  # 轉換為 PF 格式
                
                # 3. 右尾調整幅度（正值=削減右尾，負值=放大右尾，0=無調整）
                # 計算右尾風險：使用偏度和峰度來衡量
                positive_returns = daily_returns[daily_returns > 0]
                if len(positive_returns) > 0:
                    # 計算右尾的調整程度
                    if len(positive_returns) >= 10:
                        # 計算偏度
                        mean_ret = positive_returns.mean()
                        std_ret = positive_returns.std()
                        if std_ret > 0:
                            skewness = ((positive_returns - mean_ret) / std_ret) ** 3
                            skewness_mean = skewness.mean()
                            
                            # 將偏度轉換為 -1 到 1 的範圍
                            # 正偏度（右尾較長）= 負值（放大右尾）
                            # 負偏度（左尾較長）= 正值（削減右尾）
                            if skewness_mean > 0:
                                # 正偏度：右尾較長，表示放大右尾
                                right_tail_loss = -min(1.0, skewness_mean / 2)  # 轉換為 -1 到 0
                            else:
                                # 負偏度：左尾較長，表示削減右尾
                                right_tail_loss = min(1.0, abs(skewness_mean) / 2)  # 轉換為 0 到 1
                        else:
                            right_tail_loss = 0.0  # 無波動，無調整
                    else:
                        right_tail_loss = 0.0  # 數據不足，無調整
                else:
                    right_tail_loss = 1.0  # 沒有正報酬，完全削減右尾
                
                # 4. 風險觸發天數（點大小，越大＝管得越勤）
                if trade_df is not None and not trade_df.empty and 'trade_date' in trade_df.columns:
                    trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
                    # 計算有交易的天數
                    risk_trigger_days = len(trade_df['trade_date'].dt.date.unique())
                else:
                    # 如果沒有交易數據，用權益曲線的波動率來估計
                    volatility = daily_returns.std()
                    risk_trigger_days = min(100, int(volatility * 1000))  # 轉換為合理範圍
        else:
            # 如果沒有權益曲線，嘗試從交易數據計算基本指標
            if trade_df is not None and not trade_df.empty:
                # 從交易數據計算基本指標
                if 'return' in trade_df.columns:
                    profits = trade_df[trade_df['return'] > 0]['return'].sum()
                    losses = abs(trade_df[trade_df['return'] < 0]['return'].sum())
                    pf = profits / losses if losses > 0 else (profits if profits > 0 else 1.0)
                
                # 估算風險觸發天數
                if 'trade_date' in trade_df.columns:
                    trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
                    risk_trigger_days = len(trade_df['trade_date'].dt.date.unique())
                else:
                    risk_trigger_days = len(trade_df)
                
                # 如果沒有權益曲線，無法計算最大回撤和右尾損失，使用預設值
                max_drawdown = 0.1  # 預設 10% 回撤
                right_tail_loss = 0.0  # 預設：無調整
        
        return {
            'max_drawdown': max_drawdown,
            'pf': pf,
            'right_tail_loss': right_tail_loss,
            'risk_trigger_days': risk_trigger_days
        }
        
    except Exception as e:
        logger.exception(f"計算 Pareto 指標失敗：{e}")
        return None

def create_pareto_map(pareto_data):
    """創建風險-報酬地圖（Pareto Map）"""
    df = pd.DataFrame(pareto_data)
    
    # 創建散點圖
    fig = go.Figure()
    
    # 添加散點圖
    fig.add_trace(go.Scatter(
        x=df['max_drawdown'],
        y=df['pf'],
        mode='markers',
        marker=dict(
            size=df['risk_trigger_days'] / 10,  # 點大小：風險觸發天數
            color=df['right_tail_loss'],  # 顏色：右尾調整幅度
            colorscale='RdBu',  # 紅藍色階：紅色=削減右尾（正值），藍色=放大右尾（負值）
            cmin=-1,  # 最小值：-1（最大放大右尾）
            cmax=1,   # 最大值：1（最大削減右尾）
            mid=0,    # 中線：0（無調整）
            colorbar=dict(
                title="右尾調整幅度<br>（紅=削減，藍=放大）",
                titleside="right",
                tickformat=".2f",
                tickmode="array",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["放大右尾", "輕微放大", "無調整", "輕微削減", "削減右尾"]
            ),
            showscale=True
        ),
        text=df['strategy'],
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "最大回撤: %{x:.2%}<br>" +
            "PF: %{y:.2f}<br>" +
            "右尾調整: %{marker.color:.2f}<br>" +
            "風險觸發天數: %{marker.size:.0f}<br>" +
            "<extra></extra>"
        ),
        name="策略"
    ))
    
    # 添加 Pareto 邊界線（理想區域）
    # 找到最佳點（最小回撤，最大PF）
    best_idx = df['max_drawdown'].idxmin()
    best_dd = df.loc[best_idx, 'max_drawdown']
    best_pf = df.loc[best_idx, 'pf']
    
    # 添加理想區域的參考線
    fig.add_shape(
        type="rect",
        x0=0, y0=best_pf,
        x1=best_dd, y1=df['pf'].max() * 1.1,
        fillcolor="rgba(0,255,0,0.1)",
        line=dict(color="green", width=2, dash="dash"),
        name="理想區域"
    )
    
    # 添加文字標註
    fig.add_annotation(
        x=best_dd/2, y=best_pf + (df['pf'].max() - best_pf)/2,
        text="理想區域<br>（低回撤，高PF）",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="green",
        font=dict(size=12, color="green")
    )
    
    # 更新佈局
    fig.update_layout(
        title={
            'text': '風險-報酬地圖（Pareto Map）',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="最大回撤（愈左愈好）",
        yaxis_title="PF 獲利因子（愈上愈好）",
        xaxis=dict(
            tickformat=".1%",
            gridcolor="rgba(128,128,128,0.2)"
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.2)"
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        ),
        margin=dict(r=150)  # 為顏色條留出空間
    )
    
    # 添加網格
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    
    return fig

# --------- 新增：Pareto Map CSV 下載 Callback ---------
@app.callback(
    Output("pareto-csv-download", "data"),
    Input("download-pareto-csv", "n_clicks"),
    State("enhanced-trades-cache", "data"),
    State("backtest-store", "data"),
    State("rv-mode", "value"),
    prevent_initial_call=True
)
def download_pareto_csv(n_clicks, cache, backtest_data, rv_mode):
    """下載 Pareto Map 數據為 CSV 檔案"""
    if not n_clicks:
        return None
    
    try:
        # 優先使用 enhanced-trades-cache，如果沒有則嘗試從 backtest-store 生成
        if cache:
            df_raw = df_from_pack(cache.get("df_raw"))
            daily_state = df_from_pack(cache.get("daily_state"))
            data_source = "enhanced-trades-cache"
        elif backtest_data and backtest_data.get("results"):
            results = backtest_data["results"]
            selected_strategy = None
            for strategy_name, result in results.items():
                if result.get("daily_state") or result.get("daily_state_std"):
                    selected_strategy = strategy_name
                    break
            
            if not selected_strategy:
                return None
            
            result = results[selected_strategy]
            daily_state = df_from_pack(result.get("daily_state") or result.get("daily_state_std"))
            df_raw = df_from_pack(backtest_data.get("df_raw"))
            data_source = f"backtest-store ({selected_strategy})"
        else:
            return None
        
        # 資料驗證
        if df_raw is None or df_raw.empty or daily_state is None or daily_state.empty:
            return None
        
        # 欄名對齊
        c_open = "open" if "open" in df_raw.columns else _first_col(df_raw, ["Open","開盤價"])
        c_close = "close" if "close" in df_raw.columns else _first_col(df_raw, ["Close","收盤價"])
        c_high  = "high" if "high" in df_raw.columns else _first_col(df_raw, ["High","最高價"])
        c_low   = "low"  if "low"  in df_raw.columns else _first_col(df_raw, ["Low","最低價"])
        
        if c_open is None or c_close is None:
            return None
        
        # 準備輸入序列
        open_px = pd.to_numeric(df_raw[c_open], errors="coerce").dropna()
        open_px.index = pd.to_datetime(df_raw.index)
        
        # 取 open_px 後，準備 w（baseline 優先）
        ds_base = df_from_pack(cache.get("daily_state_base")) if cache else None
        wc_base = series_from_pack(cache.get("weight_curve_base")) if cache else None
        
        # 從 backtest-store 來的情況
        if ds_base is None and (not cache) and backtest_data and "results" in backtest_data:
            ds_base = df_from_pack(result.get("daily_state_base"))
            # 注意：weight_curve_base 也可能存在於 result
            try:
                wc_base = series_from_pack(result.get("weight_curve_base"))
            except Exception:
                wc_base = None
        
        # 以 baseline w 為優先；沒有再退回現行 daily_state['w']
        if ds_base is not None and (not ds_base.empty) and ("w" in ds_base.columns):
            w = pd.to_numeric(ds_base["w"], errors="coerce").reindex(open_px.index).ffill().fillna(0.0)
        elif wc_base is not None and (not wc_base.empty):
            w = pd.to_numeric(wc_base, errors="coerce").reindex(open_px.index).ffill().fillna(0.0)
        else:
            # 後備：沿用現行 daily_state（可能已被閥門壓過）
            if "w" not in daily_state.columns:
                return None
            w = pd.to_numeric(daily_state["w"], errors="coerce").reindex(open_px.index).ffill().fillna(0.0)
        
        bench = pd.DataFrame({
            "收盤價": pd.to_numeric(df_raw[c_close], errors="coerce"),
        }, index=pd.to_datetime(df_raw.index))
        if c_high and c_low:
            bench["最高價"] = pd.to_numeric(df_raw[c_high], errors="coerce")
            bench["最低價"] = pd.to_numeric(df_raw[c_low], errors="coerce")
        
        # 使用與 generate_pareto_map 相同的參數範圍和邏輯
        logger.info("=== 開始生成 Pareto Map 數據用於 CSV 下載 ===")
        caps = np.round(np.arange(0.10, 1.00 + 1e-9, 0.05), 2)
        atr_mults = np.round(np.arange(1.00, 2.00 + 1e-9, 0.05), 2)
        logger.info(f"cap 範圍: {len(caps)} 個值，從 {caps[0]} 到 {caps[-1]}")
        logger.info(f"ATR 比值範圍: {len(atr_mults)} 個值，從 {atr_mults[0]} 到 {atr_mults[-1]}")
        logger.info(f"總組合數: {len(caps) * len(atr_mults)}")
        
        pareto_data = []
        tried = 0
        succeeded = 0
        
        # 檢查是否可以匯入 risk_valve_backtest
        try:
            from SSS_EnsembleTab import risk_valve_backtest
            logger.info("成功匯入 risk_valve_backtest")
        except Exception as e:
            logger.error(f"匯入 risk_valve_backtest 失敗: {e}")
            return None
        
        logger.info("開始執行參數掃描...")
        for cap_level in caps:
            for atr_mult in atr_mults:
                tried += 1
                if tried % 50 == 0:  # 每50次記錄一次進度
                    logger.info(f"進度: {tried}/{len(caps) * len(atr_mults)} (cap={cap_level:.2f}, atr={atr_mult:.2f})")
                
                try:
                    out = risk_valve_backtest(
                        open_px=open_px, w=w, cost=None, benchmark_df=bench,
                        mode=(rv_mode or "cap"), cap_level=float(cap_level),
                        slope20_thresh=0.0, slope60_thresh=0.0,
                        atr_win=20, atr_ref_win=60, atr_ratio_mult=float(atr_mult),
                        use_slopes=True, slope_method="polyfit", atr_cmp="gt"
                    )
                    
                    if out and "metrics" in out:
                        m = out["metrics"]
                        
                        # 計算 Pareto 指標
                        equity_curve = out.get("daily_state_valve", {}).get("equity")
                        trade_df = None  # risk_valve_backtest 不直接提供交易記錄
                        
                        metrics = calculate_pareto_metrics(equity_curve, trade_df)
                        if metrics:
                            pareto_data.append({
                                'strategy': f'cap_{cap_level:.2f}_atr_{atr_mult:.2f}',
                                'cap': cap_level,
                                'atr_ratio': atr_mult,
                                'max_drawdown': metrics['max_drawdown'],
                                'pf': metrics['pf'],
                                'right_tail_loss': metrics['right_tail_loss'],
                                'risk_trigger_days': metrics['risk_trigger_days'],
                                'pf_orig': m.get('pf_orig', 0.0),
                                'pf_valve': m.get('pf_valve', 0.0),
                                'mdd_orig': m.get('mdd_orig', 0.0),
                                'mdd_valve': m.get('mdd_valve', 0.0),
                                'right_tail_reduction': m.get('right_tail_reduction', 0.0)
                            })
                            succeeded += 1
                            
                except Exception as e:
                    logger.debug(f"參數組合 cap={cap_level}, atr={atr_mult} 計算失敗: {e}")
                    continue
        
        if not pareto_data:
            logger.warning("沒有生成任何 Pareto 數據")
            return None
        
        logger.info(f"成功生成 {succeeded} 組 Pareto 數據")
        
        # 轉換為 DataFrame 並準備下載
        df_pareto = pd.DataFrame(pareto_data)
        
        # 添加時間戳記
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pareto_map_data_{timestamp}.csv"
        
        # 返回 CSV 下載數據
        return dcc.send_data_frame(
            df_pareto.to_csv,
            filename,
            index=False,
            encoding='utf-8-sig'  # 支援中文
        )
        
    except Exception as e:
        logger.exception(f"下載 Pareto Map CSV 失敗：{e}")
        return None

# --------- 新增：動態切換 Top/Worst 的 callback ---------
@app.callback(
    [dash.Output("phase-top-table", "data"),
     dash.Output("phase-worst-table", "data")],
    [dash.Input("phase-source", "value"),
     dash.Input("phase-table-store", "data")]
)
def _update_top_worst(src, store):
    if not store:
        raise dash.exceptions.PreventUpdate
    import pandas as pd
    records = store.get("records", [])
    ordered = store.get("ordered", [])
    basis   = store.get("basis", None)
    has_stage = store.get("has_stage", False)

    df = pd.DataFrame(records)
    # 來源過濾
    if has_stage and "階段" in df.columns:
        if src == "acc":
            df = df[df["階段"].astype(str).str.contains("加碼", na=False)]
        elif src == "dis":
            df = df[df["階段"].astype(str).str.contains("減碼", na=False)]

    if basis and basis in df.columns and not df.empty:
        top3   = df.nlargest(3, basis)
        worst3 = df.nsmallest(3, basis)
    else:
        top3   = df.head(3)
        worst3 = df.tail(3)

    return top3[ordered].to_dict("records"), worst3[ordered].to_dict("records")

if __name__ == '__main__':
    # 初始化日誌系統（只在實際運行 app 時）
    _initialize_app_logging()
    
    # 在主線程中執行啟動任務
    safe_startup()
    
    # 設置更安全的服務器配置
    app.run_server(
        debug=True, 
        host='127.0.0.2', 
        port=8051,
        threaded=True,
        use_reloader=False  # 避免重載器造成的線程問題
    ) 