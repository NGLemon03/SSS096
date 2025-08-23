# app_dash_modules/layout.py / 2025-08-23 03:07
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from .utils import get_version_history_html


def create_layout():
    return html.Div([
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
