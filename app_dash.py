import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import json
import io
from dash.dependencies import ALL

from SSSv095b2 import (
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
smaa_sources = ["Self", "Factor (^TWII / 2412.TW)", "Factor (^TWII / 2414.TW)"]

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
    ], className='header-controls'),
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
                    html.Label("SMAA 數據源"),
                    dcc.Dropdown(
                        id='smaa-source-dropdown',
                        options=[{'label': s, 'value': s} for s in smaa_sources],
                        value="Self"
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
                        dcc.Tab(label="各版本沿革紀錄", value="history")
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
        Input('smaa-source-dropdown', 'value'),
        Input('strategy-dropdown', 'value'),
        Input({'type': 'param-input', 'param': ALL}, 'value'),
        Input({'type': 'param-input', 'param': ALL}, 'id'),
    ],
    State('backtest-store', 'data')
)
def run_backtest(n_clicks, auto_run, ticker, start_date, end_date, discount, cooldown, bad_holding, smaa_source, strategy, param_values, param_ids, stored_data):
    ctx_trigger = ctx.triggered_id
    # 只在 auto-run 為 True 或按鈕被點擊時運算
    if not auto_run and ctx_trigger != 'run-btn':
        return stored_data
    params = {id['param']: v for id, v in zip(param_ids, param_values)}
    params['strategy_type'] = param_presets[strategy]['strategy_type']
    params['smaa_source'] = smaa_source
    df_raw, df_factor = load_data(ticker, start_date, end_date if end_date else None, smaa_source=smaa_source)
    results = {}
    for strat in strategy_names:
        strat_params = param_presets[strat].copy()
        strat_params.update(params if strat == strategy else {})
        strat_type = strat_params["strategy_type"]
        smaa_src = strat_params.get("smaa_source", "Self")
        if strat_type == 'ssma_turn':
            calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'signal_cooldown_days', 'quantile_win']
            ssma_params = {k: v for k, v in strat_params.items() if k in calc_keys}
            backtest_params = ssma_params.copy()
            backtest_params['stop_loss'] = strat_params.get('stop_loss', 0.0)
            df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(df_raw, df_factor, **ssma_params, smaa_source=smaa_src)
            if df_ind.empty:
                continue
            result = backtest_unified(df_ind, strat_type, backtest_params, buy_dates, sell_dates, discount=discount, trade_cooldown_bars=cooldown, bad_holding=bad_holding)
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
    df_raw_json = df_raw.to_json(date_format='iso')
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
            tooltip = f"{strategy} 策略說明"
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
                html.H5("交易明細"),
                dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in display_df.columns],
                    data=display_df.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center'},
                    id={'type': 'strategy-table', 'strategy': strategy}
                ),
                html.Div("**所有下單日皆為信號日的隔天（T+1），本表 signal_date=trade_date 代表信號日即下單日**", style={"fontWeight": "bold", "marginTop": "8px", "color": font_color}),
                html.Br(),
                html.Button("下載交易紀錄", id={'type': 'download-btn', 'strategy': strategy}),
                dcc.Download(id={'type': 'download-trade', 'strategy': strategy})
            ])
            
            strategy_tabs.append(dcc.Tab(label=strategy, value=f"strategy_{strategy}", children=strategy_content))
        
        # 創建策略回測的子頁籤容器
        return html.Div([
            html.H3("策略回測結果"),
            html.Div("選擇下方頁籤查看各策略的詳細回測結果"),
            html.Br(),
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
        comparison_data = []
        for strategy in strategy_names:
            result = results.get(strategy)
            if not result:
                continue
            metrics = result['metrics']
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
        return html.Div([
            dcc.Graph(figure=fig, config={'displayModeBar': True}, className='compare-graph'),
            html.Hr(),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in comparison_data[0].keys()] if comparison_data else [],
                data=comparison_data,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
                id='compare-table'
            )
        ])
    elif tab == "history":
        return html.Div([
            html.H4("各版本沿革紀錄"),
            dcc.Markdown(get_version_history_html(), dangerously_allow_html=True)
        ], className='version-history-panel')

# --------- 主題切換 ---------
@app.callback(
    Output('main-bg', 'className'),
    Output('theme-toggle', 'children'),
    Output('theme-store', 'data'),
    Input('theme-toggle', 'n_clicks'),
    State('theme-store', 'data')
)
def toggle_theme(n, theme):
    if n is None:
        return 'theme-dark', '🌑 深色主題', 'theme-dark'
    themes = ['theme-dark', 'theme-light', 'theme-blue']
    current_index = themes.index(theme)
    next_theme = themes[(current_index + 1) % len(themes)]
    return next_theme, get_theme_label(next_theme), next_theme

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

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050) 