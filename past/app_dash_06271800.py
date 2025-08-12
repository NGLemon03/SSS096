import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import json
import io
from dash.dependencies import ALL

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
smaa_sources = ["Self", "Factor (^TWII / 2412.TW)", "Factor (^TWII / 2414.TW)"]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# --------- Dash Layout ---------
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("參數設定"),
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
            html.Button("🚀 一鍵執行所有回測", id='run-btn', n_clicks=0),
            dcc.Loading(dcc.Store(id='backtest-store'), type="circle"),
        ], width=3),
        dbc.Col([
            dcc.Tabs(
                id='main-tabs',
                value='backtest',
                children=[
                    dcc.Tab(label="策略回測", value="backtest"),
                    dcc.Tab(label="所有策略買賣點比較", value="compare"),
                    dcc.Tab(label="各版本沿革紀錄", value="history")
                ]
            ),
            html.Div(id='tab-content')
        ], width=9)
    ])
], fluid=True)

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
    Input('run-btn', 'n_clicks'),
    State('ticker-dropdown', 'value'),
    State('start-date', 'value'),
    State('end-date', 'value'),
    State('discount-slider', 'value'),
    State('cooldown-bars', 'value'),
    State('bad-holding', 'value'),
    State('smaa-source-dropdown', 'value'),
    State('strategy-dropdown', 'value'),
    State({'type': 'param-input', 'param': ALL}, 'value'),
    State({'type': 'param-input', 'param': ALL}, 'id'),
)
def run_backtest(n_clicks, ticker, start_date, end_date, discount, cooldown, bad_holding, smaa_source, strategy, param_values, param_ids):
    if n_clicks == 0:
        return None
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
    State('strategy-dropdown', 'value')
)
def update_tab(data, tab, selected_strategy):
    if not data:
        return html.Div("請先執行回測")
    data = json.loads(data)
    results = data['results']
    df_raw = pd.read_json(io.StringIO(data['df_raw']))
    ticker = data['ticker']
    strategy_names = list(results.keys())
    if tab == "backtest":
        result = results.get(selected_strategy)
        if not result:
            return html.Div("該策略無回測結果")
        trade_df = pd.read_json(io.StringIO(result['trade_df']))
        if 'trade_date' in trade_df.columns:
            trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
        metrics = result.get('metrics', {})
        tooltip = f"{selected_strategy} 策略說明"
        param_display = {k: v for k, v in param_presets[selected_strategy].items() if k != "strategy_type"}
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
                            html.Div(label_map.get(k, k), style={"color": "#aaa", "font-size": "14px"}),
                            html.Div(txt, style={"font-weight": "bold", "font-size": "20px"})
                        ])
                    ], style={"background": "#1a1a1a", "border": "1px solid #444", "border-radius": "8px", "margin-bottom": "12px"})
                ], xs=12, sm=6, md=4, lg=3, style={"minWidth": "180px", "margin-bottom": "12px"})
            )
        fig1 = plot_stock_price(df_raw, trade_df, ticker)
        fig2 = plot_equity_cash(trade_df, df_raw)
        trade_table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in trade_df.columns],
            data=trade_df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'}
        )
        return html.Div([
            html.H4([
                f"回測策略: {selected_strategy} ",
                html.Span("ⓘ", title=tooltip, style={"cursor": "help", "color": "#888"})
            ]),
            html.Div(f"參數設定: {param_str}"),
            html.Br(),
            dbc.Row(metric_cards, style={"flex-wrap": "wrap"}),
            html.Br(),
            dcc.Graph(figure=fig1, config={'displayModeBar': True}),
            dcc.Graph(figure=fig2, config={'displayModeBar': True}),
            html.H5("交易明細"),
            trade_table
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
            xaxis_title='Date', yaxis_title='股價', template='plotly_white',
            legend=dict(
                x=1.05, y=1, xanchor='left', yanchor='top',
                bordercolor="Black", borderwidth=1, bgcolor="white",
                itemsizing='constant', orientation='v'
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
            dcc.Graph(figure=fig, config={'displayModeBar': True}),
            html.Hr(),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in comparison_data[0].keys()] if comparison_data else [],
                data=comparison_data,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'}
            )
        ])
    elif tab == "history":
        return html.Div([
            html.H4("各版本沿革紀錄"),
            dcc.Markdown(get_version_history_html(), dangerously_allow_html=True)
        ])
    else:
        return html.Div("請選擇頁籤")

if __name__ == '__main__':
    app.run_server(debug=True) 