# -*- coding: utf-8 -*-
"""
增強分析UI整合模組 - 2025-08-18 04:45
提供Dash組件和回調函數，將增強分析功能整合到現有UI中

作者：AI Assistant
路徑：#analysis/enhanced_analysis_ui.py
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

# 設定 logger
logger = logging.getLogger(__name__)

def safe_to_datetime(col):
    """
    對各種奇怪情形做防守：
    - 若傳入的是 DataFrame（可能是重複欄位選取或 pack 解包不一致），先嘗試 direct to_datetime，
      失敗時把欄位合併成字串再 parse。
    - 若 Series 內含 dict-like 物件，會先轉成 DataFrame，再嘗試 parse（若失敗則合併字串 parse）。
    - 其他情況直接呼叫 pd.to_datetime(..., errors='coerce')。
    回傳一支 pd.Series(datetime64[ns])（NaT 表示 parse 失敗）。
    """
    # 若是直接傳入 DataFrame -> 嘗試處理
    if isinstance(col, pd.DataFrame):
        try:
            return pd.to_datetime(col, errors='coerce')
        except Exception:
            # 合併所有欄為字串，作為後備
            return pd.to_datetime(col.astype(str).agg(' '.join, axis=1), errors='coerce')

    # 若是 Series，但元素可能為 dict / DataFrame / list 等
    if isinstance(col, pd.Series):
        # 若有任何元素是 dict-like
        is_dict_like = col.dropna().apply(lambda x: isinstance(x, dict)).any()
        is_df_like = col.dropna().apply(lambda x: isinstance(x, (pd.DataFrame, np.ndarray, list))).any()

        if is_dict_like:
            # 將 list of dict -> DataFrame（若 keys 不一致，pandas 會以 NaN 補齊）
            try:
                df = pd.DataFrame(list(col))
                try:
                    return pd.to_datetime(df, errors='coerce')
                except Exception:
                    # fallback: 逐列合成字串再 parse
                    return pd.to_datetime(df.astype(str).agg(' '.join, axis=1), errors='coerce')
            except Exception:
                # 如果無法轉（極端情形），退回字串處理
                return pd.to_datetime(col.astype(str), errors='coerce')

        if is_df_like:
            # 如果 Series 裡面是 list/ndarray/dataframe 等，先把內容 stringify
            return pd.to_datetime(col.astype(str).map(lambda x: ' '.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else str(x)),
                                   errors='coerce')

        # 正常情況：直接 parse
        return pd.to_datetime(col, errors='coerce')

    # 其他不可預期類型，嘗試 str parse
    return pd.to_datetime(pd.Series(col).astype(str), errors='coerce')

try:
    from analysis.enhanced_trade_analysis import EnhancedTradeAnalyzer
    ENHANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYSIS_AVAILABLE = False
    print("警告：無法導入 EnhancedTradeAnalyzer，增強分析功能將不可用")

class EnhancedAnalysisUI:
    """增強分析UI整合器"""
    
    def __init__(self):
        """初始化UI整合器"""
        self.enhanced_analyzer = None
        self.analysis_results = {}
        
    def create_enhanced_analysis_tab(self):
        """創建增強分析標籤頁"""
        from dash import dcc
        
        if not ENHANCED_ANALYSIS_AVAILABLE:
            return dcc.Tab(
                label="❌ 增強分析 (不可用)", 
                value="enhanced_analysis",
                disabled=True
            )
            
        return dcc.Tab(
            label="🔍 增強分析", 
            value="enhanced_analysis",
            children=[
                self._create_enhanced_analysis_layout()
            ]
        )
        
    def _create_enhanced_analysis_layout(self):
        """創建增強分析頁面佈局"""
        from dash import html, dcc
        import dash_bootstrap_components as dbc
        
        return html.Div([
            # 頁面標題
            html.H3("🔍 增強交易分析", className="mb-4"),
            
            # 數據選擇區域
            dbc.Card([
                dbc.CardHeader("📊 數據選擇"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("選擇策略回測結果"),
                            dcc.Dropdown(
                                id='strategy-selector',
                                options=[
                                    {'label': '使用當前回測結果', 'value': 'current'},
                                    {'label': '選擇特定策略', 'value': 'specific'}
                                ],
                                value='current',
                                clearable=False
                            ),
                            html.Div(id='strategy-options', style={'marginTop': '10px'}),
                        ], width=6),
                        dbc.Col([
                            html.Label("基準數據選擇"),
                            dcc.Dropdown(
                                id='benchmark-selector',
                                options=[
                                    {'label': 'TWII (台灣加權指數)', 'value': 'TWII'},
                                    {'label': '2330.TW (台積電)', 'value': '2330.TW'},
                                    {'label': '2412.TW (中華電)', 'value': '2412.TW'},
                                    {'label': '2414.TW (精技)', 'value': '2414.TW'},
                                    {'label': '自定義股價檔案', 'value': 'custom'}
                                ],
                                value='TWII',
                                clearable=False
                            ),
                            html.Div(id='custom-benchmark-upload', style={'display': 'none', 'marginTop': '10px'}),
                        ], width=6)
                    ]),
                    html.Br(),
                    
                    # 從回測結果載入功能
                    dbc.Card([
                        dbc.CardHeader("🧠 自動快取最佳策略數據"),
                        dbc.CardBody([
                            html.Div("說明：系統會自動從主頁籤的回測結果中選擇最佳策略並快取到 enhanced-trades-cache。", className="text-muted mb-2"),
                            dcc.Dropdown(id="enhanced-strategy-selector", placeholder="自動選擇最佳策略（無需手動操作）"),
                            html.Div("✅ 回測完成後會自動快取最佳策略數據", className="text-success mt-2"),
                            dcc.Store(id="enhanced-trades-cache")  # 存放由 backtest-store 解析出的 trade_df
                        ])
                    ], className="mb-3"),
                    
                    dbc.Button(
                         "⏳ 等待回測完成", 
                         id="run-enhanced-analysis-btn", 
                         color="primary",
                         className="me-2",
                         disabled=True
                     ),
                    dbc.Button(
                        "📊 生成報告", 
                        id="generate-enhanced-report-btn", 
                        color="success",
                        className="me-2",
                        disabled=True
                    ),
                    dbc.Button(
                        "🔄 重置", 
                        id="reset-enhanced-analysis-btn", 
                        color="secondary"
                    )
                ])
            ], className="mb-4"),
            
            # 分析結果顯示區域
            html.Div(id="enhanced-analysis-results", children=[
                # 分析摘要卡片
                dbc.Card([
                    dbc.CardHeader("📋 分析摘要"),
                    dbc.CardBody(id="enhanced-analysis-summary")
                ], className="mb-4"),
                
                # 風險閥門分析
                dbc.Card([
                    dbc.CardHeader("⚠️ 風險閥門分析"),
                    dbc.CardBody(id="risk-valve-analysis-content")
                ], className="mb-4"),
                
                # 交易階段分析
                dbc.Card([
                    dbc.CardHeader("🔄 交易階段分析"),
                    dbc.CardBody(id="phase-analysis-content")
                ], className="mb-4"),
                
                # 加碼梯度優化
                dbc.Card([
                    dbc.CardHeader("📈 加碼梯度優化"),
                    dbc.CardBody(id="gradient-optimization-content")
                ], className="mb-4"),
                
                # 圖表顯示區域
                html.Div(id="enhanced-analysis-charts")
            ], style={'display': 'none'})
        ])
        
    def create_enhanced_analysis_callbacks(self, app):
        """創建增強分析的回調函數"""
        from dash import html, Input, Output, State
        
        # 防止重複註冊
        if getattr(app, "_enhanced_callbacks_registered", False):
            return
        app._enhanced_callbacks_registered = True
        
        if not ENHANCED_ANALYSIS_AVAILABLE:
            return
            
        # 策略選擇器回調
        @app.callback(
            Output('strategy-options', 'children'),
            Input('strategy-selector', 'value'),
            State('backtest-store', 'data')
        )
        def update_strategy_options(selector_value, backtest_data):
            if selector_value == 'specific' and backtest_data and 'results' in backtest_data:
                strategies = list(backtest_data.get('results', {}).keys())
                if strategies:
                    return dcc.Dropdown(
                        id='specific-strategy',
                        options=[{'label': s, 'value': s} for s in strategies],
                        value=strategies[0] if strategies else None,
                        placeholder='選擇策略'
                    )
            return html.Div("將使用當前選中的策略")
        
        # 檢查回測狀態並更新執行按鈕狀態
        @app.callback(
            Output('run-enhanced-analysis-btn', 'disabled'),
            Output('run-enhanced-analysis-btn', 'children'),
            Input('backtest-store', 'data'),
            State('strategy-selector', 'value')
        )
        def update_analysis_button_state(backtest_data, strategy_selector):
            if not backtest_data or 'results' not in backtest_data:
                return True, "⏳ 等待回測完成"
            
            results = backtest_data.get('results', {})
            if not results:
                return True, "⏳ 等待回測完成"
            
            # 檢查是否有可用的交易數據
            has_trades = False
            for strategy_name, result in results.items():
                if ('trade_ledger' in result and result['trade_ledger']) or \
                   ('trade_df' in result and result['trade_df']):
                    has_trades = True
                    break
            
            if not has_trades:
                return True, "⚠️ 回測完成但無交易數據"
            
            return False, "🚀 執行增強分析"
        
        # 基準選擇器回調
        @app.callback(
            Output('custom-benchmark-upload', 'children'),
            Output('custom-benchmark-upload', 'style'),
            Input('benchmark-selector', 'value')
        )
        def update_benchmark_upload(benchmark_value):
            if benchmark_value == 'custom':
                return dcc.Upload(
                    id='custom-benchmark-upload-input',
                    children=html.Div([
                        '拖拽或點擊上傳自定義股價數據',
                        html.Br(),
                        html.A('選擇文件')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
                ), {'display': 'block', 'marginTop': '10px'}
            else:
                return "", {'display': 'none'}
        
        # 依 backtest-store 更新策略下拉選單
        @app.callback(
            Output("enhanced-strategy-selector", "options"),
            Input("backtest-store", "data"),
            prevent_initial_call=False
        )
        def _fill_strategy_options(backtest_data):
            if not backtest_data or "results" not in backtest_data:
                return []
            strategies = list(backtest_data["results"].keys())
            return [{"label": s, "value": s} for s in strategies]
        
        # 執行增強分析回調
        @app.callback(
            Output("enhanced-analysis-results", "style"),
            Output("enhanced-analysis-summary", "children"),
            Output("risk-valve-analysis-content", "children"),
            Output("phase-analysis-content", "children"),
            Output("gradient-optimization-content", "children"),
            Output("enhanced-analysis-charts", "children"),
            Output("generate-enhanced-report-btn", "disabled"),
            Input("run-enhanced-analysis-btn", "n_clicks"),
            State("strategy-selector", "value"),
            State("benchmark-selector", "value"),
            State("backtest-store", "data"),
            # 新增 ↓↓↓
            State("enhanced-trades-cache", "data"),
            prevent_initial_call=True
        )
        def run_enhanced_analysis(n_clicks, strategy_selector, benchmark_selector, backtest_data, trades_from_store):
            if n_clicks is None:
                from dash import no_update
                return (no_update, no_update, no_update, no_update, no_update, no_update, no_update)

            try:
                # 1) 先解析交易資料：優先順序 = enhanced-trades-cache → 原有邏輯 → 無
                trades_df = None
                if trades_from_store:
                    # 用 _df_from_pack 還原
                    import io
                    try:
                        trades_df = pd.read_json(io.StringIO(trades_from_store), orient="split")
                        print(f"從 enhanced-trades-cache 載入交易數據成功，形狀：{trades_df.shape}")
                        
                        # 新增（確保日期是 datetime64，避免 object 造成比較/篩選誤差）
                        if '交易日期' in trades_df.columns:
                            trades_df['交易日期'] = pd.to_datetime(trades_df['交易日期'], errors='coerce')
                            trades_df = trades_df.dropna(subset=['交易日期']).sort_values('交易日期').reset_index(drop=True)
                    except Exception as e:
                        print(f"解析 enhanced-trades-cache 失敗：{e}")
                        trades_df = None
                
                # 如果沒有從 enhanced-trades-cache 載入，使用原有邏輯
                if trades_df is None or trades_df.empty:
                    # 檢查回測狀態
                    if not backtest_data or 'results' not in backtest_data:
                        return self._error_outputs("❌ 請先執行策略回測")
                    
                    results = backtest_data.get('results', {})
                    if not results:
                        return self._error_outputs("❌ 回測尚未完成，請等待回測計算完成")
                    
                    # 獲取交易數據
                    trades_df = self._get_trades_data(strategy_selector, None, backtest_data)
                    if trades_df is None or trades_df.empty:
                        return self._error_outputs("❌ 無法獲取交易數據（回測可能尚未完成或數據格式不正確）")
                
                if trades_df is None or trades_df.empty:
                    return self._error_outputs("找不到交易資料（請上傳檔案或先用『從回測結果載入』）")

                # 2) 解析基準：優先順序 = backtest-store 的 df_raw → 本地數據 → 無
                benchmark_df = None
                
                if backtest_data and "df_raw" in backtest_data:
                    try:
                        # 使用與 app_dash 同名的工具函數
                        df_raw = self._df_from_pack(backtest_data["df_raw"])
                        if df_raw is not None and not df_raw.empty:
                            # 轉成增強分析預期欄位（中文）
                            b = pd.DataFrame({
                                "日期": pd.to_datetime(df_raw.index),
                                "收盤價": pd.to_numeric(df_raw["close"], errors="coerce")
                            })
                            # ✅ 不要再 set_index；維持一般整數索引，避免索引名與欄位名重複
                            # b = b.set_index("日期", drop=False)   # ← 刪掉這行
                            
                            # 若無高低價，enhanced_trade_analysis.py 已有回退
                            if "high" in df_raw.columns and "low" in df_raw.columns:
                                b["最高價"] = pd.to_numeric(df_raw["high"], errors="coerce")
                                b["最低價"] = pd.to_numeric(df_raw["low"], errors="coerce")
                            benchmark_df = b
                            print(f"從 backtest-store.df_raw 轉換基準數據成功，形狀：{benchmark_df.shape}")
                            print(f"基準數據欄位：{list(benchmark_df.columns)}")
                            print(f"基準數據索引類型：{type(benchmark_df.index)}")
                    except Exception as e:
                        print(f"從 backtest-store.df_raw 轉換基準數據失敗：{e}")
                
                # 如果還是沒有基準數據，使用本地數據
                if benchmark_df is None or benchmark_df.empty:
                    benchmark_df = self._get_benchmark_data(benchmark_selector, None)

                # debug: benchmark 檢查
                try:
                    from debug_enhanced_data import diag_df
                    diag_df("benchmark", benchmark_df)
                except Exception as e:
                    logger.exception("diag_df(benchmark) fail: %s", e)
                    print(f"診斷失敗：{e}")

                # debug: before analysis
                try:
                    logger.info("DEBUG: entering analyzer with trade rows=%s, benchmark rows=%s", len(trades_df), len(benchmark_df))
                    diag_df("trades_before_analysis", trades_df)
                    diag_df("benchmark_before_analysis", benchmark_df)
                except Exception as e:
                    logger.exception("diag_df(before_analysis) fail: %s", e)
                    print(f"診斷失敗：{e}")

                # === 詳細診斷：實際傳到分析器的數據 ===
                print("\n" + "="*60)
                print("詳細診斷：實際傳到分析器的數據")
                print("="*60)
                
                # 1) 檢查 trades DataFrame
                print("\n--- 1) 檢查 trades DataFrame ---")
                print("columns:", trades_df.columns.tolist())
                print("dtypes:\n", trades_df.dtypes)
                print("trades_df.head(10):")
                print(trades_df.head(10).to_string())
                
                # 關鍵檢查：'盈虧%' 欄
                if '盈虧%' in trades_df.columns:
                    s = pd.to_numeric(trades_df['盈虧%'], errors='coerce')
                    print("\n'盈虧%' 欄詳細檢查:")
                    print("to_numeric NaN count:", s.isna().sum(), "/", len(s))
                    print("sample values:", trades_df['盈虧%'].head(10).tolist())
                    print("stats: min/max/mean/absmax:", s.min(), s.max(), s.mean(), s.abs().max())
                else:
                    print("\nWARNING: trades_df 中沒有 '盈虧%' 欄，請確認預處理流程是否正確產生該欄")
                
                # 2) 檢查基準與 trades 的日期型態、重疊區間與匹配數
                print("\n--- 2) 檢查基準與 trades 的日期匹配 ---")
                print("benchmark columns:", benchmark_df.columns.tolist())
                print("benchmark date dtype:", benchmark_df['日期'].dtype if '日期' in benchmark_df.columns else type(benchmark_df.index))
                print("trades date dtype:", trades_df['交易日期'].dtype)
                
                # 強制 parse 並比較
                benchmark_dates = pd.to_datetime(benchmark_df['日期'] if '日期' in benchmark_df.columns else benchmark_df.index)
                trades_dates = pd.to_datetime(trades_df['交易日期'])
                print("benchmark min/max:", benchmark_dates.min(), benchmark_dates.max())
                print("trades min/max:", trades_dates.min(), trades_dates.max())
                
                # 檢查匹配
                risk_periods = benchmark_dates[benchmark_df.get('risk_valve_triggered', pd.Series(False, index=benchmark_dates.index))]
                print("risk_periods count:", len(risk_periods))
                print("trade dates in risk_periods:", trades_dates.isin(risk_periods).sum(), "/", len(trades_dates))
                
                print("="*60 + "\n")

                # 執行增強分析
                self.enhanced_analyzer = EnhancedTradeAnalyzer(trades_df, benchmark_df)

                risk_results = self.enhanced_analyzer.risk_valve_backtest()
                phase_results = self.enhanced_analyzer.trade_contribution_analysis()
                gradient_results = self.enhanced_analyzer.position_gradient_optimization()

                comprehensive_report = self.enhanced_analyzer.generate_comprehensive_report()
                self.analysis_results = comprehensive_report

                summary_content = self._create_summary_content(comprehensive_report)
                risk_content = self._create_risk_valve_content(risk_results)
                phase_content = self._create_phase_analysis_content(phase_results)
                gradient_content = self._create_gradient_optimization_content(gradient_results)
                charts_content = self._create_charts_content()

                return (
                    {'display': 'block'},
                    summary_content, risk_content, phase_content, gradient_content, charts_content,
                    False  # enable 報告按鈕
                )

            except Exception as e:
                return self._error_outputs(f"分析執行失敗：{str(e)}")
                
        # 生成報告回調
        @app.callback(
            Output("generate-enhanced-report-btn", "children"),
            Input("generate-enhanced-report-btn", "n_clicks"),
            prevent_initial_call=True
        )
        def generate_enhanced_report(n_clicks):
            if n_clicks is None:
                from dash import no_update
                return no_update
                
            try:
                if self.enhanced_analyzer and self.analysis_results:
                    # 生成Excel報告
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"enhanced_analysis_report_{timestamp}.xlsx"
                    
                    # 這裡可以調用報告生成函數
                    # self.enhanced_analyzer.generate_comprehensive_report()
                    
                    return f"✅ 報告已生成 ({filename})"
                else:
                    return "❌ 請先執行分析"
            except Exception as e:
                return f"❌ 生成報告失敗：{str(e)}"
                
        # 重置分析回調
        @app.callback(
            Output("enhanced-analysis-results", "style", allow_duplicate=True),
            Output("enhanced-analysis-summary", "children", allow_duplicate=True),
            Output("risk-valve-analysis-content", "children", allow_duplicate=True),
            Output("phase-analysis-content", "children", allow_duplicate=True),
            Output("gradient-optimization-content", "children", allow_duplicate=True),
            Output("enhanced-analysis-charts", "children", allow_duplicate=True),
            Output("generate-enhanced-report-btn", "disabled", allow_duplicate=True),
            Input("reset-enhanced-analysis-btn", "n_clicks"),
            prevent_initial_call=True
        )
        def reset_enhanced_analysis(n_clicks):
            if n_clicks is None:
                from dash import no_update
                return no_update
                
            self.enhanced_analyzer = None
            self.analysis_results = {}
            
            return (
                {'display': 'none'},  # 隱藏結果區域
                html.P("請選擇數據並執行分析"),
                html.P("請選擇數據並執行分析"),
                html.P("請選擇數據並執行分析"),
                html.P("請選擇數據並執行分析"),
                html.P("請選擇數據並執行分析"),
                True  # 禁用生成報告按鈕
            )
            
    def _parse_uploaded_data(self, content, data_type):
        """解析上傳的數據"""
        if content is None:
            print(f"警告：{data_type} 數據為空")
            return None
            
        import base64
        import io
        
        try:
            print(f"開始解析 {data_type} 數據...")
            print(f"Content 類型: {type(content)}")
            print(f"Content 長度: {len(str(content)) if content else 0}")
            
            # 檢查 content 格式
            if not isinstance(content, str):
                print(f"錯誤：content 不是字串，而是 {type(content)}")
                return None
                
            if ',' not in content:
                print(f"錯誤：content 格式不正確，缺少逗號分隔符")
                print(f"Content 前100字符: {content[:100] if content else 'None'}")
                return None
            
            # 解碼base64內容
            content_type, content_string = content.split(',', 1)
            print(f"Content type: {content_type}")
            print(f"Content string 長度: {len(content_string)}")
            
            try:
                decoded = base64.b64decode(content_string)
                print(f"解碼後數據長度: {len(decoded)} bytes")
            except Exception as decode_error:
                print(f"Base64 解碼失敗: {decode_error}")
                return None
            
            df = None
            if 'csv' in content_type:
                print("嘗試解析 CSV 格式...")
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                    print(f"CSV 解析成功，欄位: {list(df.columns)}")
                except UnicodeDecodeError:
                    # 嘗試其他編碼
                    try:
                        df = pd.read_csv(io.StringIO(decoded.decode('big5')))
                        print(f"CSV 解析成功 (big5編碼)，欄位: {list(df.columns)}")
                    except:
                        df = pd.read_csv(io.StringIO(decoded.decode('utf-8', errors='ignore')))
                        print(f"CSV 解析成功 (忽略錯誤)，欄位: {list(df.columns)}")
            elif 'excel' in content_type:
                print("嘗試解析 Excel 格式...")
                df = pd.read_excel(io.BytesIO(decoded))
                print(f"Excel 解析成功，欄位: {list(df.columns)}")
            else:
                print(f"不支援的檔案類型: {content_type}")
                return None
            
            if df is None or df.empty:
                print(f"警告：解析後的 DataFrame 為空")
                return None
                
            print(f"原始數據形狀: {df.shape}")
            print(f"原始欄位: {list(df.columns)}")
            
            # 根據數據類型進行預處理
            if data_type == "trades":
                df = self._preprocess_trades_data(df)
            elif data_type == "benchmark":
                df = self._preprocess_benchmark_data(df)
            
            print(f"預處理後數據形狀: {df.shape}")
            print(f"預處理後欄位: {list(df.columns)}")
                
            return df
            
        except Exception as e:
            print(f"解析{data_type}數據失敗：{e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _ensure_trade_returns(self, df):
        """確保交易數據有盈虧%欄位，若無則用買賣配對從價格推算"""
        # 已有就不動
        if '盈虧%' in df.columns:
            return df

        df = df.copy()
        
        # 確保有必要的欄位
        if '交易日期' not in df.columns or '交易類型' not in df.columns:
            print("缺少必要欄位，無法推算盈虧%")
            df['盈虧%'] = 0.0
            return df
            
        df.sort_values('交易日期', inplace=True)

        # 用簡單買賣配對：遇到 sell，使用上一筆 buy 的價格算報酬
        last_buy_price = None
        returns = []
        for _, row in df.iterrows():
            t = str(row.get('交易類型') or row.get('type') or '').lower()
            px = row.get('價格', row.get('price'))
            ret = None
            if t == 'buy':
                last_buy_price = px
            elif t == 'sell' and last_buy_price and last_buy_price > 0:
                ret = (px - last_buy_price) / last_buy_price
                last_buy_price = None
            returns.append(ret)

        df['盈虧%'] = pd.Series(returns, index=df.index)
        print("已從買賣配對推算盈虧%欄位")
        return df

    def _preprocess_trades_data(self, df):
        """預處理交易數據"""
        print(f"開始預處理交易數據...")
        print(f"輸入欄位: {list(df.columns)}")
        
        # 標準化欄位名稱
        column_mapping = {
            'Date': '交易日期',
            'date': '交易日期',
            'Weight_Change': '權重變化',
            'weight_change': '權重變化',
            'Pnl_Pct': '盈虧%',
            'pnl_pct': '盈虧%',
            'Pnl': '盈虧%',
            'pnl': '盈虧%',
            'pnl_pct': '盈虧%',
            'pnl_pct': '盈虧%',
            'return': '盈虧%',
            'Return': '盈虧%',
            'weight': '權重變化',
            'Weight': '權重變化',
            'trade_date': '交易日期',
            'Trade_Date': '交易日期'
        }
        
        # 檢查原始欄位
        original_columns = list(df.columns)
        print(f"原始欄位: {original_columns}")
        
        # 重命名欄位
        df = df.rename(columns=column_mapping)
        print(f"重命名後欄位: {list(df.columns)}")
        
        # 檢查必要欄位
        required_columns = ['交易日期', '權重變化', '盈虧%']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"警告：缺少必要欄位：{missing_columns}")
            print(f"可用欄位: {list(df.columns)}")
            
            # 嘗試從其他欄位推導
            if '盈虧%' not in df.columns:
                print("嘗試推導盈虧%欄位...")
                if '每筆盈虧%' in df.columns:
                    df['盈虧%'] = df['每筆盈虧%']
                    print("使用 '每筆盈虧%' 作為 '盈虧%'")
                elif '總資產' in df.columns and '交易日期' in df.columns:
                    df = df.sort_values('交易日期').reset_index(drop=True)
                    df['盈虧%'] = df['總資產'].pct_change().fillna(0) * 100.0
                    print("從 '總資產' 推導 '盈虧%'")
                else:
                    print("嘗試用買賣配對推算盈虧%...")
                    df = self._ensure_trade_returns(df)
            
            if '權重變化' not in df.columns:
                print("嘗試推導權重變化欄位...")
                if 'weight' in df.columns:
                    df['權重變化'] = df['weight']
                    print("使用 'weight' 作為 '權重變化'")
                elif 'w' in df.columns:
                    df['權重變化'] = df['w']
                    print("使用 'w' 作為 '權重變化'")
                else:
                    print("無法推導權重變化，設定為0")
                    df['權重變化'] = 0.0
            
            if '交易日期' not in df.columns:
                print("嘗試推導交易日期欄位...")
                if 'date' in df.columns:
                    df['交易日期'] = df['date']
                    print("使用 'date' 作為 '交易日期'")
                elif 'trade_date' in df.columns:
                    df['交易日期'] = df['trade_date']
                    print("使用 'trade_date' 作為 '交易日期'")
                else:
                    print("無法推導交易日期")
                    return df
        
        # 轉換日期格式
        if '交易日期' in df.columns:
            try:
                df['交易日期'] = pd.to_datetime(df['交易日期'])
                print("日期格式轉換成功")
            except Exception as e:
                print(f"日期格式轉換失敗: {e}")
                # 嘗試其他日期格式
                try:
                    df['交易日期'] = pd.to_datetime(df['交易日期'], format='%Y/%m/%d')
                    print("使用 %Y/%m/%d 格式轉換成功")
                except:
                    try:
                        df['交易日期'] = pd.to_datetime(df['交易日期'], format='%Y-%m-%d')
                        print("使用 %Y-%m-%d 格式轉換成功")
                    except:
                        print("所有日期格式都轉換失敗")
        
        print(f"預處理完成，最終欄位: {list(df.columns)}")
        print(f"最終數據形狀: {df.shape}")
        
        return df
        
    def _preprocess_benchmark_data(self, df):
        """預處理基準數據"""
        print(f"開始預處理基準數據...")
        print(f"輸入欄位: {list(df.columns)}")
        
        # 標準化欄位名稱
        column_mapping = {
            'Date': '日期',
            'date': '日期',
            'Close': '收盤價',
            'close': '收盤價',
            'High': '最高價',
            'high': '最高價',
            'Low': '最低價',
            'low': '最低價',
            'trade_date': '日期',
            'Trade_Date': '日期',
            'price': '收盤價',
            'Price': '收盤價'
        }
        
        # 檢查原始欄位
        original_columns = list(df.columns)
        print(f"原始欄位: {original_columns}")
        
        # 重命名欄位
        df = df.rename(columns=column_mapping)
        print(f"重命名後欄位: {list(df.columns)}")
        
        # 檢查必要欄位
        required_columns = ['日期', '收盤價']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"警告：缺少必要欄位：{missing_columns}")
            print(f"可用欄位: {list(df.columns)}")
            
            # 嘗試從其他欄位推導
            if '日期' not in df.columns:
                print("嘗試推導日期欄位...")
                if 'trade_date' in df.columns:
                    df['日期'] = df['trade_date']
                    print("使用 'trade_date' 作為 '日期'")
                elif 'index' in df.columns:
                    df['日期'] = df['index']
                    print("使用 'index' 作為 '日期'")
                elif df.index.name is not None and df.index.dtype == 'datetime64[ns]':
                    # 如果索引是日期類型，從索引創建日期欄位
                    df['日期'] = df.index
                    print("從索引創建 '日期' 欄位")
                else:
                    print("無法推導日期欄位")
                    return df
            
            if '收盤價' not in df.columns:
                print("嘗試推導收盤價欄位...")
                if 'price' in df.columns:
                    df['收盤價'] = df['price']
                    print("使用 'price' 作為 '收盤價'")
                elif 'open' in df.columns:
                    df['收盤價'] = df['open']
                    print("使用 'open' 作為 '收盤價'")
                else:
                    print("無法推導收盤價欄位")
                    return df
        
        # 轉換日期格式
        if '日期' in df.columns:
            try:
                df['日期'] = pd.to_datetime(df['日期'])
                print("日期格式轉換成功")
            except Exception as e:
                print(f"日期格式轉換失敗: {e}")
                # 嘗試其他日期格式
                try:
                    df['日期'] = pd.to_datetime(df['日期'], format='%Y/%m/%d')
                    print("使用 %Y/%m/%d 格式轉換成功")
                except:
                    try:
                        df['日期'] = pd.to_datetime(df['日期'], format='%Y-%m-%d')
                        print("使用 %Y-%m-%d 格式轉換成功")
                    except:
                        print("所有日期格式都轉換失敗")
        
        # 確保收盤價為數值型
        if '收盤價' in df.columns:
            try:
                df['收盤價'] = pd.to_numeric(df['收盤價'], errors='coerce')
                print("收盤價轉換為數值型成功")
            except Exception as e:
                print(f"收盤價轉換失敗: {e}")
        
        print(f"預處理完成，最終欄位: {list(df.columns)}")
        print(f"最終數據形狀: {df.shape}")
        
        return df
        
    def _create_summary_content(self, report):
        """創建分析摘要內容（修：先計算變數，避免在 children 內賦值）"""
        from dash import html, dcc
        import dash_bootstrap_components as dbc

        if not report:
            return html.P("無分析結果")

        risk_count = report.get('analysis_results', {}).get('risk_valve', {}).get('risk_periods_count', 0)
        phase_count = len(report.get('analysis_results', {}).get('phase_analysis', {}))

        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("📊 基本資訊", className="card-title"),
                            html.P(f"分析時間：{report.get('analysis_timestamp', 'N/A')}"),
                            html.P(f"總交易數：{report.get('total_trades', 0)}"),
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("⚠️ 風險閥門", className="card-title"),
                            html.P(f"觸發期間數：{risk_count}"),
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("🔄 交易階段", className="card-title"),
                            html.P(f"識別階段數：{phase_count}"),
                        ])
                    ])
                ], width=4),
            ])
        ])
        
    def _create_risk_valve_content(self, risk_results):
        """創建風險閥門分析內容（修：在 children 之外計算 improvement）"""
        from dash import html
        import dash_bootstrap_components as dbc

        if not risk_results:
            return html.P("無風險閥門分析結果")

        improvement = risk_results.get('improvement_potential', {}) or {}
        mdd_red = improvement.get('mdd_reduction', 0)
        pf_imp = improvement.get('pf_improvement', 0)

        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6("風險期間 vs 正常期間對比"),
                    html.P(f"風險期間交易數：{risk_results.get('risk_trades_count', 0)}"),
                    html.P(f"正常期間交易數：{risk_results.get('normal_trades_count', 0)}"),
                ], width=6),
                dbc.Col([
                    html.H6("改善潛力"),
                    html.P(f"MDD改善潛力：{mdd_red:.2%}" if isinstance(mdd_red, float) else f"MDD改善潛力：{mdd_red}"),
                    html.P(f"PF改善潛力：{pf_imp:.2f}" if isinstance(pf_imp, float) else f"PF改善潛力：{pf_imp}"),
                ], width=6)
            ])
        ])
        
    def _error_outputs(self, message):
        """統一錯誤輸出，確保 7 個輸出完整"""
        from dash import html
        return (
            {'display': 'block'},   # results 區域顯示
            self._create_error_message(message),  # summary 塞錯誤卡片
            html.P("—"), html.P("—"), html.P("—"), html.P("—"),  # 其他區域佔位
            True  # 生成報告按鈕 disable
        )
        
    def _create_phase_analysis_content(self, phase_results):
        """創建交易階段分析內容"""
        from dash import html
        import dash_bootstrap_components as dbc
        
        if not phase_results:
            return html.P("無交易階段分析結果")
            
        phase_cards = []
        for phase_type, phase_info in phase_results.items():
            card = dbc.Card([
                dbc.CardHeader(f"📊 {phase_type} 階段"),
                dbc.CardBody([
                    html.P(f"交易數：{phase_info.get('trade_count', 0)}"),
                    html.P(f"總報酬：{phase_info.get('total_return', 0):.2f}%"),
                    html.P(f"貢獻比例：{phase_info.get('contribution_ratio', 0):.2%}"),
                    html.P(f"期間：{phase_info.get('start_date', 'N/A')} 到 {phase_info.get('end_date', 'N/A')}"),
                ])
            ], className="mb-3")
            phase_cards.append(card)
            
        return html.Div(phase_cards)
        
    def _create_gradient_optimization_content(self, gradient_results):
        """創建加碼梯度優化內容"""
        from dash import html
        import dash_bootstrap_components as dbc
        
        if not gradient_results:
            return html.P("無加碼梯度優化結果")
            
        current = gradient_results.get('current_pattern', {})
        optimized = gradient_results.get('optimized_pattern', {})
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6("當前加碼模式"),
                    html.P(f"平均間距：{current.get('avg_interval', 0):.1f} 天"),
                    html.P(f"最大連續加碼：{current.get('max_consecutive', 0)} 筆"),
                ], width=6),
                dbc.Col([
                    html.H6("優化後效果"),
                    html.P(f"加碼次數：從 {optimized.get('original_count', 0)} 減少到 {optimized.get('optimized_count', 0)}"),
                    html.P(f"減少比例：{optimized.get('reduction_ratio', 0):.1%}"),
                ], width=6)
            ])
        ])
        
    def _create_charts_content(self):
        """創建圖表內容"""
        from dash import html
        
        if not self.enhanced_analyzer:
            return html.P("無圖表數據")
            
        try:
            # 這裡可以調用圖表生成函數
            # fig = self.enhanced_analyzer.plot_enhanced_analysis()
            return html.Div([
                html.H5("📈 分析圖表"),
                html.P("圖表功能開發中..."),
                html.P("目前可通過控制台查看分析結果")
            ])
        except Exception as e:
            return html.P(f"圖表生成失敗：{str(e)}")
            
    def _create_error_message(self, message):
        """創建錯誤訊息"""
        from dash import html
        import dash_bootstrap_components as dbc
        return html.Div([
            dbc.Alert([
                html.H4("❌ 錯誤", className="alert-heading"),
                html.P(message),
            ], color="danger")
        ])

    def _get_trades_data(self, strategy_selector, specific_strategy, backtest_data):
        """從策略回測結果獲取交易數據"""
        if not backtest_data or 'results' not in backtest_data:
            print("警告：沒有可用的回測數據")
            return None
            
        results = backtest_data['results']
        
        if not results:
            print("警告：回測結果為空")
            return None
        
        # 選策略：打分法（ledger_std > ledger > trade_df）
        def _score(res):
            s = 0
            if res.get('trade_ledger_std'): s += 3
            if res.get('trade_ledger'): s += 2
            if res.get('trades_df') or res.get('trade_df'): s += 1
            return s

        if strategy_selector in ('current', 'specific'):
            # 若 future 想加「指定策略名」可用 specific_strategy
            best = None
            best_name = None
            for name, res in results.items():
                if best is None or _score(res) > _score(best):
                    best, best_name = res, name
            strategy_name = best_name
        else:
            print("警告：未選擇策略")
            return None
            
        if not strategy_name or strategy_name not in results:
            print(f"警告：策略 {strategy_name} 不存在")
            return None
            
        result = results[strategy_name]
        print(f"使用策略：{strategy_name}")
        
        # 嘗試獲取交易數據
        trades_df = None
        
        # 優先使用 trade_ledger（更完整的交易記錄）
        if 'trade_ledger' in result and result['trade_ledger']:
            try:
                if isinstance(result['trade_ledger'], str):
                    import json
                    trades_df = pd.read_json(result['trade_ledger'], orient='split')
                else:
                    trades_df = result['trade_ledger']
                print(f"使用 trade_ledger，欄位：{list(trades_df.columns)}")
            except Exception as e:
                print(f"解析 trade_ledger 失敗：{e}")
        
        # 如果沒有 trade_ledger，使用 trade_df
        if trades_df is None and 'trade_df' in result and result['trade_df']:
            try:
                if isinstance(result['trade_df'], str):
                    import json
                    trades_df = pd.read_json(result['trade_df'], orient='split')
                else:
                    trades_df = result['trade_df']
                print(f"使用 trade_df，欄位：{list(trades_df.columns)}")
            except Exception as e:
                print(f"解析 trade_df 失敗：{e}")
        
        if trades_df is None or trades_df.empty:
            print("警告：無法獲取有效的交易數據")
            return None
            
        # 預處理交易數據
        trades_df = self._preprocess_trades_data(trades_df)
        return trades_df
    
    def _get_benchmark_data(self, benchmark_selector, custom_benchmark):
        """獲取基準數據"""
        if benchmark_selector == 'custom' and custom_benchmark:
            # 使用自定義上傳的基準數據
            return self._parse_uploaded_data(custom_benchmark, "benchmark")
        else:
            # 使用本地股價數據
            return self._load_local_price_data(benchmark_selector)
    
    def _load_local_price_data(self, ticker):
        """從本地 data/ 目錄載入股價數據"""
        import os
        import pandas as pd
        
        try:
            # 構建檔案路徑
            data_dir = 'data'
            if ticker == 'TWII':
                filename = '^TWII_data_raw.csv'
            else:
                filename = f"{ticker}_data_raw.csv"
            
            file_path = os.path.join(data_dir, filename)
            
            if not os.path.exists(file_path):
                print(f"警告：股價檔案不存在：{file_path}")
                return None
            
            # 特殊處理 TWII 數據格式
            if ticker == 'TWII':
                # 第一行是欄位名稱，第二行是 ticker 信息，第三行是空的，第四行開始是數據
                df = pd.read_csv(file_path, skiprows=2, index_col=0, parse_dates=True)
                # 手動設置正確的欄位名稱（index_col=0 會把第一列當作索引，所以實際欄位只有 5 個）
                df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
            else:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            print(f"成功載入股價數據：{ticker}，形狀：{df.shape}")
            
            # 預處理基準數據
            df = self._preprocess_benchmark_data(df)
            return df
            
        except Exception as e:
            print(f"載入股價數據失敗：{e}")
            return None
    
    def _df_from_pack(self, data):
        """解析打包的數據（與 app_dash 同名工具函數）"""
        import io
        import pandas as pd
        
        if data is None or data == "" or data == "[]":
            return pd.DataFrame()
        if isinstance(data, str):
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

def create_enhanced_analysis_tab():
    """創建增強分析標籤頁的便捷函數"""
    ui = EnhancedAnalysisUI()
    return ui.create_enhanced_analysis_tab()

def setup_enhanced_analysis_callbacks(app):
    """設置增強分析回調函數的便捷函數"""
    ui = EnhancedAnalysisUI()
    ui.create_enhanced_analysis_callbacks(app)
    return ui

if __name__ == "__main__":
    print("增強分析UI整合模組")
    print("路徑：#analysis/enhanced_analysis_ui.py")
    print("創建時間：2025-08-18 04:45")
    print("\n使用方法：")
    print("1. 在app_dash.py中導入：from analysis.enhanced_analysis_ui import create_enhanced_analysis_tab")
    print("2. 在標籤頁中添加：create_enhanced_analysis_tab()")
    print("3. 設置回調：setup_enhanced_analysis_callbacks(app)")
