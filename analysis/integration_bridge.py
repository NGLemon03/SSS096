# -*- coding: utf-8 -*-
"""
增強分析整合橋接模組 - 2025-08-18 04:38
將新的增強分析功能整合到現有的分析工作流程中

作者：AI Assistant
路徑：#analysis/integration_bridge.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from enhanced_trade_analysis import EnhancedTradeAnalyzer
    ENHANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYSIS_AVAILABLE = False
    print("警告：無法導入 EnhancedTradeAnalyzer，增強分析功能將不可用")

class AnalysisIntegrationBridge:
    """分析整合橋接器"""
    
    def __init__(self):
        """初始化橋接器"""
        self.enhanced_analyzer = None
        self.integration_status = {
            'enhanced_analysis_available': ENHANCED_ANALYSIS_AVAILABLE,
            'last_analysis_time': None,
            'analysis_results': {}
        }
        
        print(f"分析整合橋接器初始化完成")
        print(f"增強分析模組可用性：{ENHANCED_ANALYSIS_AVAILABLE}")
        print(f"路徑：#analysis/integration_bridge.py")
        
    def load_trade_data(self, file_path, data_format='auto'):
        """
        載入交易數據
        
        Args:
            file_path: 數據文件路徑
            data_format: 數據格式 ('auto', 'csv', 'excel', 'json')
        """
        print(f"載入交易數據：{file_path}")
        
        try:
            if data_format == 'auto':
                if file_path.endswith('.csv'):
                    data_format = 'csv'
                elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                    data_format = 'excel'
                elif file_path.endswith('.json'):
                    data_format = 'json'
                    
            if data_format == 'csv':
                df = pd.read_csv(file_path)
            elif data_format == 'excel':
                df = pd.read_excel(file_path)
            elif data_format == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"不支援的數據格式：{data_format}")
                
            print(f"成功載入 {len(df)} 筆交易記錄")
            return df
            
        except Exception as e:
            print(f"載入數據失敗：{e}")
            return None
            
    def load_benchmark_data(self, file_path, benchmark_type='TWII'):
        """
        載入基準數據
        
        Args:
            file_path: 基準數據文件路徑
            benchmark_type: 基準類型 ('TWII', '00631L', 'custom')
        """
        print(f"載入基準數據：{file_path} (類型: {benchmark_type})")
        
        try:
            df = pd.read_csv(file_path)
            
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
                'Volume': '成交量',
                'volume': '成交量'
            }
            
            df = df.rename(columns=column_mapping)
            
            # 確保必要欄位存在
            required_columns = ['日期', '收盤價']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"警告：缺少必要欄位：{missing_columns}")
                return None
                
            print(f"成功載入 {len(df)} 筆基準數據")
            return df
            
        except Exception as e:
            print(f"載入基準數據失敗：{e}")
            return None
            
    def run_enhanced_analysis(self, trades_df, benchmark_df=None):
        """
        執行增強分析
        
        Args:
            trades_df: 交易數據DataFrame
            benchmark_df: 基準數據DataFrame（可選）
        """
        if not ENHANCED_ANALYSIS_AVAILABLE:
            print("錯誤：增強分析模組不可用")
            return None
            
        print("\n" + "="*60)
        print("執行增強交易分析")
        print("="*60)
        
        try:
            # 創建增強分析器
            self.enhanced_analyzer = EnhancedTradeAnalyzer(trades_df, benchmark_df)
            
            # 執行所有分析
            print("\n1. 風險閥門回測...")
            risk_results = self.enhanced_analyzer.risk_valve_backtest()
            
            print("\n2. 交易貢獻拆解...")
            phase_results = self.enhanced_analyzer.trade_contribution_analysis()
            
            print("\n3. 加碼梯度優化...")
            gradient_results = self.enhanced_analyzer.position_gradient_optimization()
            
            # 生成綜合報告
            print("\n4. 生成綜合報告...")
            comprehensive_report = self.enhanced_analyzer.generate_comprehensive_report()
            
            # 更新狀態
            self.integration_status['last_analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.integration_status['analysis_results'] = comprehensive_report
            
            print(f"\n✅ 增強分析完成！")
            print(f"分析時間：{self.integration_status['last_analysis_time']}")
            
            return comprehensive_report
            
        except Exception as e:
            print(f"❌ 增強分析執行失敗：{e}")
            import traceback
            traceback.print_exc()
            return None
            
    def generate_enhanced_report(self, output_format='excel'):
        """
        生成增強分析報告
        
        Args:
            output_format: 輸出格式 ('excel', 'csv', 'json', 'html')
        """
        if not self.enhanced_analyzer or not self.integration_status['analysis_results']:
            print("錯誤：請先執行增強分析")
            return None
            
        print(f"\n生成增強分析報告 (格式: {output_format})")
        
        try:
            results = self.integration_status['analysis_results']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if output_format == 'excel':
                filename = f"enhanced_analysis_report_{timestamp}.xlsx"
                self._export_to_excel(results, filename)
            elif output_format == 'csv':
                filename = f"enhanced_analysis_report_{timestamp}.csv"
                self._export_to_csv(results, filename)
            elif output_format == 'json':
                filename = f"enhanced_analysis_report_{timestamp}.json"
                self._export_to_json(results, filename)
            elif output_format == 'html':
                filename = f"enhanced_analysis_report_{timestamp}.html"
                self._export_to_html(results, filename)
            else:
                raise ValueError(f"不支援的輸出格式：{output_format}")
                
            print(f"✅ 報告已生成：{filename}")
            return filename
            
        except Exception as e:
            print(f"❌ 生成報告失敗：{e}")
            return None
            
    def _export_to_excel(self, results, filename):
        """導出到Excel"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 摘要工作表
            summary_data = {
                '分析項目': ['分析時間', '總交易數', '風險閥門觸發期間', '交易階段數'],
                '數值': [
                    results.get('analysis_timestamp', 'N/A'),
                    results.get('total_trades', 0),
                    results.get('analysis_results', {}).get('risk_valve', {}).get('risk_periods_count', 0),
                    len(results.get('analysis_results', {}).get('phase_analysis', {}))
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='分析摘要', index=False)
            
            # 風險閥門分析工作表
            if 'risk_valve' in results.get('analysis_results', {}):
                risk_data = results['analysis_results']['risk_valve']
                risk_summary = pd.DataFrame([
                    ['風險期間交易數', risk_data.get('risk_trades_count', 0)],
                    ['正常期間交易數', risk_data.get('normal_trades_count', 0)],
                    ['MDD改善潛力', f"{risk_data.get('improvement_potential', {}).get('mdd_reduction', 0):.2%}"],
                    ['PF改善潛力', f"{risk_data.get('improvement_potential', {}).get('pf_improvement', 0):.2f}"]
                ], columns=['指標', '數值'])
                risk_summary.to_excel(writer, sheet_name='風險閥門分析', index=False)
                
            # 交易階段分析工作表
            if 'phase_analysis' in results.get('analysis_results', {}):
                phase_data = results['analysis_results']['phase_analysis']
                phase_summary = []
                for phase_type, phase_info in phase_data.items():
                    phase_summary.append({
                        '階段類型': phase_type,
                        '交易數': phase_info.get('trade_count', 0),
                        '總報酬': f"{phase_info.get('total_return', 0):.2f}%",
                        '貢獻比例': f"{phase_info.get('contribution_ratio', 0):.2%}",
                        '開始日期': phase_info.get('start_date', 'N/A'),
                        '結束日期': phase_info.get('end_date', 'N/A')
                    })
                pd.DataFrame(phase_summary).to_excel(writer, sheet_name='交易階段分析', index=False)
                
            # 加碼梯度優化工作表
            if 'gradient_optimization' in results.get('analysis_results', {}):
                grad_data = results['analysis_results']['gradient_optimization']
                grad_summary = pd.DataFrame([
                    ['當前平均間距', f"{grad_data.get('current_pattern', {}).get('avg_interval', 0):.1f} 天"],
                    ['最大連續加碼', grad_data.get('current_pattern', {}).get('max_consecutive', 0)],
                    ['優化後加碼次數', grad_data.get('optimized_pattern', {}).get('optimized_count', 0)],
                    ['減少比例', f"{grad_data.get('optimized_pattern', {}).get('reduction_ratio', 0):.1%}"]
                ], columns=['指標', '數值'])
                grad_summary.to_excel(writer, sheet_name='加碼梯度優化', index=False)
                
    def _export_to_csv(self, results, filename):
        """導出到CSV"""
        # 簡化版本，只導出摘要
        summary_data = {
            '分析項目': ['分析時間', '總交易數'],
            '數值': [
                results.get('analysis_timestamp', 'N/A'),
                results.get('total_trades', 0)
            ]
        }
        pd.DataFrame(summary_data).to_csv(filename, index=False, encoding='utf-8-sig')
        
    def _export_to_json(self, results, filename):
        """導出到JSON"""
        import json
        
        # 處理日期序列化
        def serialize_dates(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d')
            elif isinstance(obj, datetime):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=serialize_dates)
            
    def _export_to_html(self, results, filename):
        """導出到HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>增強交易分析報告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>增強交易分析報告</h1>
                <p>生成時間：{results.get('analysis_timestamp', 'N/A')}</p>
                <p>總交易數：{results.get('total_trades', 0)}</p>
            </div>
            
            <div class="section">
                <h2>分析摘要</h2>
                <div class="metric">分析時間：{results.get('analysis_timestamp', 'N/A')}</div>
                <div class="metric">總交易數：{results.get('total_trades', 0)}</div>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
    def get_analysis_summary(self):
        """獲取分析摘要"""
        if not self.integration_status['analysis_results']:
            return "尚未執行分析"
            
        results = self.integration_status['analysis_results']
        
        summary = f"""
=== 增強分析摘要 ===
分析時間：{results.get('analysis_timestamp', 'N/A')}
總交易數：{results.get('total_trades', 0)}

風險閥門分析：
- 觸發期間數：{results.get('analysis_results', {}).get('risk_valve', {}).get('risk_periods_count', 0)}
- 風險期間交易數：{results.get('analysis_results', {}).get('risk_valve', {}).get('risk_trades_count', 0)}

交易階段分析：
- 識別階段數：{len(results.get('analysis_results', {}).get('phase_analysis', {}))}

加碼梯度優化：
- 當前平均間距：{results.get('analysis_results', {}).get('gradient_optimization', {}).get('current_pattern', {}).get('avg_interval', 0):.1f} 天
- 優化後加碼次數：{results.get('analysis_results', {}).get('gradient_optimization', {}).get('optimized_pattern', {}).get('optimized_count', 0)}
        """
        
        return summary
        
    def plot_integrated_analysis(self):
        """繪製整合分析圖表"""
        if not self.enhanced_analyzer:
            print("錯誤：請先執行增強分析")
            return None
            
        try:
            print("繪製整合分析圖表...")
            fig = self.enhanced_analyzer.plot_enhanced_analysis()
            print("✅ 圖表繪製完成")
            return fig
        except Exception as e:
            print(f"❌ 圖表繪製失敗：{e}")
            return None

def main():
    """主函數 - 示範用法"""
    print("增強分析整合橋接模組")
    print("路徑：#analysis/integration_bridge.py")
    print("創建時間：2025-08-18 04:38")
    
    # 創建橋接器實例
    bridge = AnalysisIntegrationBridge()
    
    print("\n使用方法：")
    print("1. 載入交易數據：bridge.load_trade_data('trades.csv')")
    print("2. 載入基準數據：bridge.load_benchmark_data('benchmark.csv')")
    print("3. 執行增強分析：bridge.run_enhanced_analysis(trades_df, benchmark_df)")
    print("4. 生成報告：bridge.generate_enhanced_report('excel')")
    print("5. 繪製圖表：bridge.plot_integrated_analysis()")
    
    return bridge

if __name__ == "__main__":
    main()
