# -*- coding: utf-8 -*-
"""
增強交易分析模組使用範例 - 2025-08-18 04:38
展示如何使用 EnhancedTradeAnalyzer 進行三項改進分析

作者：AI Assistant
路徑：#analysis/enhanced_analysis_example.py
"""

import pandas as pd
import numpy as np
from enhanced_trade_analysis import EnhancedTradeAnalyzer
import matplotlib.pyplot as plt

def create_sample_data():
    """創建範例數據"""
    print("創建範例交易數據...")
    
    # 生成範例交易數據
    dates = pd.date_range('2019-01-01', '2024-12-31', freq='D')
    np.random.seed(42)  # 固定隨機種子以便重現
    
    # 模擬交易記錄
    trades_data = []
    current_weight = 0
    current_date = dates[0]
    
    for i, date in enumerate(dates):
        # 每30-60天隨機產生一筆交易
        if np.random.random() < 0.02:  # 2%機率產生交易
            # 權重變化（-0.1 到 0.2 之間）
            weight_change = np.random.uniform(-0.1, 0.2)
            
            # 模擬盈虧%（基於權重變化和市場環境）
            if weight_change > 0:  # 買入
                # 買入後可能獲利或虧損
                pnl = np.random.normal(5, 15)  # 平均5%，標準差15%
            else:  # 賣出
                # 賣出通常獲利
                pnl = np.random.normal(10, 12)  # 平均10%，標準差12%
                
            trades_data.append({
                '交易日期': date,
                '權重變化': weight_change,
                '盈虧%': pnl,
                '交易金額': abs(weight_change) * 100000  # 假設每單位權重10萬
            })
            
            current_weight += weight_change
            current_date = date
    
    trades_df = pd.DataFrame(trades_data)
    print(f"創建了 {len(trades_df)} 筆範例交易")
    
    # 生成範例基準數據（模擬TWII）
    benchmark_data = []
    base_price = 10000
    current_price = base_price
    
    for date in dates:
        # 模擬股價變動
        daily_return = np.random.normal(0.0005, 0.015)  # 平均日報酬0.05%，波動1.5%
        current_price *= (1 + daily_return)
        
        # 模擬高低價
        high_price = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low_price = current_price * (1 - abs(np.random.normal(0, 0.01)))
        
        benchmark_data.append({
            '日期': date,
            '收盤價': current_price,
            '最高價': high_price,
            '最低價': low_price
        })
    
    benchmark_df = pd.DataFrame(benchmark_data)
    print(f"創建了 {len(benchmark_df)} 筆基準數據")
    
    return trades_df, benchmark_df

def run_enhanced_analysis():
    """執行增強分析"""
    print("\n" + "="*60)
    print("增強交易分析模組示範")
    print("路徑：#analysis/enhanced_analysis_example.py")
    print("="*60)
    
    # 創建範例數據
    trades_df, benchmark_df = create_sample_data()
    
    # 創建分析器
    print("\n初始化增強交易分析器...")
    analyzer = EnhancedTradeAnalyzer(trades_df, benchmark_df)
    
    # 1. 風險閥門回測
    print("\n" + "-"*40)
    print("1. 執行風險閥門回測分析")
    print("-"*40)
    
    risk_results = analyzer.risk_valve_backtest()
    
    # 2. 交易貢獻拆解
    print("\n" + "-"*40)
    print("2. 執行交易貢獻拆解分析")
    print("-"*40)
    
    phase_results = analyzer.trade_contribution_analysis()
    
    # 3. 加碼梯度優化
    print("\n" + "-"*40)
    print("3. 執行加碼梯度優化分析")
    print("-"*40)
    
    # 測試不同的參數組合
    optimization_params = [
        {'min_interval_days': 3, 'cooldown_days': 7},
        {'min_interval_days': 5, 'cooldown_days': 10},
        {'min_interval_days': 7, 'cooldown_days': 14}
    ]
    
    for params in optimization_params:
        print(f"\n測試參數：最小間距 {params['min_interval_days']} 天，冷卻期 {params['cooldown_days']} 天")
        analyzer.position_gradient_optimization(**params)
    
    # 4. 生成綜合報告
    print("\n" + "-"*40)
    print("4. 生成綜合分析報告")
    print("-"*40)
    
    comprehensive_report = analyzer.generate_comprehensive_report()
    
    # 5. 繪製分析圖表
    print("\n" + "-"*40)
    print("5. 繪製增強分析圖表")
    print("-"*40)
    
    try:
        fig = analyzer.plot_enhanced_analysis()
        print("圖表繪製完成！")
    except Exception as e:
        print(f"圖表繪製失敗：{e}")
    
    # 6. 輸出關鍵洞察
    print("\n" + "="*60)
    print("關鍵分析洞察")
    print("="*60)
    
    _print_key_insights(analyzer)
    
    return analyzer, comprehensive_report

def _print_key_insights(analyzer):
    """輸出關鍵洞察"""
    results = analyzer.analysis_results
    
    print("\n📊 風險閥門分析洞察：")
    if 'risk_valve' in results:
        risk_data = results['risk_valve']
        print(f"  • 風險閥門觸發期間：{risk_data.get('risk_periods_count', 0)} 個")
        print(f"  • 風險期間交易數：{risk_data.get('risk_trades_count', 0)} 筆")
        print(f"  • 正常期間交易數：{risk_data.get('normal_trades_count', 0)} 筆")
        
        improvement = risk_data.get('improvement_potential', {})
        if improvement:
            print(f"  • MDD改善潛力：{improvement.get('mdd_reduction', 0):.2%}")
            print(f"  • PF改善潛力：{improvement.get('pf_improvement', 0):.2f}")
    
    print("\n🔄 交易階段分析洞察：")
    if 'phase_analysis' in results:
        phase_data = results['phase_analysis']
        for phase_type, phase_info in phase_data.items():
            print(f"  • {phase_type} 階段：{phase_info.get('trade_count', 0)} 筆交易")
            print(f"    貢獻比例：{phase_info.get('contribution_ratio', 0):.2%}")
            print(f"    期間：{phase_info.get('start_date', 'N/A')} 到 {phase_info.get('end_date', 'N/A')}")
    
    print("\n📈 加碼梯度優化洞察：")
    if 'gradient_optimization' in results:
        grad_data = results['gradient_optimization']
        current = grad_data.get('current_pattern', {})
        optimized = grad_data.get('optimized_pattern', {})
        
        print(f"  • 當前平均間距：{current.get('avg_interval', 0):.1f} 天")
        print(f"  • 最大連續加碼：{current.get('max_consecutive', 0)} 筆")
        print(f"  • 優化後加碼次數：從 {optimized.get('original_count', 0)} 減少到 {optimized.get('optimized_count', 0)}")
        print(f"  • 減少比例：{optimized.get('reduction_ratio', 0):.1%}")

def demonstrate_custom_risk_rules():
    """示範自定義風險規則"""
    print("\n" + "="*60)
    print("自定義風險規則示範")
    print("="*60)
    
    # 創建範例數據
    trades_df, benchmark_df = create_sample_data()
    analyzer = EnhancedTradeAnalyzer(trades_df, benchmark_df)
    
    # 自定義風險規則
    custom_rules = {
        'twii_slope_20d': {'threshold': -0.001, 'window': 20},  # 更嚴格的斜率要求
        'twii_slope_60d': {'threshold': -0.0005, 'window': 60},  # 更嚴格的長期趨勢
        'atr_threshold': {'window': 20, 'multiplier': 2.0},      # 更高的波動率門檻
        'volume_spike': {'window': 20, 'multiplier': 2.5}        # 新增成交量異常規則
    }
    
    print("使用自定義風險規則：")
    for rule_name, rule_params in custom_rules.items():
        print(f"  • {rule_name}: {rule_params}")
    
    # 執行自定義風險閥門回測
    custom_results = analyzer.risk_valve_backtest(custom_rules)
    
    return custom_results

def main():
    """主函數"""
    print("增強交易分析模組使用範例")
    print("路徑：#analysis/enhanced_analysis_example.py")
    print("創建時間：2025-08-18 04:38")
    
    try:
        # 執行基本分析
        analyzer, report = run_enhanced_analysis()
        
        # 示範自定義風險規則
        custom_results = demonstrate_custom_risk_rules()
        
        print("\n" + "="*60)
        print("分析完成！")
        print("="*60)
        print(f"總交易數：{report['total_trades']}")
        print(f"分析時間：{report['analysis_timestamp']}")
        print("\n建議：")
        print("1. 根據風險閥門分析結果調整加碼策略")
        print("2. 利用交易階段分析優化進出場時機")
        print("3. 實施加碼梯度優化降低過度交易風險")
        
    except Exception as e:
        print(f"分析過程中發生錯誤：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
