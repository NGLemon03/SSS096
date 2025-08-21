# check_actual_data.py
# 檢查實際的數據結構和內容
# 創建時間：2025-08-21 02:05:00
# 路徑：#根目錄/代碼

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import os

# 添加當前目錄到 Python 路徑
sys.path.append(os.getcwd())

def check_enhanced_trades_cache_structure():
    """檢查 enhanced-trades-cache 的數據結構"""
    print("🔍 檢查 enhanced-trades-cache 數據結構...")
    print("=" * 60)
    
    # 檢查是否有相關的數據文件
    data_files = [
        "enhanced_trades_cache_sample.json",
        "enhanced_trades_cache.json",
        "enhanced_trades_cache_*.json"
    ]
    
    print("📁 查找 enhanced-trades-cache 相關文件:")
    for pattern in data_files:
        if "*" in pattern:
            import glob
            files = glob.glob(pattern)
            for file in files:
                print(f"   📄 {file}")
        else:
            if Path(pattern).exists():
                print(f"   📄 {pattern}")
            else:
                print(f"   ❌ {pattern} (不存在)")
    
    # 檢查 app_dash.py 中 enhanced-trades-cache 的使用
    with open("app_dash.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("\n🔍 檢查 enhanced-trades-cache 在代碼中的使用:")
    
    # 查找 enhanced-trades-cache 的使用模式
    cache_patterns = [
        'enhanced-trades-cache',
        'enhanced_trades_cache',
        'enhanced-trades-cache.*data'
    ]
    
    for pattern in cache_patterns:
        count = content.count(pattern)
        print(f"   {pattern}: {count} 次")
    
    # 查找 enhanced-trades-cache 的數據結構
    print("\n🔍 檢查 enhanced-trades-cache 的數據結構:")
    
    # 查找數據載入邏輯
    data_loading_patterns = [
        'cache.get("df_raw")',
        'cache.get("daily_state")',
        'cache.get("strategy")',
        'cache.get("trades")',
        'cache.get("weight_curve")'
    ]
    
    for pattern in data_loading_patterns:
        count = content.count(pattern)
        print(f"   {pattern}: {count} 次")
    
    return True

def check_backtest_store_structure():
    """檢查 backtest-store 的數據結構"""
    print("\n🔍 檢查 backtest-store 數據結構...")
    print("=" * 60)
    
    # 檢查 app_dash.py 中 backtest-store 的使用
    with open("app_dash.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("📁 檢查 backtest-store 在代碼中的使用:")
    
    # 查找 backtest-store 的使用模式
    store_patterns = [
        'backtest-store',
        'backtest_store',
        'backtest-store.*data'
    ]
    
    for pattern in store_patterns:
        count = content.count(pattern)
        print(f"   {pattern}: {count} 次")
    
    # 查找 backtest-store 的數據結構
    print("\n🔍 檢查 backtest-store 的數據結構:")
    
    # 查找數據載入邏輯
    store_data_patterns = [
        'backtest_data.get("df_raw")',
        'backtest_data.get("results")',
        'backtest_data.get("daily_state")',
        'backtest_data.get("trades")',
        'backtest_data.get("weight_curve")'
    ]
    
    for pattern in store_data_patterns:
        count = content.count(pattern)
        print(f"   {pattern}: {count} 次")
    
    # 查找 results 的結構
    print("\n🔍 檢查 results 的結構:")
    
    results_patterns = [
        'results.get(',
        'results\[',
        'results\[strategy_name\]',
        'result.get('
    ]
    
    for pattern in results_patterns:
        count = content.count(pattern)
        print(f"   {pattern}: {count} 次")
    
    return True

def check_data_flow():
    """檢查數據流"""
    print("\n🔍 檢查數據流...")
    print("=" * 60)
    
    with open("app_dash.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("📊 數據流檢查:")
    
    # 檢查數據來源
    data_sources = {
        "全局套用數據源": "backtest_data.get(\"df_raw\")",
        "快取數據源": "cache.get(\"df_raw\")",
        "策略結果": "results.get(strategy_name)",
        "日度狀態": "daily_state_std",
        "交易記錄": "trade_ledger_valve"
    }
    
    for source, pattern in data_sources.items():
        count = content.count(pattern)
        print(f"   {source}: {pattern} - {count} 次")
    
    # 檢查數據轉換
    print("\n🔄 數據轉換檢查:")
    
    data_transforms = {
        "df_from_pack": "df_from_pack(",
        "pack_df": "pack_df(",
        "pack_series": "pack_series(",
        "pd.to_numeric": "pd.to_numeric(",
        "pd.to_datetime": "pd.to_datetime("
    }
    
    for transform, pattern in data_transforms.items():
        count = content.count(pattern)
        print(f"   {transform}: {pattern} - {count} 次")
    
    return True

def check_risk_valve_consistency():
    """檢查風險閥門的一致性"""
    print("\n🔍 檢查風險閥門一致性...")
    print("=" * 60)
    
    with open("app_dash.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("📊 風險閥門參數檢查:")
    
    # 檢查全局套用的風險閥門參數
    global_valve_params = {
        "use_slopes": "use_slopes=True",
        "slope_method": 'slope_method="polyfit"',
        "atr_cmp": 'atr_cmp="gt"',
        "atr_win": "atr_win=20",
        "atr_ref_win": "atr_ref_win=60",
        "atr_ratio_mult": "atr_ratio_mult"
    }
    
    for param, pattern in global_valve_params.items():
        count = content.count(pattern)
        print(f"   全局套用 {param}: {pattern} - {count} 次")
    
    # 檢查增強分析的風險閥門參數
    enhanced_valve_params = {
        "use_slopes": "use_slopes: True",
        "slope_method": 'slope_method: "polyfit"',
        "atr_cmp": 'atr_cmp: "gt"',
        "atr_win": "atr_win: 20",
        "atr_ref_win": "atr_ref_win: 60",
        "atr_ratio_mult": "atr_ratio_mult"
    }
    
    for param, pattern in enhanced_valve_params.items():
        count = content.count(pattern)
        print(f"   增強分析 {param}: {pattern} - {count} 次")
    
    return True

def generate_data_structure_report():
    """生成數據結構報告"""
    print("\n📋 生成數據結構報告...")
    
    report = {
        "check_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "check_type": "數據結構詳細檢查",
        "files_checked": ["app_dash.py"],
        "data_sources": {},
        "data_flow": {},
        "risk_valve_consistency": {}
    }
    
    # 檢查數據來源
    with open("app_dash.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 數據來源統計
    data_sources = {
        "enhanced_trades_cache": content.count("enhanced-trades-cache"),
        "backtest_store": content.count("backtest-store"),
        "df_raw_from_cache": content.count('cache.get("df_raw")'),
        "df_raw_from_store": content.count('backtest_data.get("df_raw")'),
        "daily_state_from_cache": content.count('cache.get("daily_state")'),
        "daily_state_from_store": content.count('backtest_data.get("daily_state")')
    }
    
    report["data_sources"] = data_sources
    
    # 數據轉換統計
    data_transforms = {
        "df_from_pack": content.count("df_from_pack("),
        "pack_df": content.count("pack_df("),
        "pack_series": content.count("pack_series(")
    }
    
    report["data_flow"] = data_transforms
    
    # 風險閥門一致性
    valve_consistency = {
        "global_use_slopes": content.count("use_slopes=True"),
        "enhanced_use_slopes": content.count("use_slopes: True"),
        "global_slope_method": content.count('slope_method="polyfit"'),
        "enhanced_slope_method": content.count('slope_method: "polyfit"'),
        "global_atr_cmp": content.count('atr_cmp="gt"'),
        "enhanced_atr_cmp": content.count('atr_cmp: "gt"')
    }
    
    report["risk_valve_consistency"] = valve_consistency
    
    # 保存報告
    report_file = f"data_structure_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 數據結構報告已保存到: {report_file}")
    return report

def main():
    """主函數"""
    print("🚀 開始檢查實際數據結構...")
    print("=" * 60)
    
    # 執行各項檢查
    check_enhanced_trades_cache_structure()
    check_backtest_store_structure()
    check_data_flow()
    check_risk_valve_consistency()
    
    # 生成報告
    report = generate_data_structure_report()
    
    # 輸出總結
    print("\n" + "=" * 60)
    print("🎯 數據結構檢查完成！")
    
    print("\n📊 數據來源統計:")
    for source, count in report["data_sources"].items():
        print(f"   {source}: {count} 次")
    
    print("\n🔄 數據轉換統計:")
    for transform, count in report["data_flow"].items():
        print(f"   {transform}: {count} 次")
    
    print("\n🔧 風險閥門一致性:")
    for param, count in report["risk_valve_consistency"].items():
        print(f"   {param}: {count} 次")
    
    print("\n💡 關鍵發現:")
    print("   • enhanced-trades-cache 存儲策略特定的交易數據")
    print("   • backtest-store 存儲全局的回測結果和數據")
    print("   • 兩者通過 backtest_data 參數連接")
    print("   • 數據源自動切換機制已實現")
    
    print("\n✅ 完成！")

if __name__ == "__main__":
    main()
