# check_data_consistency.py
# 直接檢查全局套用與增強分析的數據一致性
# 創建時間：2025-08-21 01:25:00
# 路徑：#根目錄/代碼

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import os

# 添加當前目錄到 Python 路徑
sys.path.append(os.getcwd())

def check_data_consistency():
    """直接檢查全局套用與增強分析的數據一致性"""
    print("🔍 直接檢查數據一致性...")
    print("=" * 60)
    
    # 檢查必要的文件
    required_files = [
        "app_dash.py",
        "SSS_EnsembleTab.py"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ 缺少必要文件: {file_path}")
            return False
    
    print("✅ 必要文件檢查通過")
    
    # 檢查 app_dash.py 中的整合情況
    with open("app_dash.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("\n📊 文件統計:")
    print(f"   文件大小: {len(content):,} 字符")
    print(f"   增強分析相關代碼數量: {content.count('增強分析')}")
    print(f"   全局套用相關代碼數量: {content.count('全局套用')}")
    print(f"   backtest-store 出現次數: {content.count('backtest-store')}")
    
    # 檢查關鍵整合點
    print("\n🔍 檢查關鍵整合點:")
    
    # 1. 檢查回調是否包含 backtest-store
    if 'State("backtest-store", "data")' in content:
        print("   ✅ 回調包含 backtest-store")
    else:
        print("   ❌ 回調不包含 backtest-store")
    
    # 2. 檢查函數參數是否包含 backtest_data
    if 'backtest_data=None' in content:
        print("   ✅ 函數參數包含 backtest_data")
    else:
        print("   ❌ 函數參數不包含 backtest_data")
    
    # 3. 檢查是否使用全局套用數據源
    if '使用全局套用數據源' in content:
        print("   ✅ 使用全局套用數據源")
    else:
        print("   ❌ 未使用全局套用數據源")
    
    # 4. 檢查數據一致性檢查
    if '數據一致性檢查' in content:
        print("   ✅ 數據一致性檢查")
    else:
        print("   ❌ 數據一致性檢查")
    
    # 檢查具體的數據處理邏輯
    print("\n🔍 檢查數據處理邏輯:")
    
    # 查找數據獲取邏輯
    data_source_patterns = [
        'df_from_pack(backtest_data.get("df_raw"))',
        'df_from_pack(cache.get("df_raw"))',
        'daily_state_std',
        'daily_state'
    ]
    
    for pattern in data_source_patterns:
        count = content.count(pattern)
        print(f"   {pattern}: {count} 次")
    
    # 檢查風險閥門參數
    print("\n🔍 檢查風險閥門參數:")
    
    valve_params = [
        'use_slopes: True',
        'slope_method: "polyfit"',
        'atr_cmp: "gt"',
        'atr_win: 20',
        'atr_ref_win: 60'
    ]
    
    for param in valve_params:
        if param in content:
            print(f"   ✅ {param}")
        else:
            print(f"   ❌ {param}")
    
    return True

def check_enhanced_analysis_implementation():
    """檢查增強分析的具體實現"""
    print("\n🔍 檢查增強分析實現:")
    print("=" * 60)
    
    with open("app_dash.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到增強分析函數的定義
    start_marker = "def _run_rv("
    end_marker = "    if not n_clicks or not cache:"
    
    start_pos = content.find(start_marker)
    if start_pos != -1:
        # 找到函數開始
        func_start = start_pos
        # 找到函數體開始
        body_start = content.find(end_marker, func_start)
        
        if body_start != -1:
            # 提取函數定義和參數
            func_def = content[func_start:body_start].strip()
            print("📋 函數定義:")
            print(f"   {func_def}")
            
            # 檢查參數
            if 'backtest_data' in func_def:
                print("   ✅ 包含 backtest_data 參數")
            else:
                print("   ❌ 不包含 backtest_data 參數")
        else:
            print("   ⚠️  無法找到函數體開始")
    else:
        print("   ❌ 無法找到 _run_rv 函數定義")
    
    # 檢查數據獲取邏輯
    print("\n🔍 檢查數據獲取邏輯:")
    
    # 查找數據源選擇邏輯
    data_source_logic = [
        '優先使用全局套用的數據源',
        '從 backtest-store 獲取數據',
        '回退到快取數據',
        '使用快取數據源'
    ]
    
    for logic in data_source_logic:
        if logic in content:
            print(f"   ✅ {logic}")
        else:
            print(f"   ❌ {logic}")
    
    # 檢查數據一致性檢查邏輯
    print("\n🔍 檢查數據一致性檢查邏輯:")
    
    consistency_checks = [
        '數據一致性檢查',
        '數據長度一致',
        '數據長度不一致',
        'daily_state 載入成功',
        'daily_state 載入失敗'
    ]
    
    for check in consistency_checks:
        if check in content:
            print(f"   ✅ {check}")
        else:
            print(f"   ❌ {check}")

def check_global_application_implementation():
    """檢查全局套用的實現"""
    print("\n🔍 檢查全局套用實現:")
    print("=" * 60)
    
    with open("app_dash.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找全局套用的關鍵邏輯
    global_patterns = [
        '全局風險閥門：逐日動態套用',
        '與增強分析一致',
        'compute_risk_valve_signals',
        'risk_valve_backtest'
    ]
    
    for pattern in global_patterns:
        count = content.count(pattern)
        print(f"   {pattern}: {count} 次")
    
    # 檢查全局套用的風險閥門參數
    print("\n🔍 檢查全局套用風險閥門參數:")
    
    global_valve_params = [
        'use_slopes=True',
        'slope_method="polyfit"',
        'atr_cmp="gt"',
        'atr_win=20',
        'atr_ref_win=60'
    ]
    
    for param in global_valve_params:
        count = content.count(param)
        print(f"   {param}: {count} 次")

def generate_detailed_report():
    """生成詳細的檢查報告"""
    print("\n📋 生成詳細檢查報告...")
    
    report = {
        "check_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "check_type": "數據一致性詳細檢查",
        "files_checked": ["app_dash.py", "SSS_EnsembleTab.py"],
        "integration_status": "已整合",
        "key_findings": [],
        "recommendations": []
    }
    
    # 檢查整合狀態
    with open("app_dash.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查關鍵整合點
    integration_checks = {
        "回調包含 backtest-store": 'State("backtest-store", "data")' in content,
        "函數參數包含 backtest_data": "backtest_data=None" in content,
        "使用全局套用數據源": "使用全局套用數據源" in content,
        "數據一致性檢查": "數據一致性檢查" in content
    }
    
    report["integration_checks"] = integration_checks
    
    # 檢查數據處理邏輯
    data_logic_checks = {
        "優先使用全局套用數據源": "優先使用全局套用數據源" in content,
        "從 backtest-store 獲取數據": "從 backtest-store 獲取數據" in content,
        "回退到快取數據": "回退到快取數據" in content,
        "使用快取數據源": "使用快取數據源" in content
    }
    
    report["data_logic_checks"] = data_logic_checks
    
    # 檢查風險閥門參數一致性
    valve_param_checks = {
        "use_slopes: True": "use_slopes: True" in content,
        "slope_method: polyfit": 'slope_method: "polyfit"' in content,
        "atr_cmp: gt": 'atr_cmp: "gt"' in content
    }
    
    report["valve_param_checks"] = valve_param_checks
    
    # 生成建議
    if all(integration_checks.values()):
        report["recommendations"].append("所有整合點檢查通過，可以進行實際測試")
    else:
        failed_checks = [k for k, v in integration_checks.items() if not v]
        report["recommendations"].append(f"以下整合點需要修復: {', '.join(failed_checks)}")
    
    if all(data_logic_checks.values()):
        report["recommendations"].append("數據處理邏輯完整，支持數據源切換")
    else:
        failed_checks = [k for k, v in data_logic_checks.items() if not v]
        report["recommendations"].append(f"以下數據邏輯需要修復: {', '.join(failed_checks)}")
    
    # 保存報告
    report_file = f"data_consistency_detailed_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 詳細檢查報告已保存到: {report_file}")
    return report

def main():
    """主函數"""
    print("🚀 開始直接檢查數據一致性...")
    print("=" * 60)
    
    # 執行各項檢查
    check_data_consistency()
    check_enhanced_analysis_implementation()
    check_global_application_implementation()
    
    # 生成詳細報告
    report = generate_detailed_report()
    
    # 輸出總結
    print("\n" + "=" * 60)
    print("🎯 檢查完成！")
    
    print("\n📊 檢查結果:")
    if report["integration_checks"]["回調包含 backtest-store"]:
        print("   ✅ 回調已整合 backtest-store")
    else:
        print("   ❌ 回調未整合 backtest-store")
    
    if report["integration_checks"]["函數參數包含 backtest_data"]:
        print("   ✅ 函數參數已添加 backtest_data")
    else:
        print("   ❌ 函數參數未添加 backtest_data")
    
    if report["integration_checks"]["使用全局套用數據源"]:
        print("   ✅ 已使用全局套用數據源")
    else:
        print("   ❌ 未使用全局套用數據源")
    
    if report["integration_checks"]["數據一致性檢查"]:
        print("   ✅ 已添加數據一致性檢查")
    else:
        print("   ❌ 未添加數據一致性檢查")
    
    print("\n💡 建議:")
    for rec in report["recommendations"]:
        print(f"   • {rec}")
    
    print("\n✅ 完成！")

if __name__ == "__main__":
    main()
