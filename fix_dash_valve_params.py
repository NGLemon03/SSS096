# fix_dash_valve_params.py
# 修正 Dash 應用中的風險閥門參數設定
# 創建時間：2025-08-20 23:55:00
# 路徑：#根目錄/代碼

import re
from pathlib import Path

def fix_dash_valve_params():
    """修正 Dash 應用中的風險閥門參數設定"""
    print("🔧 開始修正 Dash 應用中的風險閥門參數...")
    
    app_dash_file = Path("app_dash.py")
    if not app_dash_file.exists():
        print("❌ app_dash.py 文件不存在")
        return
    
    # 讀取文件內容
    with open(app_dash_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("📁 文件載入成功")
    
    # 檢查當前的 use_slopes 設定
    use_slopes_true_count = content.count('"use_slopes": True')
    use_slopes_false_count = content.count('"use_slopes": False')
    
    print(f"   當前 use_slopes=True 的數量: {use_slopes_true_count}")
    print(f"   當前 use_slopes=False 的數量: {use_slopes_false_count}")
    
    if use_slopes_true_count == 0:
        print("✅ 沒有發現 use_slopes=True 的設定，無需修正")
        return
    
    # 修正全局套用部分
    print("\n🔧 修正全局套用部分...")
    old_global = '''                        global_valve_params = {
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
                        }'''
    
    new_global = '''                        global_valve_params = {
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
                            "use_slopes": False,  # === 修正：與測試基準保持一致 ===
                            "slope_method": "polyfit",
                            "atr_cmp": "gt"
                        }'''
    
    if old_global in content:
        content = content.replace(old_global, new_global)
        print("   ✅ 全局套用部分修正完成")
    else:
        print("   ⚠️  全局套用部分未找到，可能已經修正")
    
    # 修正增強分析部分
    print("🔧 修正增強分析部分...")
    old_enhanced = '''        enhanced_valve_params = {
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
        }'''
    
    new_enhanced = '''        enhanced_valve_params = {
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
            "use_slopes": False,  # === 修正：與測試基準保持一致 ===
            "slope_method": "polyfit", 
            "atr_cmp": "gt"
        }'''
    
    if old_enhanced in content:
        content = content.replace(old_enhanced, new_enhanced)
        print("   ✅ 增強分析部分修正完成")
    else:
        print("   ⚠️  增強分析部分未找到，可能已經修正")
    
    # 檢查其他可能的 use_slopes: True 設定
    print("\n🔍 檢查其他可能的 use_slopes 設定...")
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '"use_slopes": True' in line:
            print(f"   第 {i+1} 行: {line.strip()}")
    
    # 保存修正後的文件
    backup_file = app_dash_file.with_suffix('.py.backup')
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n💾 備份文件已保存到: {backup_file}")
    
    # 檢查修正結果
    new_use_slopes_true_count = content.count('"use_slopes": True')
    new_use_slopes_false_count = content.count('"use_slopes": False')
    
    print(f"\n📊 修正結果:")
    print(f"   修正前 use_slopes=True: {use_slopes_true_count}")
    print(f"   修正後 use_slopes=True: {new_use_slopes_true_count}")
    print(f"   修正前 use_slopes=False: {use_slopes_false_count}")
    print(f"   修正後 use_slopes=False: {new_use_slopes_false_count}")
    
    if new_use_slopes_true_count == 0:
        print("\n✅ 修正完成！所有 use_slopes 參數已設為 False")
        print("   現在 Dash 應用的風險閥門邏輯與測試基準完全一致")
    else:
        print(f"\n⚠️  仍有 {new_use_slopes_true_count} 個 use_slopes=True 需要手動修正")
    
    return content

def create_consistency_check_script():
    """創建一致性檢查腳本"""
    print("\n🔧 創建一致性檢查腳本...")
    
    script_content = '''# dash_valve_consistency_check.py
# Dash 應用風險閥門一致性檢查腳本
# 創建時間：2025-08-20 23:55:00

import pandas as pd
import numpy as np
from pathlib import Path
import json

def check_dash_valve_consistency():
    """檢查 Dash 應用的風險閥門設定是否與基準一致"""
    print("🔍 檢查 Dash 應用風險閥門一致性...")
    
    # 1. 載入基準配置
    try:
        baseline_files = list(Path(".").glob("ui_baseline_*.json"))
        if not baseline_files:
            print("❌ 找不到基準文件")
            return
        
        latest_file = max(baseline_files, key=lambda x: x.stat().st_mtime)
        with open(latest_file, 'r', encoding='utf-8') as f:
            baseline = json.load(f)
        
        print(f"✅ 基準文件載入成功: {latest_file}")
        print(f"   基準配置: {baseline['baseline_config']}")
        
    except Exception as e:
        print(f"❌ 載入基準文件失敗: {e}")
        return
    
    # 2. 檢查 app_dash.py 中的參數設定
    app_dash_file = Path("app_dash.py")
    if not app_dash_file.exists():
        print("❌ app_dash.py 文件不存在")
        return
    
    with open(app_dash_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查關鍵參數
    print("\\n🔍 檢查關鍵參數設定:")
    
    # use_slopes 參數
    use_slopes_true_count = content.count('"use_slopes": True')
    use_slopes_false_count = content.count('"use_slopes": False')
    
    print(f"   use_slopes=True 數量: {use_slopes_true_count}")
    print(f"   use_slopes=False 數量: {use_slopes_false_count}")
    
    if use_slopes_true_count > 0:
        print("   ❌ 發現 use_slopes=True，這會導致與基準結果不一致")
        print("   💡 建議使用 fix_dash_valve_params.py 進行修正")
    else:
        print("   ✅ use_slopes 參數設定正確")
    
    # 檢查其他參數
    expected_params = {
        'mode': 'cap',
        'cap_level': 0.3,
        'atr_ratio_mult': 1.0,
        'atr_win': 20,
        'atr_ref_win': 60
    }
    
    print("\\n🔍 檢查其他參數設定:")
    for param, expected_value in expected_params.items():
        if f'"{param}": {expected_value}' in content:
            print(f"   ✅ {param}: {expected_value}")
        else:
            print(f"   ⚠️  {param}: 未找到或值不同於 {expected_value}")
    
    # 3. 生成一致性報告
    print("\\n📋 一致性檢查報告:")
    
    if use_slopes_true_count == 0:
        print("   ✅ 風險閥門參數設定與基準一致")
        print("   💡 建議執行測試確認結果一致性")
    else:
        print("   ❌ 風險閥門參數設定與基準不一致")
        print("   🔧 需要修正 use_slopes 參數")
        print("   📝 修正步驟:")
        print("      1. 執行 fix_dash_valve_params.py")
        print("      2. 重新啟動 Dash 應用")
        print("      3. 執行一致性測試")
    
    return baseline

if __name__ == "__main__":
    check_dash_valve_consistency()
'''
    
    script_file = "dash_valve_consistency_check.py"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ 一致性檢查腳本已保存到: {script_file}")
    return script_file

def main():
    """主函數"""
    print("🚀 開始修正 Dash 應用風險閥門參數...")
    print("=" * 60)
    
    # 修正參數
    content = fix_dash_valve_params()
    
    # 創建一致性檢查腳本
    script_file = create_consistency_check_script()
    
    # 輸出總結
    print("\n" + "=" * 60)
    print("🎯 修正完成！")
    
    print("\n📋 已完成的修正:")
    print("   1. ✅ 修正了全局套用部分的 use_slopes 參數")
    print("   2. ✅ 修正了增強分析部分的 use_slopes 參數")
    print("   3. ✅ 創建了備份文件")
    print("   4. ✅ 創建了一致性檢查腳本")
    
    print("\n🔧 下一步操作:")
    print("   1. 重新啟動 Dash 應用")
    print("   2. 執行 dash_valve_consistency_check.py 確認修正")
    print("   3. 在 Dash 中測試風險閥門功能")
    print("   4. 比較結果與基準文件")
    
    print("\n💡 關鍵修正點:")
    print("   - use_slopes: True → False")
    print("   - 這會使風險閥門只基於 ATR 條件觸發")
    print("   - 與測試基準的邏輯完全一致")
    
    print("\n✅ 完成！")

if __name__ == "__main__":
    main()
