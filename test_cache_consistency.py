import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加路徑
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SSSv095b2 import compute_single, load_data
from analysis.config import SMAA_CACHE_DIR

def test_cache_consistency():
    """測試快取一致性"""
    
    print("=== 快取一致性測試 ===")
    
    # Single 2 策略參數
    params = {
        'linlen': 90,
        'factor': 40,
        'smaalen': 30,
        'devwin': 30,
        'buy_mult': 1.45,
        'sell_mult': 1.25,
        'stop_loss': 0.2
    }
    
    print(f"測試策略參數: {params}")
    print()
    
    # 載入數據
    print("載入數據...")
    df_price, df_factor = load_data("00631L.TW", "2014-10-23", "2025-06-30", smaa_source="Self")
    print(f"價格數據形狀: {df_price.shape}")
    print(f"因子數據形狀: {df_factor.shape}")
    print()
    
    # 檢查數據哈希值
    print("檢查數據哈希值...")
    try:
        import pandas as pd
        df_cleaned = df_price.dropna(subset=['close'])
        df_cleaned['close'] = df_cleaned['close'].round(6)
        data_hash = str(pd.util.hash_pandas_object(df_cleaned['close']).sum())
        print(f"當前數據哈希值: {data_hash}")
    except Exception as e:
        print(f"計算哈希值失敗: {e}")
    
    # 檢查快取文件
    print("\n檢查快取文件...")
    cache_pattern = f"smaa_Self_unknown_{params['linlen']}_{params['factor']}_{params['smaalen']}_*.npy"
    cache_files = list(SMAA_CACHE_DIR.glob(cache_pattern))
    
    if cache_files:
        print(f"找到 {len(cache_files)} 個快取文件:")
        for cache_file in cache_files:
            print(f"  {cache_file.name}")
            # 檢查文件大小和修改時間
            stat = cache_file.stat()
            print(f"    大小: {stat.st_size} bytes")
            print(f"    修改時間: {pd.Timestamp(stat.st_mtime, unit='s')}")
    else:
        print("未找到對應的快取文件")
    
    # 計算指標（會觸發快取）
    print("\n計算指標...")
    df_ind = compute_single(df_price, df_factor, 
                           params['linlen'], params['factor'], 
                           params['smaalen'], params['devwin'])
    
    print(f"指標數據形狀: {df_ind.shape}")
    print(f"指標數據欄位: {list(df_ind.columns)}")
    
    # 檢查最後幾天的指標值
    print("\n最後10天的指標值:")
    print(df_ind[['close', 'smaa', 'base', 'sd']].tail(10))
    
    # 檢查買入信號
    print("\n檢查買入信號...")
    for i, (date, row) in enumerate(df_ind.tail(10).iterrows()):
        buy_threshold = row['base'] + row['sd'] * params['buy_mult']
        buy_condition = row['smaa'] < buy_threshold
        
        if buy_condition:
            print(f"  {date.strftime('%Y-%m-%d')}: 買入信號!")
            print(f"    SMAA: {row['smaa']:.3f}")
            print(f"    買入閾值: {buy_threshold:.3f}")
            print(f"    收盤價: {row['close']:.3f}")
    
    # 再次檢查快取文件
    print("\n重新檢查快取文件...")
    cache_files_after = list(SMAA_CACHE_DIR.glob(cache_pattern))
    if cache_files_after:
        print(f"計算後找到 {len(cache_files_after)} 個快取文件:")
        for cache_file in cache_files_after:
            print(f"  {cache_file.name}")
            stat = cache_file.stat()
            print(f"    大小: {stat.st_size} bytes")
            print(f"    修改時間: {pd.Timestamp(stat.st_mtime, unit='s')}")

def clear_smaa_cache():
    """清空 SMAA 快取"""
    print("=== 清空 SMAA 快取 ===")
    
    if SMAA_CACHE_DIR.exists():
        cache_files = list(SMAA_CACHE_DIR.glob("*.npy"))
        print(f"找到 {len(cache_files)} 個快取文件")
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                print(f"已刪除: {cache_file.name}")
            except Exception as e:
                print(f"刪除失敗 {cache_file.name}: {e}")
        
        print("SMAA 快取清空完成")
    else:
        print("SMAA 快取目錄不存在")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "test":
            test_cache_consistency()
        elif command == "clear":
            clear_smaa_cache()
        else:
            print("可用命令: test, clear")
    else:
        print("快取一致性測試工具")
        print("用法: python test_cache_consistency.py [test|clear]")
        print("  test  - 測試快取一致性")
        print("  clear - 清空 SMAA 快取") 