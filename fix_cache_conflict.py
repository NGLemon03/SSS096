import os
import shutil
from pathlib import Path
import sys

def fix_cache_conflicts():
    """修復快取衝突問題"""
    
    print("=== 快取衝突修復工具 ===")
    
    # 定義所有可能的快取目錄
    cache_dirs = [
        "cache",
        "cache_smaa", 
        "cache/cache_smaa",
        "cache/joblib",
        "analysis/cache",
        "analysis/cache_smaa"
    ]
    
    print("檢查快取目錄...")
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            size = sum(f.stat().st_size for f in Path(cache_dir).rglob('*') if f.is_file())
            print(f"  {cache_dir}: {size / 1024 / 1024:.2f} MB")
        else:
            print(f"  {cache_dir}: 不存在")
    
    print("\n建議的修復步驟:")
    print("1. 停止所有正在運行的程式")
    print("2. 清空所有快取目錄")
    print("3. 重新建立統一的快取結構")
    
    response = input("\n是否要清空所有快取目錄？(y/N): ")
    if response.lower() == 'y':
        print("\n清空快取目錄...")
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                    print(f"  已清空: {cache_dir}")
                except Exception as e:
                    print(f"  清空失敗 {cache_dir}: {e}")
        
        # 重新建立統一的快取結構
        print("\n重新建立快取結構...")
        os.makedirs("cache", exist_ok=True)
        os.makedirs("cache/cache_smaa", exist_ok=True)
        os.makedirs("cache/joblib", exist_ok=True)
        print("  已建立: cache/")
        print("  已建立: cache/cache_smaa/")
        print("  已建立: cache/joblib/")
        
        print("\n快取修復完成！")
    else:
        print("取消操作")

def check_cache_consistency():
    """檢查快取一致性"""
    
    print("=== 快取一致性檢查 ===")
    
    # 檢查 config.py 中的快取設定
    print("檢查 config.py 快取設定...")
    try:
        from analysis.config import CACHE_DIR, SMAA_CACHE_DIR
        print(f"  CACHE_DIR: {CACHE_DIR}")
        print(f"  SMAA_CACHE_DIR: {SMAA_CACHE_DIR}")
        print(f"  CACHE_DIR 存在: {CACHE_DIR.exists()}")
        print(f"  SMAA_CACHE_DIR 存在: {SMAA_CACHE_DIR.exists()}")
    except ImportError as e:
        print(f"  無法導入 config: {e}")
    
    # 檢查實際目錄
    print("\n檢查實際目錄...")
    actual_dirs = [
        Path("cache"),
        Path("cache_smaa"),
        Path("cache/cache_smaa"),
        Path("cache/joblib")
    ]
    
    for dir_path in actual_dirs:
        if dir_path.exists():
            files = list(dir_path.rglob("*"))
            print(f"  {dir_path}: {len(files)} 個文件")
        else:
            print(f"  {dir_path}: 不存在")

def create_unified_cache_structure():
    """建立統一的快取結構"""
    
    print("=== 建立統一快取結構 ===")
    
    # 定義統一的快取結構
    cache_structure = {
        "cache": {
            "cache_smaa": {},  # SMAA 指標快取
            "joblib": {},      # Joblib 快取
            "backtest": {},    # 回測結果快取
            "temp": {}         # 臨時文件
        }
    }
    
    # 建立目錄
    for main_dir, sub_dirs in cache_structure.items():
        main_path = Path(main_dir)
        main_path.mkdir(exist_ok=True)
        print(f"建立目錄: {main_path}")
        
        for sub_dir in sub_dirs:
            sub_path = main_path / sub_dir
            sub_path.mkdir(exist_ok=True)
            print(f"  子目錄: {sub_path}")
    
    # 建立 .gitkeep 文件
    for dir_path in Path("cache").rglob("*"):
        if dir_path.is_dir():
            gitkeep_file = dir_path / ".gitkeep"
            gitkeep_file.touch(exist_ok=True)
    
    print("\n統一快取結構建立完成！")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "fix":
            fix_cache_conflicts()
        elif command == "check":
            check_cache_consistency()
        elif command == "create":
            create_unified_cache_structure()
        else:
            print("可用命令: fix, check, create")
    else:
        print("快取管理工具")
        print("用法: python fix_cache_conflict.py [fix|check|create]")
        print("  fix   - 修復快取衝突")
        print("  check - 檢查快取一致性")
        print("  create - 建立統一快取結構") 