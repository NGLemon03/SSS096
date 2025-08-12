#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble 工作流程示例腳本

這個腳本展示瞭如何按順序運行整個工作流程
"""

import logging
import subprocess
import sys
from pathlib import Path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """運行命令並處理結果"""
    logger.info(f"執行: {description}")
    logger.info(f"命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            logger.info(f"✓ {description} 成功")
            if result.stdout:
                logger.info("輸出:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
        else:
            logger.error(f"✗ {description} 失敗")
            if result.stderr:
                logger.error("錯誤信息:")
                for line in result.stderr.strip().split('\n'):
                    if line.strip():
                        logger.error(f"  {line}")
            return False
            
    except Exception as e:
        logger.error(f"✗ 執行命令時發生異常: {e}")
        return False
    
    return True

def main():
    """主函數"""
    logger.info("開始運行 Ensemble 工作流程...")
    
    # 檢查當前目錄
    current_dir = Path.cwd()
    logger.info(f"當前工作目錄: {current_dir}")
    
    # 檢查必要文件
    required_files = [
        "SSSv096.py",
        "ensemble_wrapper.py", 
        "convert_results_to_trades.py",
        "run_enhanced_ensemble.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"缺少必要文件: {missing_files}")
        logger.error("請確保在正確的目錄中運行此腳本")
        return False
    
    # 步驟 1: 轉換 Optuna 結果為 Trades
    logger.info("\n" + "="*60)
    logger.info("步驟 1: 轉換 Optuna 結果為 Trades")
    logger.info("="*60)
    
    convert_cmd = [
        "python", "convert_results_to_trades.py",
        "--results_dir", "results",
        "--output_dir", "sss_backtest_outputs", 
        "--top_k", "5",
        "--ticker", "00631L.TW"
    ]
    
    if not run_command(" ".join(convert_cmd), "轉換 Optuna 結果為 Trades"):
        logger.error("步驟 1 失敗，停止工作流程")
        return False
    
    # 檢查輸出目錄
    output_dir = Path("sss_backtest_outputs")
    if not output_dir.exists():
        logger.error("輸出目錄不存在，轉換可能失敗")
        return False
    
    trades_files = list(output_dir.glob("trades_from_results_*.csv"))
    if not trades_files:
        logger.error("未找到轉換後的 trades 文件")
        return False
    
    logger.info(f"找到 {len(trades_files)} 個轉換後的 trades 文件")
    
    # 步驟 2: 運行增強版 Ensemble
    logger.info("\n" + "="*60)
    logger.info("步驟 2: 運行增強版 Ensemble")
    logger.info("="*60)
    
    ensemble_cmd = [
        "python", "run_enhanced_ensemble.py",
        "--method", "majority",
        "--scan_params",
        "--trades_dir", "sss_backtest_outputs"
    ]
    
    if not run_command(" ".join(ensemble_cmd), "運行增強版 Ensemble"):
        logger.error("步驟 2 失敗")
        return False
    
    # 檢查最終輸出
    logger.info("\n" + "="*60)
    logger.info("工作流程完成檢查")
    logger.info("="*60)
    
    ensemble_files = list(output_dir.glob("ensemble_*.csv"))
    if ensemble_files:
        logger.info(f"✓ 找到 {len(ensemble_files)} 個 ensemble 輸出文件:")
        for file in ensemble_files:
            logger.info(f"  - {file.name}")
    else:
        logger.warning("⚠ 未找到 ensemble 輸出文件")
    
    logger.info("\n🎉 Ensemble 工作流程執行完成！")
    logger.info("請檢查 sss_backtest_outputs 目錄中的結果文件")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
