#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
結果轉換器腳本 - 將 results 目錄下的 optuna_results_*.csv 轉換為 trades_*.csv

使用方法:
python convert_results_to_trades.py --results_dir results --output_dir sss_backtest_outputs --top_k 5

這個腳本會:
1. 讀取 results 目錄下的所有 optuna_results_*.csv 文件
2. 按 score 排序，每個策略取前 K 個最佳結果
3. 解析參數並調用回測函數
4. 生成 trades_from_results_*.csv 文件
5. 輸出策略列表供 ensemble 使用
"""

import argparse
import logging
import sys
from pathlib import Path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='將 optuna 結果轉換為 trades 文件')
    parser.add_argument('--results_dir', default='results', help='包含 optuna_results_*.csv 的目錄')
    parser.add_argument('--output_dir', default='sss_backtest_outputs', help='輸出 trades_*.csv 的目錄')
    parser.add_argument('--top_k', type=int, default=5, help='每個策略取前K個最佳結果')
    parser.add_argument('--ticker', default='00631L.TW', help='股票代碼')
    parser.add_argument('--dry_run', action='store_true', help='只顯示會處理的策略，不實際轉換')
    
    args = parser.parse_args()
    
    # 驗證輸入目錄
    results_path = Path(args.results_dir)
    if not results_path.exists():
        logger.error(f"結果目錄不存在: {args.results_dir}")
        sys.exit(1)
    
    # 導入轉換函數
    try:
        from ensemble_wrapper import convert_optuna_results_to_trades, select_top_strategies_from_results
    except ImportError:
        logger.error("無法導入 ensemble_wrapper，請確保 ensemble_wrapper.py 在當前目錄")
        sys.exit(1)
    
    if args.dry_run:
        # 只顯示會處理的策略
        logger.info("DRY RUN 模式 - 只顯示會處理的策略")
        selected_strategies = select_top_strategies_from_results(
            results_dir=args.results_dir,
            top_k_per_strategy=args.top_k
        )
        
        logger.info(f"會處理 {len(selected_strategies)} 個策略:")
        for i, strategy in enumerate(selected_strategies, 1):
            logger.info(f"  {i:2d}. {strategy}")
        
        return
    
    # 執行轉換
    logger.info(f"開始轉換結果文件...")
    logger.info(f"結果目錄: {args.results_dir}")
    logger.info(f"輸出目錄: {args.output_dir}")
    logger.info(f"每個策略取前 {args.top_k} 個最佳結果")
    
    try:
        generated_strategies = convert_optuna_results_to_trades(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            top_k_per_strategy=args.top_k,
            ticker=args.ticker
        )
        
        logger.info(f"轉換完成！生成了 {len(generated_strategies)} 個策略")
        
        # 輸出策略列表
        logger.info("生成的策略列表:")
        for i, strategy in enumerate(generated_strategies, 1):
            logger.info(f"  {i:2d}. {strategy}")
        
        # 輸出使用説明
        logger.info("\n" + "="*60)
        logger.info("使用説明:")
        logger.info("1. 這些策略現在可以在 ensemble 中使用了")
        logger.info("2. 策略名稱格式: trades_from_results_<strategy>_<datasource>_trial<id>.csv")
        logger.info("3. 在 ensemble 中指定 strategies 參數時，使用上述策略名稱")
        logger.info("4. 建議的 ensemble 參數:")
        logger.info("   - majority_k = ceil(len(strategies) * 0.55)")
        logger.info("   - floor = 0.2, ema_span = 3")
        logger.info("   - delta_cap = 0.10, min_cooldown_days = 5, min_trade_dw = 0.02")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"轉換失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
