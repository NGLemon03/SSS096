#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強 Ensemble 策略運行腳本

這個腳本會:
1. 自動從新生成的 trades_from_results_*.csv 中選擇策略
2. 運行 ensemble 策略（Majority 或 Proportional）
3. 進行小網格掃描優化參數
4. 輸出最佳參數組合
5. 支持交易成本設置（與 SSS 默認設置一致）

使用方法:
python run_enhanced_ensemble.py --method majority --top_k 5 --scan_params
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import os

# 建立 logger（檔案最上方 import 之後）
logger = logging.getLogger(__name__)

# 全局費率常數（與 SSS 保持一致）
BASE_FEE_RATE = 0.001425  # 基礎手續費 = 0.1425%
TAX_RATE = 0.003          # 賣出交易税率 = 0.3%
DEFAULT_DISCOUNT = 0.30   # 預設折扣 = 0.3

def setup_logging(method: str, scan_params: bool = False) -> tuple:
    """
    設置日誌記錄，同時輸出到控制枱和文件
    
    Args:
        method: 集成方法名稱
        scan_params: 是否進行參數掃描
    
    Returns:
        (logger, log_file_path) 元組
    """
    # 創建日誌目錄
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    
    # 生成日誌文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scan_suffix = "_scan" if scan_params else ""
    log_filename = f"ensemble_{method}{scan_suffix}_{timestamp}.log"
    log_file_path = log_dir / log_filename
    
    # 創建 logger
    logger = logging.getLogger(f"ensemble_{method}")
    logger.setLevel(logging.INFO)
    
    # 清除現有的 handlers
    logger.handlers.clear()
    
    # 創建文件 handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 創建控制枱 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 創建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加 handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file_path

def get_default_cost_params(discount: float = None) -> dict:
    """
    獲取默認的交易成本參數
    
    Args:
        discount: 折扣率，None 表示使用默認值
    
    Returns:
        交易成本參數字典
    """
    if discount is None:
        discount = DEFAULT_DISCOUNT
    
    # 計算實際費率（與 SSS 保持一致）
    buy_fee_rate = BASE_FEE_RATE * discount
    sell_fee_rate = BASE_FEE_RATE * discount
    sell_tax_rate = TAX_RATE
    
    # 轉換為基點 (bp)
    buy_fee_bp = buy_fee_rate * 10000
    sell_fee_bp = sell_fee_rate * 10000
    sell_tax_bp = sell_tax_rate * 10000
    
    return {
        'buy_fee_bp': buy_fee_bp,
        'sell_fee_bp': sell_fee_bp,
        'sell_tax_bp': sell_tax_bp
    }

def format_performance_display(total_return: float, num_trades: int = None) -> str:
    """
    格式化性能顯示，同時顯示百分比和倍數
    
    Args:
        total_return: 總報酬率
        num_trades: 交易次數（可選）
    
    Returns:
        格式化的性能字符串
    """
    # 百分比格式
    percentage = total_return * 100
    
    # 倍數格式
    multiple = total_return + 1.0
    
    # 構建顯示字符串
    if num_trades is not None:
        return f"報酬率: {percentage:+.2f}% (倍數: {multiple:.3f}x), 交易次數: {num_trades}"
    else:
        return f"報酬率: {percentage:+.2f}% (倍數: {multiple:.3f}x)"

def save_final_results(result: dict, method: str, params: dict, cost_params: dict, log_file_path: Path) -> Path:
    """
    保存最終結果到單獨的文件
    
    Args:
        result: 結果字典
        method: 集成方法
        params: 使用的參數
        cost_params: 交易成本參數
        log_file_path: 日誌文件路徑（用於生成結果文件名）
    
    Returns:
        結果文件路徑
    """
    # 創建結果目錄
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 生成結果文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"ensemble_results_{method}_{timestamp}.txt"
    results_file_path = results_dir / results_filename
    
    # 寫入結果
    with open(results_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Ensemble 策略最終結果報告\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"集成方法: {method}\n")
        f.write(f"參數設置: {params}\n")
        f.write(f"交易成本: {cost_params}\n")
        f.write(f"日誌文件: {log_file_path.name}\n")
        f.write(f"\n")
        
        if result:
            f.write(f"權益曲線長度: {len(result['equity'])}\n")
            f.write(f"交易記錄數: {result['num_trades']}\n")
            f.write(f"總報酬率: {format_performance_display(result['total_return'], result['num_trades'])}\n")
            
            # 如果有統計信息，也保存
            if 'stats' in result and result['stats']:
                f.write(f"\n績效指標:\n")
                for key, value in result['stats'].items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
        else:
            f.write(f"運行失敗，無結果\n")
    
    return results_file_path

def run_ensemble_with_strategies(method: str, 
                                strategies: list,
                                params: dict,
                                cost_params: dict,
                                trades_dir: str = "sss_backtest_outputs") -> dict:
    """
    運行 ensemble 策略
    
    Args:
        method: 集成方法 ("majority" 或 "proportional")
        strategies: 策略列表
        params: ensemble 參數
        cost_params: 交易成本參數
        trades_dir: trades 文件目錄
    
    Returns:
        包含結果的字典
    """
    try:
        from ensemble_wrapper import EnsembleStrategyWrapper
        
        wrapper = EnsembleStrategyWrapper(trades_dir=trades_dir)
        
        # 運行 ensemble 策略
        equity, trades, stats, method_name = wrapper.ensemble_strategy(
            method=method,
            params=params,
            ticker="00631L.TW",
            strategies=strategies,
            cost_params=cost_params
        )
        
        return {
            'equity': equity,
            'trades': trades,
            'stats': stats,
            'method_name': method_name,
            'num_trades': len(trades) if not trades.empty else 0,
            'total_return': equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"運行 ensemble 策略失敗: {e}")
        return None

def scan_ensemble_parameters(method: str,
                           strategies: list,
                           cost_params: dict,
                           trades_dir: str = "sss_backtest_outputs",
                           logger=None) -> dict:
    """
    掃描 ensemble 參數，尋找最佳組合
    
    Args:
        method: 集成方法
        strategies: 策略列表
        cost_params: 交易成本參數
        trades_dir: trades 文件目錄
        logger: 日誌記錄器
    
    Returns:
        最佳參數組合
    """
    logger.info("開始參數掃描...")
    
    # 參數網格
    param_grid = {
        'min_cooldown_days': [1, 3, 5],
        'delta_cap': [0.10, 0.20, 0.30],
        'min_trade_dw': [0.01, 0.02, 0.03]
    }
    
    # 固定參數 - 根據方法決定是否包含 majority_k
    base_params = {
        'floor': 0.2,
        'ema_span': 3
    }
    
    # 只有 majority 方法才添加 majority_k 參數
    if method == 'majority':
        base_params['majority_k'] = max(1, int(len(strategies) * 0.55))
    
    best_result = None
    best_score = -np.inf
    best_params = None
    
    total_combinations = len(param_grid['min_cooldown_days']) * len(param_grid['delta_cap']) * len(param_grid['min_trade_dw'])
    current_combination = 0
    
    for cooldown in param_grid['min_cooldown_days']:
        for delta_cap in param_grid['delta_cap']:
            for min_trade_dw in param_grid['min_trade_dw']:
                current_combination += 1
                
                # 構建參數
                params = base_params.copy()
                params.update({
                    'min_cooldown_days': cooldown,
                    'delta_cap': delta_cap,
                    'min_trade_dw': min_trade_dw
                })
                
                logger.info(f"測試參數組合 {current_combination}/{total_combinations}: {params}")
                
                try:
                    # 運行 ensemble 策略
                    result = run_ensemble_with_strategies(method, strategies, params, cost_params, trades_dir)
                    
                    if result is not None:
                        # 計算評分（在總報酬率≥基準-2%的條件下，交易次數最少者勝）
                        total_return = result['total_return']
                        num_trades = result['num_trades']
                        
                        # 如果報酬率太低，跳過
                        if total_return < -0.02:  # 基準-2%
                            logger.info(f"  報酬率過低 ({format_performance_display(total_return)})，跳過")
                            continue
                        
                        # 評分：報酬率 - 交易次數懲罰
                        score = total_return - (num_trades * 0.001)  # 每次交易懲罰0.1%
                        
                        logger.info(f"  結果: {format_performance_display(total_return, num_trades)}, 評分={score:.4f}")
                        
                        if score > best_score:
                            best_score = score
                            best_result = result
                            best_params = params
                            logger.info(f"  *** 新的最佳結果 ***")
                    
                except Exception as e:
                    logger.warning(f"  參數組合失敗: {e}")
                    continue
    
    if best_params is not None:
        logger.info(f"\n最佳參數組合: {best_params}")
        logger.info(f"最佳評分: {best_score:.4f}")
        logger.info(f"最佳結果: {format_performance_display(best_result['total_return'], best_result['num_trades'])}")
    else:
        logger.warning("未找到有效的參數組合")
    
    return best_params, best_result

def main():
    parser = argparse.ArgumentParser(description='運行增強的 Ensemble 策略')
    parser.add_argument('--method', choices=['majority', 'proportional'], default='majority', 
                       help='集成方法')
    parser.add_argument('--top_k', type=int, default=5, 
                       help='每個策略取前K個最佳結果')
    parser.add_argument('--scan_params', action='store_true', 
                       help='是否進行參數掃描')
    parser.add_argument('--trades_dir', default='sss_backtest_outputs', 
                       help='trades 文件目錄')
    parser.add_argument('--results_dir', default='results', 
                       help='results 文件目錄')
    parser.add_argument('--discount', type=float, default=DEFAULT_DISCOUNT,
                       help=f'交易成本折扣率 (默認: {DEFAULT_DISCOUNT})')
    parser.add_argument('--no_cost', action='store_true',
                       help='不使用交易成本（僅用於對比測試）')
    
    args = parser.parse_args()
    
    # 設置日誌記錄
    logger, log_file_path = setup_logging(args.method, args.scan_params)
    logger.info(f"開始運行 Ensemble 策略 - 方法: {args.method}")
    logger.info(f"日誌文件: {log_file_path}")
    
    # 設置交易成本參數
    if args.no_cost:
        cost_params = get_default_cost_params(0.0)  # 無成本
        logger.info("交易成本: 無成本模式（僅用於對比測試）")
    else:
        cost_params = get_default_cost_params(args.discount)
        logger.info(f"交易成本: 折扣率={args.discount}, 買進費率={cost_params['buy_fee_bp']:.2f}bp, 賣出費率={cost_params['sell_fee_bp']:.2f}bp, 税率={cost_params['sell_tax_bp']:.2f}bp")
    
    # 檢查是否有新生成的 trades 文件
    trades_path = Path(args.trades_dir)
    # 支持兩種文件名格式：trades_from_results_*.csv 和 trades_*.csv
    new_trades_files = list(trades_path.glob("trades_from_results_*.csv"))
    if not new_trades_files:
        # 如果沒有找到 trades_from_results_*.csv，嘗試尋找 trades_*.csv
        new_trades_files = list(trades_path.glob("trades_*.csv"))
        # 過濾掉 ensemble_debug_*.csv 等非策略文件
        new_trades_files = [f for f in new_trades_files if not f.name.startswith("ensemble_debug_")]
    
    if not new_trades_files:
        logger.info("未找到新生成的 trades*.csv 文件")
        logger.info("請先運行 convert_results_to_trades.py 生成 trades 文件")
        return
    
    logger.info(f"找到 {len(new_trades_files)} 個新生成的 trades 文件")
    
    # 提取策略名稱
    strategies = []
    for file_path in new_trades_files:
        strategy_name = file_path.stem
        if strategy_name.startswith("trades_from_results_"):
            strategy_name = strategy_name.replace("trades_from_results_", "")
        elif strategy_name.startswith("trades_"):
            strategy_name = strategy_name.replace("trades_", "")
        
        # 剝掉 from_results_ 前綴（如果還有的話）
        strategy_name = strategy_name.replace('from_results_', '')
        
        # 正規化策略名稱：將底線轉換為空白，讓兩邊名稱能 match
        strategy_name = strategy_name.replace("_", " ")
        strategies.append(strategy_name)
    
    logger.info(f"可用策略: {strategies}")
    
    # 設置基礎參數
    base_params = {
        'floor': 0.2,
        'ema_span': 3,
        'delta_cap': 0.10,
        'min_cooldown_days': 5,
        'min_trade_dw': 0.02
    }
    
    if args.method == 'majority':
        base_params['majority_k'] = max(1, int(len(strategies) * 0.55))
    
    logger.info(f"基礎參數: {base_params}")
    
    if args.scan_params:
        # 進行參數掃描
        logger.info("開始參數掃描優化...")
        best_params, best_result = scan_ensemble_parameters(
            args.method, strategies, cost_params, args.trades_dir, logger
        )
        
        if best_params is not None:
            # 使用最佳參數運行一次
            logger.info("\n使用最佳參數運行 ensemble 策略...")
            final_result = run_ensemble_with_strategies(
                args.method, strategies, best_params, cost_params, args.trades_dir
            )
            
            if final_result:
                logger.info("最終結果:")
                logger.info(f"  權益曲線長度: {len(final_result['equity'])}")
                logger.info(f"  交易記錄數: {final_result['num_trades']}")
                logger.info(f"  總報酬率: {format_performance_display(final_result['total_return'], final_result['num_trades'])}")
                logger.info(f"  績效指標: {final_result['stats']}")
                
                # 保存最終結果到單獨文件
                results_file_path = save_final_results(final_result, args.method, best_params, cost_params, log_file_path)
                logger.info(f"結果已保存到: {results_file_path}")
        else:
            logger.warning("參數掃描未找到有效結果")
    else:
        # 使用基礎參數運行
        logger.info("使用基礎參數運行 ensemble 策略...")
        result = run_ensemble_with_strategies(
            args.method, strategies, base_params, cost_params, args.trades_dir
        )
        
        if result:
            logger.info("運行結果:")
            logger.info(f"  權益曲線長度: {len(result['equity'])}")
            logger.info(f"  交易記錄數: {result['num_trades']}")
            logger.info(f"  總報酬率: {format_performance_display(result['total_return'], result['num_trades'])}")
            logger.info(f"  績效指標: {result['stats']}")
            
            # 保存最終結果到單獨文件
            results_file_path = save_final_results(result, args.method, base_params, cost_params, log_file_path)
            logger.info(f"結果已保存到: {results_file_path}")
        else:
            logger.error("策略運行失敗")
    
    logger.info("Ensemble 策略運行完成")


if __name__ == "__main__":
    main()
