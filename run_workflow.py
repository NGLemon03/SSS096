#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一鍵式 Ensemble 工作流程執行腳本

這個腳本整合了完整的工作流程：
1. 結果轉換：將 Optuna 結果轉換為交易文件
2. Ensemble 執行：運行 Majority 和 Proportional 策略
3. 參數掃描：自動優化 Ensemble 參數
4. 結果彙總：生成綜合報告

使用方法:
python run_workflow.py --method both --top_k 5 --scan_params
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import yaml

# 設置日誌 - 按需初始化
try:
    from analysis.logging_config import get_logger
    logger = get_logger("workflow")
except ImportError:
    # 後備方案：如果無法導入專案日誌配置，使用基本配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

def load_config():
    """加載配置文件"""
    config_file = Path("config.yaml")
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"配置文件加載失敗: {e}，使用默認配置")
    
    # 默認配置
    return {
        'defaults': {
            'ticker': '00631L.TW',
            'top_k': 5,
            'results_dir': 'results',
            'output_dir': 'sss_backtest_outputs',
            'trades_dir': 'sss_backtest_outputs'
        }
    }

def run_command(cmd, description):
    """運行命令並處理結果"""
    logger.info(f"執行: {description}")
    logger.info(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            logger.info(f"✓ {description} 成功")
            if result.stdout:
                logger.info("輸出:")
                for line in result.stdout.strip().split('\n')[:10]:  # 只顯示前10行
                    if line.strip():
                        logger.info(f"  {line}")
                if len(result.stdout.strip().split('\n')) > 10:
                    logger.info("  ... (輸出已截斷)")
            return True
        else:
            logger.error(f"✗ {description} 失敗")
            if result.stderr:
                logger.error("錯誤信息:")
                for line in result.stderr.strip().split('\n'):
                    if line.strip():
                        logger.error(f"  {line}")
            return False
            
    except Exception as e:
        logger.error(f"✗ {description} 執行異常: {e}")
        return False

def convert_results(config, args):
    """執行結果轉換"""
    logger.info("=" * 60)
    logger.info("步驟 1: 結果轉換")
    logger.info("=" * 60)
    
    cmd = [
        "python", "convert_results_to_trades.py",
        "--results_dir", args.results_dir or config['defaults']['results_dir'],
        "--output_dir", args.output_dir or config['defaults']['output_dir'],
        "--top_k", str(args.top_k or config['defaults']['top_k']),
        "--ticker", args.ticker or config['defaults']['ticker']
    ]
    
    if args.dry_run:
        cmd.append("--dry_run")
        logger.info("DRY RUN 模式 - 只預覽不實際轉換")
    
    return run_command(cmd, "結果轉換")

def run_ensemble(config, args, method):
    """執行 Ensemble 策略"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"步驟 2: 執行 {method.title()} Ensemble")
    logger.info(f"{'=' * 60}")
    
    cmd = [
        "python", "run_enhanced_ensemble.py",
        "--method", method,
        "--trades_dir", args.trades_dir or config['defaults']['trades_dir']
    ]
    
    if args.scan_params:
        cmd.append("--scan_params")
        logger.info(f"啓用參數掃描優化")
    
    if args.discount:
        cmd.extend(["--discount", str(args.discount)])
        logger.info(f"使用折扣率: {args.discount}")
    
    if args.no_cost:
        cmd.append("--no_cost")
        logger.info("無交易成本模式")
    
    return run_command(cmd, f"{method.title()} Ensemble 執行")

def generate_summary_report(config, args, results):
    """生成彙總報告"""
    logger.info("\n" + "=" * 60)
    logger.info("工作流程執行彙總")
    logger.info("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 創建彙總報告
    report_file = Path(f"workflow_summary_{timestamp}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Ensemble 工作流程執行彙總報告\n")
        f.write("=" * 50 + "\n")
        f.write(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"參數設置:\n")
        f.write(f"  - 策略數量: {args.top_k or config['defaults']['top_k']}\n")
        f.write(f"  - 標的代碼: {args.ticker or config['defaults']['ticker']}\n")
        f.write(f"  - 參數掃描: {'是' if args.scan_params else '否'}\n")
        f.write(f"  - 交易成本: {'否' if args.no_cost else '是'}\n")
        if args.discount:
            f.write(f"  - 折扣率: {args.discount}\n")
        f.write(f"\n執行結果:\n")
        
        for step, success in results.items():
            status = "✓ 成功" if success else "✗ 失敗"
            f.write(f"  - {step}: {status}\n")
        
        f.write(f"\n輸出文件位置:\n")
        f.write(f"  - 交易文件: {args.output_dir or config['defaults']['output_dir']}/\n")
        f.write(f"  - Ensemble 結果: results/\n")
        f.write(f"  - 日誌文件: log/\n")
        
        f.write(f"\n下一步操作建議:\n")
        if all(results.values()):
            f.write("1. 檢查生成的交易文件質量\n")
            f.write("2. 分析 Ensemble 績效表現\n")
            f.write("3. 根據結果調整參數設置\n")
            f.write("4. 考慮進一步優化策略組合\n")
        else:
            f.write("1. 檢查失敗步驟的錯誤信息\n")
            f.write("2. 驗證輸入文件和參數設置\n")
            f.write("3. 重新運行失敗的步驟\n")
    
    logger.info(f"彙總報告已保存到: {report_file}")
    return report_file

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='一鍵式 Ensemble 工作流程執行')
    parser.add_argument('--method', choices=['majority', 'proportional', 'both'], 
                       default='both', help='Ensemble 方法')
    parser.add_argument('--top_k', type=int, help='每個策略取前K個最佳結果')
    parser.add_argument('--ticker', help='標的代碼')
    parser.add_argument('--results_dir', help='Optuna 結果目錄')
    parser.add_argument('--output_dir', help='輸出交易文件目錄')
    parser.add_argument('--trades_dir', help='交易文件目錄')
    parser.add_argument('--scan_params', action='store_true', help='啓用參數掃描')
    parser.add_argument('--discount', type=float, help='交易成本折扣率')
    parser.add_argument('--no_cost', action='store_true', help='無交易成本模式')
    parser.add_argument('--dry_run', action='store_true', help='只預覽不實際執行')
    parser.add_argument('--skip_convert', action='store_true', help='跳過轉換步驟')
    
    args = parser.parse_args()
    
    # 加載配置
    config = load_config()
    
    logger.info("🚀 開始執行 Ensemble 工作流程")
    logger.info(f"方法: {args.method}")
    logger.info(f"參數掃描: {'是' if args.scan_params else '否'}")
    logger.info(f"交易成本: {'否' if args.no_cost else '是'}")
    
    results = {}
    
    # 步驟 1: 結果轉換
    if not args.skip_convert:
        results['結果轉換'] = convert_results(config, args)
        if not results['結果轉換'] and not args.dry_run:
            logger.error("結果轉換失敗，終止工作流程")
            return 1
    else:
        logger.info("跳過結果轉換步驟")
        results['結果轉換'] = True
    
    # 步驟 2: Ensemble 執行
    if args.method in ['majority', 'both']:
        results['Majority Ensemble'] = run_ensemble(config, args, 'majority')
    
    if args.method in ['proportional', 'both']:
        results['Proportional Ensemble'] = run_ensemble(config, args, 'proportional')
    
    # 生成彙總報告
    report_file = generate_summary_report(config, args, results)
    
    # 輸出最終結果
    logger.info("\n" + "=" * 60)
    logger.info("工作流程執行完成")
    logger.info("=" * 60)
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    logger.info(f"執行結果: {success_count}/{total_count} 步驟成功")
    
    if success_count == total_count:
        logger.info("🎉 所有步驟執行成功！")
        logger.info(f"詳細報告: {report_file}")
        
        if not args.dry_run:
            logger.info("\n下一步建議:")
            logger.info("1. 檢查 sss_backtest_outputs/ 目錄下的交易文件")
            logger.info("2. 查看 results/ 目錄下的 Ensemble 結果")
            logger.info("3. 分析 log/ 目錄下的執行日誌")
            logger.info("4. 根據結果調整參數設置")
    else:
        logger.error("❌ 部分步驟執行失敗，請檢查上述錯誤信息")
        logger.error(f"詳細報告: {report_file}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
