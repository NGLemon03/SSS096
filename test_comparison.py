import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from SSSv095b2 import backtest_unified as sss_backtest, compute_ssma_turn_combined, load_data
from analysis.OSv3 import backtest_unified as os_backtest
from analysis.data_loader import load_data as os_load_data

def test_ssma_turn_comparison():
    """比較OSv3和SSSv095b2使用相同SSMA_turn參數的回測結果"""
    
    # SSMA_turn參數（從param_presets中選擇）
    ssma_turn_params = {
        "SSMA_turn 0": {
            "linlen": 25, "smaalen": 85, "factor": 80.0, "prom_factor": 9, 
            "min_dist": 8, "buy_shift": 0, "exit_shift": 6, "vol_window": 90, 
            "quantile_win": 65, "signal_cooldown_days": 7, "buy_mult": 0.15, 
            "sell_mult": 0.1, "stop_loss": 0.13, "smaa_source": "Factor (^TWII / 2414.TW)"
        },
        "SSMA_turn 1": {
            "linlen": 15, "smaalen": 40, "factor": 40.0, "prom_factor": 70, 
            "min_dist": 10, "buy_shift": 6, "exit_shift": 4, "vol_window": 40, 
            "quantile_win": 65, "signal_cooldown_days": 10, "buy_mult": 1.55, 
            "sell_mult": 2.1, "stop_loss": 0.15, "smaa_source": "Self"
        }
    }
    
    print("開始比較SSSv095b2的SSMA_turn回測結果（使用不同參數設定）...")
    print("=" * 80)
    
    for strategy_name, params in ssma_turn_params.items():
        print(f"\n測試策略: {strategy_name}")
        print("-" * 50)
        
        # 提取計算參數和回測參數
        calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 
                    'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
        calc_params = {k: v for k, v in params.items() if k in calc_keys}
        smaa_source = params['smaa_source']
        
        print(f"計算參數: {calc_params}")
        print(f"數據源: {smaa_source}")
        
        try:
            # 載入數據
            print("載入數據...")
            df_price, df_factor = load_data(ticker="00631L.TW", smaa_source=smaa_source)
            os_df_price, os_df_factor = os_load_data(ticker="00631L.TW", smaa_source=smaa_source)
            print(f"數據長度: {len(df_price)}")
            
            # 測試1: 使用SSSv095b2預設參數
            print("\n測試1: 使用SSSv095b2預設參數 (discount=0.30, trade_cooldown_bars=7)")
            df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(
                df_price, df_factor, **calc_params, smaa_source=smaa_source
            )
            os_df_ind, os_buy_dates, os_sell_dates = compute_ssma_turn_combined(
                os_df_price, os_df_factor, **calc_params, smaa_source=smaa_source
            )
            print(f"買入信號數: {len(buy_dates)}, 賣出信號數: {len(sell_dates)}")
            
            sss_result = sss_backtest(
                df_ind, "ssma_turn", params, buy_dates, sell_dates,
                discount=0.30, trade_cooldown_bars=7, bad_holding=False
            )
            os_result = os_backtest(
                os_df_ind, "ssma_turn", params, os_buy_dates, os_sell_dates,
                discount=0.30, trade_cooldown_bars=7, bad_holding=False
            )
            
            # 比較結果
            print("\n比較結果:")
            print(f"{'指標':<20} {'SSSv095b2預設':<15} {'OSv3':<15} {'差異':<15}")
            print("-" * 65)
            
            sss_metrics = sss_result['metrics']
            os_metrics = os_result['metrics']
            
            for key in ['total_return', 'num_trades', 'sharpe_ratio', 'max_drawdown']:
                val1 = sss_metrics.get(key, 0)
                val2 = os_metrics.get(key, 0)
                diff = val2 - val1
                
                if key == 'total_return':
                    print(f"{key:<20} {val1:>14.2%} {val2:>14.2%} {diff:>14.2%}")
                elif key == 'num_trades':
                    print(f"{key:<20} {val1:>14d} {val2:>14d} {diff:>14d}")
                else:
                    print(f"{key:<20} {val1:>14.4f} {val2:>14.4f} {diff:>14.4f}")
            
            # 檢查是否有顯著差異
            total_return_diff = abs(os_metrics.get('total_return', 0) - sss_metrics.get('total_return', 0))
            if total_return_diff > 0.01:  # 1%的差異
                print(f"\n⚠️  警告: 總報酬率差異過大 ({total_return_diff:.2%})")
            else:
                print(f"\n✅ 總報酬率差異在可接受範圍內 ({total_return_diff:.2%})")
                
        except Exception as e:
            print(f"❌ 測試失敗: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    test_ssma_turn_comparison() 