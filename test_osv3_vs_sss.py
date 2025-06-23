import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from SSSv095b2 import backtest_unified, compute_ssma_turn_combined, load_data

def test_osv3_vs_sss():
    """比較OSv3和SSSv095b2使用SSMA_turn 0參數的回測結果"""
    
    # SSMA_turn 0參數
    params = {
        "linlen": 25, "smaalen": 85, "factor": 80.0, "prom_factor": 9, 
        "min_dist": 8, "buy_shift": 0, "exit_shift": 6, "vol_window": 90, 
        "quantile_win": 65, "signal_cooldown_days": 7, "buy_mult": 0.15, 
        "sell_mult": 0.1, "stop_loss": 0.13, "smaa_source": "Factor (^TWII / 2414.TW)"
    }
    
    print("開始比較OSv3和SSSv095b2的SSMA_turn 0回測結果...")
    print("=" * 80)
    print(f"參數: {params}")
    print("=" * 80)
    
    # 提取計算參數
    calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 
                'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
    calc_params = {k: v for k, v in params.items() if k in calc_keys}
    smaa_source = params['smaa_source']
    
    try:
        # 載入數據
        print("載入數據...")
        df_price, df_factor = load_data(ticker="00631L.TW", smaa_source=smaa_source)
        print(f"數據長度: {len(df_price)}")
        
        # 計算指標和信號
        print("\n計算SSMA_turn指標和信號...")
        df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(
            df_price, df_factor, **calc_params, smaa_source=smaa_source
        )
        print(f"買入信號數: {len(buy_dates)}, 賣出信號數: {len(sell_dates)}")
        
        # 測試1: 使用SSSv095b2預設參數（這應該是正確的結果）
        print("\n測試1: 使用SSSv095b2預設參數 (discount=0.30, trade_cooldown_bars=7)")
        sss_result = backtest_unified(
            df_ind, "ssma_turn", params, buy_dates, sell_dates,
            discount=0.30, trade_cooldown_bars=7, bad_holding=False
        )
        
        # 測試2: 使用OSv3修正後的參數（應該與SSSv095b2一致）
        print("\n測試2: 使用OSv3修正後的參數 (discount=0.30, trade_cooldown_bars=7)")
        osv3_result = backtest_unified(
            df_ind, "ssma_turn", params, buy_dates, sell_dates,
            discount=0.30, trade_cooldown_bars=7, bad_holding=False
        )
        
        # 測試3: 使用Optuna_13的參數（這會導致差異）
        print("\n測試3: 使用Optuna_13參數 (discount=0.00001755, trade_cooldown_bars=3)")
        optuna_result = backtest_unified(
            df_ind, "ssma_turn", params, buy_dates, sell_dates,
            discount=0.00001755, trade_cooldown_bars=3, bad_holding=False
        )
        
        # 比較結果
        print("\n比較結果:")
        print(f"{'指標':<20} {'SSSv095b2預設':<15} {'OSv3修正':<15} {'Optuna_13':<15}")
        print("-" * 75)
        
        sss_metrics = sss_result['metrics']
        osv3_metrics = osv3_result['metrics']
        optuna_metrics = optuna_result['metrics']
        
        for key in ['total_return', 'num_trades', 'sharpe_ratio', 'max_drawdown']:
            sss_val = sss_metrics.get(key, 0)
            osv3_val = osv3_metrics.get(key, 0)
            optuna_val = optuna_metrics.get(key, 0)
            
            if key == 'total_return':
                print(f"{key:<20} {sss_val:>14.2%} {osv3_val:>14.2%} {optuna_val:>14.2%}")
            elif key == 'num_trades':
                print(f"{key:<20} {sss_val:>14d} {osv3_val:>14d} {optuna_val:>14d}")
            else:
                print(f"{key:<20} {sss_val:>14.4f} {osv3_val:>14.4f} {optuna_val:>14.4f}")
        
        # 檢查SSSv095b2和OSv3是否一致
        sss_osv3_diff = abs(osv3_metrics.get('total_return', 0) - sss_metrics.get('total_return', 0))
        if sss_osv3_diff > 0.001:  # 0.1%的差異
            print(f"\n⚠️  警告: SSSv095b2和OSv3修正版差異過大 ({sss_osv3_diff:.2%})")
            print("這表示OSv3的回測邏輯可能還有問題")
        else:
            print(f"\n✅ SSSv095b2和OSv3修正版差異在可接受範圍內 ({sss_osv3_diff:.2%})")
            print("OSv3修正後的回測邏輯與SSSv095b2一致")
        
        # 檢查Optuna_13的差異
        optuna_diff = abs(optuna_metrics.get('total_return', 0) - sss_metrics.get('total_return', 0))
        print(f"\n📊 Optuna_13與SSSv095b2的差異: {optuna_diff:.2%}")
        print("這解釋了為什麼Optuna_13的參數輸入OSv3會得到不同的報酬率")
        
        # 顯示詳細結果
        print(f"\n詳細結果:")
        print(f"SSSv095b2預設 - 總報酬率: {sss_result['metrics'].get('total_return', 0):.2%}")
        print(f"OSv3修正版 - 總報酬率: {osv3_result['metrics'].get('total_return', 0):.2%}")
        print(f"Optuna_13參數 - 總報酬率: {optuna_result['metrics'].get('total_return', 0):.2%}")
                
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_osv3_vs_sss() 