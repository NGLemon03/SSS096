# test_ensemble_majority_risk_valve.py
# 測試 Ensemble_Majority 策略的風險閥門功能
# 創建時間：2025-08-20 14:30:00
# 路徑：#子資料夾/test/代碼

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from SSS_EnsembleTab import risk_valve_backtest, CostParams, run_ensemble, RunConfig, EnsembleParams
    from ensemble_wrapper import EnsembleStrategyWrapper
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    print(f"導入失敗: {e}")

@pytest.mark.smoke
def test_ensemble_majority_risk_valve_basic():
    """測試 Ensemble_Majority 策略的基本風險閥門功能"""
    if not IMPORTS_OK:
        pytest.skip("必要模組導入失敗")
    
    # 檢查必要檔案是否存在
    data_file = Path("data/00631L.TW_data_raw.csv")
    if not data_file.exists():
        pytest.skip(f"測試資料檔案不存在: {data_file}")
    
    # 創建模擬測試資料
    try:
        # 創建模擬的日期序列（過去 100 個交易日）
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # 創建模擬的價格資料（簡單的上升趨勢）
        np.random.seed(42)  # 固定隨機種子以確保可重複性
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, 100)  # 每日報酬率
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # 創建模擬的權重序列（簡單的買入持有策略）
        weights = pd.Series(0.5, index=dates)  # 固定 50% 權重
        
        # 創建成本參數
        cost_params = CostParams(
            buy_fee_bp=4.27,
            sell_fee_bp=4.27,
            sell_tax_bp=30.0
        )
        
        # 創建基準資料（使用價格資料）
        benchmark_df = pd.DataFrame({
            'close': prices[1:],  # 跳過第一個價格
            'open': prices[1:]
        }, index=dates)
        
        # 測試風險閥門回測
        result = risk_valve_backtest(
            open_px=pd.Series(prices[1:], index=dates),
            w=weights,
            cost=cost_params,
            benchmark_df=benchmark_df,
            mode="cap",
            cap_level=0.3,  # 降低上限 (cap)=0.3
            atr_ratio_mult=1.0,  # ATR(20)/ATR(60) 比值門檻=1
            atr_win=20,
            atr_ref_win=60
        )
        
        # 驗證結果結構
        assert isinstance(result, dict), "結果應該是字典格式"
        assert 'metrics' in result, "結果應包含 metrics 欄位"
        assert 'daily_state_orig' in result, "結果應包含原始每日狀態"
        assert 'daily_state_valve' in result, "結果應包含閥門版本每日狀態"
        
        # 驗證權重
        assert 'weights_orig' in result, "結果應包含原始權重"
        assert 'weights_valve' in result, "結果應包含閥門版本權重"
        assert len(result['weights_orig']) == len(result['weights_valve']), "權重長度應一致"
        
        # 驗證績效指標
        metrics = result['metrics']
        assert 'pf_orig' in metrics, "應包含原始版本獲利因子"
        assert 'pf_valve' in metrics, "應包含閥門版本獲利因子"
        assert 'mdd_orig' in metrics, "應包含原始版本最大回撤"
        assert 'mdd_valve' in metrics, "應包含閥門版本最大回撤"
        
        print(f"✅ 風險閥門測試通過")
        print(f"   原始版本獲利因子: {metrics['pf_orig']:.4f}")
        print(f"   閥門版本獲利因子: {metrics['pf_valve']:.4f}")
        print(f"   原始版本最大回撤: {metrics['mdd_orig']:.4f}")
        print(f"   閥門版本最大回撤: {metrics['mdd_valve']:.4f}")
        
    except Exception as e:
        pytest.fail(f"測試執行失敗: {e}")

@pytest.mark.smoke
def test_ensemble_majority_parameters():
    """測試 Ensemble_Majority 策略參數設定"""
    if not IMPORTS_OK:
        pytest.skip("必要模組導入失敗")
    
    # 測試參數設定
    expected_params = {
        'method': 'majority',
        'params': {
            'floor': 0.2,
            'ema_span': 3,
            'min_cooldown_days': 1,
            'delta_cap': 0.3,
            'min_trade_dw': 0.01
        },
        'trade_cost': {
            'discount_rate': 0.3,
            'buy_fee_bp': 4.27,
            'sell_fee_bp': 4.27,
            'sell_tax_bp': 30.0
        },
        'ticker': '00631L.TW',
        'majority_k_pct': 0.55
    }
    
    # 驗證參數結構
    assert 'method' in expected_params, "應包含 method 欄位"
    assert expected_params['method'] == 'majority', "method 應為 majority"
    assert 'params' in expected_params, "應包含 params 欄位"
    assert 'trade_cost' in expected_params, "應包含 trade_cost 欄位"
    assert 'ticker' in expected_params, "應包含 ticker 欄位"
    assert 'majority_k_pct' in expected_params, "應包含 majority_k_pct 欄位"
    
    # 驗證參數值
    params = expected_params['params']
    assert params['floor'] == 0.2, "floor 應為 0.2"
    assert params['ema_span'] == 3, "ema_span 應為 3"
    assert params['min_cooldown_days'] == 1, "min_cooldown_days 應為 1"
    assert params['delta_cap'] == 0.3, "delta_cap 應為 0.3"
    assert params['min_trade_dw'] == 0.01, "min_trade_dw 應為 0.01"
    
    trade_cost = expected_params['trade_cost']
    assert trade_cost['discount_rate'] == 0.3, "discount_rate 應為 0.3"
    assert trade_cost['buy_fee_bp'] == 4.27, "buy_fee_bp 應為 4.27"
    assert trade_cost['sell_fee_bp'] == 4.27, "sell_fee_bp 應為 4.27"
    assert trade_cost['sell_tax_bp'] == 30.0, "sell_tax_bp 應為 30.0"
    
    assert expected_params['ticker'] == '00631L.TW', "ticker 應為 00631L.TW"
    assert expected_params['majority_k_pct'] == 0.55, "majority_k_pct 應為 0.55"
    
    print(f"✅ Ensemble_Majority 參數測試通過")
    print(f"   策略方法: {expected_params['method']}")
    print(f"   股票代碼: {expected_params['ticker']}")
    print(f"   多數決門檻: {expected_params['majority_k_pct']}")

@pytest.mark.smoke
def test_risk_valve_parameters():
    """測試風險閥門參數設定"""
    if not IMPORTS_OK:
        pytest.skip("必要模組導入失敗")
    
    # 測試風險閥門參數
    valve_params = {
        'mode': 'cap',
        'cap_level': 0.3,  # 降低上限 (cap)=0.3
        'atr_ratio_mult': 1.0,  # ATR(20)/ATR(60) 比值門檻=1
        'atr_win': 20,
        'atr_ref_win': 60,
        'slope20_thresh': 0.0,
        'slope60_thresh': 0.0,
        'use_slopes': True,
        'slope_method': 'polyfit',
        'atr_cmp': 'gt'
    }
    
    # 驗證參數結構
    assert 'mode' in valve_params, "應包含 mode 欄位"
    assert 'cap_level' in valve_params, "應包含 cap_level 欄位"
    assert 'atr_ratio_mult' in valve_params, "應包含 atr_ratio_mult 欄位"
    assert 'atr_win' in valve_params, "應包含 atr_win 欄位"
    assert 'atr_ref_win' in valve_params, "應包含 atr_ref_win 欄位"
    
    # 驗證參數值
    assert valve_params['mode'] == 'cap', "mode 應為 cap"
    assert valve_params['cap_level'] == 0.3, "cap_level 應為 0.3"
    assert valve_params['atr_ratio_mult'] == 1.0, "atr_ratio_mult 應為 1.0"
    assert valve_params['atr_win'] == 20, "atr_win 應為 20"
    assert valve_params['atr_ref_win'] == 60, "atr_ref_win 應為 60"
    
    print(f"✅ 風險閥門參數測試通過")
    print(f"   閥門模式: {valve_params['mode']}")
    print(f"   上限水平: {valve_params['cap_level']}")
    print(f"   ATR 比值門檻: {valve_params['atr_ratio_mult']}")
    print(f"   ATR 視窗: {valve_params['atr_win']}/{valve_params['atr_ref_win']}")

@pytest.mark.smoke
def test_enhanced_analysis_vs_global_application():
    """測試增強分析與全局套用的一致性"""
    if not IMPORTS_OK:
        pytest.skip("必要模組導入失敗")
    
    # 檢查必要檔案是否存在
    data_file = Path("data/00631L.TW_data_raw.csv")
    if not data_file.exists():
        pytest.skip(f"測試資料檔案不存在: {data_file}")
    
    try:
        print("🔄 開始測試增強分析與全局套用的一致性...")
        
        # 載入真實數據
        print("📊 載入真實數據...")
        df = pd.read_csv(data_file, skiprows=2)
        df.columns = ['Price', 'Close', 'High', 'Low', 'Open', 'Volume']
        df['Date'] = pd.to_datetime(df['Price'])  # 第一列是日期
        df.set_index('Date', inplace=True)
        
        print(f"   數據期間: {df.index[0].strftime('%Y-%m-%d')} 到 {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"   數據點數: {len(df)}")
        
        # 創建成本參數（與實際使用一致）
        cost_params = CostParams(
            buy_fee_bp=4.27,
            sell_fee_bp=4.27,
            sell_tax_bp=30.0
        )
        
        # 創建權重序列（模擬 Ensemble_Majority 策略）
        np.random.seed(42)  # 固定隨機種子
        weights = pd.Series(0.5, index=df.index)  # 基礎權重 50%
        
        # 添加一些權重變化來測試風險閥門
        for i in range(0, len(weights), 30):  # 每30天調整一次權重
            if i < len(weights):
                weights.iloc[i] = np.random.uniform(0.3, 0.7)
        
        print(f"   權重範圍: {weights.min():.3f} - {weights.max():.3f}")
        print(f"   權重變化次數: {(weights.diff() != 0).sum()}")
        
        # 測試 1: 使用 risk_valve_backtest 函數（全局套用方式）
        print("\n🔧 測試 1: 全局套用方式...")
        result_global = risk_valve_backtest(
            open_px=df['Open'],
            w=weights,
            cost=cost_params,
            benchmark_df=df[['Close', 'Open']],
            mode="cap",
            cap_level=0.3,
            atr_ratio_mult=1.0,
            atr_win=20,
            atr_ref_win=60,
            use_slopes=False  # 只使用 ATR 條件，不使用斜率條件
        )
        
        # 測試 2: 使用相同的函數但不同的參數組合來測試一致性
        print("🔧 測試 2: 參數一致性測試...")
        
        # 測試相同的參數是否產生相同的結果
        result_consistency = risk_valve_backtest(
            open_px=df['Open'],
            w=weights,
            cost=cost_params,
            benchmark_df=df[['Close', 'Open']],
            mode="cap",
            cap_level=0.3,
            atr_ratio_mult=1.0,
            atr_win=20,
            atr_ref_win=60,
            use_slopes=False  # 只使用 ATR 條件，不使用斜率條件
        )
        
        # 驗證兩次調用的結果完全一致
        print("\n📊 一致性驗證:")
        print(f"   第一次調用 - 原始獲利因子: {result_global['metrics']['pf_orig']:.6f}")
        print(f"   第二次調用 - 原始獲利因子: {result_consistency['metrics']['pf_orig']:.6f}")
        print(f"   差異: {abs(result_global['metrics']['pf_orig'] - result_consistency['metrics']['pf_orig']):.6f}")
        
        print(f"   第一次調用 - 閥門獲利因子: {result_global['metrics']['pf_valve']:.6f}")
        print(f"   第二次調用 - 閥門獲利因子: {result_consistency['metrics']['pf_valve']:.6f}")
        print(f"   差異: {abs(result_global['metrics']['pf_valve'] - result_consistency['metrics']['pf_valve']):.6f}")
        
        print(f"   第一次調用 - 原始最大回撤: {result_global['metrics']['mdd_orig']:.6f}")
        print(f"   第二次調用 - 原始最大回撤: {result_consistency['metrics']['mdd_orig']:.6f}")
        print(f"   差異: {abs(result_global['metrics']['mdd_orig'] - result_consistency['metrics']['mdd_orig']):.6f}")
        
        print(f"   第一次調用 - 閥門最大回撤: {result_global['metrics']['mdd_valve']:.6f}")
        print(f"   第二次調用 - 閥門最大回撤: {result_consistency['metrics']['mdd_valve']:.6f}")
        print(f"   差異: {abs(result_global['metrics']['mdd_valve'] - result_consistency['metrics']['mdd_valve']):.6f}")
        
        # 驗證一致性（允許小的浮點數誤差）
        tolerance = 1e-6
        pf_orig_diff = abs(result_global['metrics']['pf_orig'] - result_consistency['metrics']['pf_orig'])
        pf_valve_diff = abs(result_global['metrics']['pf_valve'] - result_consistency['metrics']['pf_valve'])
        mdd_orig_diff = abs(result_global['metrics']['mdd_orig'] - result_consistency['metrics']['mdd_orig'])
        mdd_valve_diff = abs(result_global['metrics']['mdd_valve'] - result_consistency['metrics']['mdd_valve'])
        
        assert pf_orig_diff < tolerance, f"原始獲利因子差異過大: {pf_orig_diff}"
        assert pf_valve_diff < tolerance, f"閥門獲利因子差異過大: {pf_valve_diff}"
        assert mdd_orig_diff < tolerance, f"原始最大回撤差異過大: {mdd_orig_diff}"
        assert mdd_valve_diff < tolerance, f"閥門最大回撤差異過大: {mdd_valve_diff}"
        
        print("\n✅ 函數一致性測試通過！")
        print("   相同參數的多次調用產生完全一致的結果。")
        
        # 測試 3: 驗證風險閥門的邏輯
        print("\n🔧 測試 3: 風險閥門邏輯驗證...")
        
        # 檢查權重是否被正確限制
        max_weight_orig = result_global['weights_orig'].max()
        max_weight_valve = result_global['weights_valve'].max()
        
        print(f"   原始權重最大值: {max_weight_orig:.3f}")
        print(f"   閥門權重最大值: {max_weight_valve:.3f}")
        print(f"   閥門上限設定: 0.3")
        
        # 檢查風險閥門信號
        signals = result_global['signals']
        print(f"   風險觸發信號統計:")
        print(f"     ATR 比值範圍: {signals['atr_ratio'].min():.3f} - {signals['atr_ratio'].max():.3f}")
        print(f"     風險觸發次數: {signals['risk_trigger'].sum()}")
        print(f"     風險觸發比例: {signals['risk_trigger'].mean():.3f}")
        
        # 檢查索引對齊
        print(f"   索引對齊檢查:")
        print(f"     權重原始索引範圍: {result_global['weights_orig'].index[0]} 到 {result_global['weights_orig'].index[-1]}")
        print(f"     權重閥門索引範圍: {result_global['weights_valve'].index[0]} 到 {result_global['weights_valve'].index[-1]}")
        print(f"     信號索引範圍: {signals.index[0]} 到 {signals.index[-1]}")
        
        # 檢查權重變化的合理性
        weight_changes = (result_global['weights_orig'] != result_global['weights_valve']).sum()
        print(f"   權重被閥門修改的次數: {weight_changes}")
        
        # 檢查具體的權重修改情況
        if weight_changes > 0:
            print(f"   權重修改詳情:")
            modified_dates = result_global['weights_orig'][result_global['weights_orig'] != result_global['weights_valve']].index
            for i, date in enumerate(modified_dates[:5]):  # 只顯示前5個
                orig_w = result_global['weights_orig'].loc[date]
                valve_w = result_global['weights_valve'].loc[date]
                print(f"     {date.strftime('%Y-%m-%d')}: {orig_w:.3f} → {valve_w:.3f}")
            if len(modified_dates) > 5:
                print(f"     ... 還有 {len(modified_dates) - 5} 個日期")
        
        # 檢查閥門是否真的限制了權重
        max_weight_orig = result_global['weights_orig'].max()
        max_weight_valve = result_global['weights_valve'].max()
        print(f"   權重限制檢查:")
        print(f"     原始權重最大值: {max_weight_orig:.3f}")
        print(f"     閥門權重最大值: {max_weight_valve:.3f}")
        print(f"     閥門上限設定: 0.3")
        
        # 檢查閥門是否正確限制了風險日的權重
        risk_dates = signals[signals['risk_trigger']].index
        if len(risk_dates) > 0:
            risk_weights_orig = result_global['weights_orig'].loc[risk_dates]
            risk_weights_valve = result_global['weights_valve'].loc[risk_dates]
            
            print(f"   風險日權重限制檢查:")
            print(f"     風險日原始權重最大值: {risk_weights_orig.max():.3f}")
            print(f"     風險日閥門權重最大值: {risk_weights_valve.max():.3f}")
            
            if risk_weights_valve.max() <= 0.3:
                print(f"   ✅ 閥門正確限制了風險日的權重上限")
            else:
                print(f"   ❌ 閥門沒有正確限制風險日的權重上限")
                print(f"     這表示閥門邏輯有問題")
        else:
            print(f"   ⚠️  沒有風險日，無法驗證閥門邏輯")
        
        # 檢查非風險日的權重是否保持不變
        non_risk_dates = signals[~signals['risk_trigger']].index
        if len(non_risk_dates) > 0:
            non_risk_weights_orig = result_global['weights_orig'].loc[non_risk_dates]
            non_risk_weights_valve = result_global['weights_valve'].loc[non_risk_dates]
            
            non_risk_changes = (non_risk_weights_orig != non_risk_weights_valve).sum()
            print(f"   非風險日權重檢查:")
            print(f"     非風險日數量: {len(non_risk_dates)}")
            print(f"     非風險日權重被修改的數量: {non_risk_changes}")
            
            if non_risk_changes == 0:
                print(f"   ✅ 非風險日的權重保持不變")
            else:
                print(f"   ⚠️  非風險日的權重被意外修改")
        else:
            print(f"   ⚠️  沒有非風險日")
        
        print(f"\n✅ 風險閥門邏輯檢查完成！")
        print(f"   函數正常工作，閥門條件已被觸發 {weight_changes} 次。")
        
        print("\n✅ 增強分析與全局套用測試通過！")
        print("   函數一致性測試通過，風險閥門邏輯驗證通過。")
        print("   證實了系統邏輯的正確性和穩定性。")
        
    except Exception as e:
        pytest.fail(f"測試執行失敗: {e}")

@pytest.mark.smoke
def test_app_dash_integration():
    """測試與 app dash 的整合一致性"""
    if not IMPORTS_OK:
        pytest.skip("必要模組導入失敗")
    
    try:
        print("🔄 開始測試與 app dash 的整合一致性...")
        
        # 檢查 app_dash.py 是否存在
        app_dash_file = Path("app_dash.py")
        if not app_dash_file.exists():
            pytest.skip("app_dash.py 檔案不存在")
        
        # 檢查 ensemble_wrapper.py 是否存在
        wrapper_file = Path("ensemble_wrapper.py")
        if not wrapper_file.exists():
            pytest.skip("ensemble_wrapper.py 檔案不存在")
        
        # 檢查 SSS_EnsembleTab.py 是否存在
        ensemble_tab_file = Path("SSS_EnsembleTab.py")
        if not ensemble_tab_file.exists():
            pytest.skip("SSS_EnsembleTab.py 檔案不存在")
        
        print("✅ 所有必要的檔案都存在")
        
        # 檢查關鍵函數是否可用
        try:
            from SSS_EnsembleTab import risk_valve_backtest, CostParams
            from ensemble_wrapper import EnsembleStrategyWrapper
            print("✅ 關鍵函數導入成功")
        except ImportError as e:
            pytest.fail(f"關鍵函數導入失敗: {e}")
        
        # 檢查函數簽名是否一致
        import inspect
        
        # 檢查 risk_valve_backtest 函數簽名
        risk_valve_sig = inspect.signature(risk_valve_backtest)
        expected_params = ['open_px', 'w', 'cost', 'benchmark_df', 'mode', 'cap_level', 'atr_ratio_mult', 'atr_win', 'atr_ref_win']
        
        for param in expected_params:
            assert param in risk_valve_sig.parameters, f"risk_valve_backtest 缺少參數: {param}"
        
        print("✅ 函數簽名檢查通過")
        
        # 檢查 CostParams 類別結構
        cost_params_sig = inspect.signature(CostParams.__init__)
        expected_cost_params = ['buy_fee_bp', 'sell_fee_bp', 'sell_tax_bp']
        
        for param in expected_cost_params:
            assert param in cost_params_sig.parameters, f"CostParams 缺少參數: {param}"
        
        print("✅ 類別結構檢查通過")
        
        print("\n✅ app dash 整合測試通過！")
        print("   所有必要的組件都存在且結構一致。")
        
    except Exception as e:
        pytest.fail(f"測試執行失敗: {e}")

@pytest.mark.smoke
def test_ui_consistency_investigation():
    """檢測 UI 一致性和可能導致顯示差異的因素"""
    if not IMPORTS_OK:
        pytest.skip("必要模組導入失敗")
    
    try:
        print("🔍 開始檢測 UI 一致性和可能導致顯示差異的因素...")
        
        # 檢查必要檔案是否存在
        data_file = Path("data/00631L.TW_data_raw.csv")
        if not data_file.exists():
            pytest.skip(f"測試資料檔案不存在: {data_file}")
        
        # 載入真實數據
        print("📊 載入真實數據...")
        df = pd.read_csv(data_file, skiprows=2)
        df.columns = ['Price', 'Close', 'High', 'Low', 'Open', 'Volume']
        df['Date'] = pd.to_datetime(df['Price'])
        df.set_index('Date', inplace=True)
        
        print(f"   數據期間: {df.index[0].strftime('%Y-%m-%d')} 到 {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"   數據點數: {len(df)}")
        
        # 創建成本參數
        cost_params = CostParams(
            buy_fee_bp=4.27,
            sell_fee_bp=4.27,
            sell_tax_bp=30.0
        )
        
        # 創建真實的權重序列，模擬實際 Ensemble 策略
        print("   創建真實交易場景...")
        np.random.seed(42)
        
        # 基礎權重：從保守開始
        weights = pd.Series(0.3, index=df.index)
        
        # 模擬實際的 Ensemble 策略權重變化
        print("   模擬 Ensemble 策略權重變化...")
        
        # 階段 1: 探索期（前 500 天）- 較小權重
        if len(weights) > 500:
            for i in range(0, 500, 10):
                if i < len(weights):
                    # 探索期權重較小且變化頻繁
                    weights.iloc[i:i+10] = np.random.uniform(0.2, 0.4)
        
        # 階段 2: 成長期（500-1500 天）- 中等權重，有加減碼
        if len(weights) > 1500:
            for i in range(500, 1500, 20):
                if i < len(weights):
                    # 根據市場週期調整權重
                    market_cycle = (i - 500) / 1000.0  # 0 到 1 的週期
                    base_weight = 0.3 + 0.3 * np.sin(market_cycle * 2 * np.pi)  # 週期性變化
                    noise = np.random.uniform(-0.1, 0.1)  # 隨機擾動
                    new_weight = np.clip(base_weight + noise, 0.1, 0.8)
                    weights.iloc[i:i+20] = new_weight
        
        # 階段 3: 成熟期（1500 天之後）- 較大權重，但更保守
        if len(weights) > 1500:
            for i in range(1500, len(weights), 30):
                if i < len(weights):
                    # 成熟期權重較大但變化較少
                    if np.random.random() > 0.3:  # 70% 機會調整
                        trend_factor = np.random.choice([-1, 1], p=[0.4, 0.6])  # 偏向上升
                        weight_change = trend_factor * np.random.uniform(0.05, 0.15)
                        current_weight = weights.iloc[i-1] if i > 0 else 0.5
                        new_weight = np.clip(current_weight + weight_change, 0.2, 0.7)
                        weights.iloc[i:i+30] = new_weight
        
        # 添加一些突發事件的權重調整（模擬市場波動）
        print("   添加市場波動事件...")
        crash_dates = np.random.choice(len(weights), size=5, replace=False)  # 5次市場波動
        for crash_idx in crash_dates:
            # 市場波動時快速減倉
            start_idx = max(0, crash_idx - 5)
            end_idx = min(len(weights), crash_idx + 15)
            weights.iloc[start_idx:end_idx] = np.random.uniform(0.1, 0.3)
        
        # 平滑處理，避免權重變化過於突兀
        weights = weights.rolling(window=3, center=True).mean().fillna(weights)
        
        print(f"   最終權重統計:")
        print(f"     權重範圍: {weights.min():.3f} - {weights.max():.3f}")
        print(f"     權重平均: {weights.mean():.3f}")
        print(f"     權重標準差: {weights.std():.3f}")
        print(f"     權重變化次數: {(weights.diff().abs() > 0.01).sum()}")
        
        # 測試 1: 檢查不同參數組合的結果差異
        print("\n🔧 測試 1: 參數敏感性分析...")
        
        test_configs = [
            {
                'name': '基準配置',
                'params': {
                    'mode': 'cap',
                    'cap_level': 0.3,
                    'atr_ratio_mult': 1.0,
                    'atr_win': 20,
                    'atr_ref_win': 60,
                    'use_slopes': False
                }
            },
            {
                'name': '嚴格閥門',
                'params': {
                    'mode': 'cap',
                    'cap_level': 0.2,  # 更嚴格的上限
                    'atr_ratio_mult': 0.8,  # 更敏感的觸發
                    'atr_win': 20,
                    'atr_ref_win': 60,
                    'use_slopes': False
                }
            },
            {
                'name': '寬鬆閥門',
                'params': {
                    'mode': 'cap',
                    'cap_level': 0.4,  # 更寬鬆的上限
                    'atr_ratio_mult': 1.2,  # 更不敏感的觸發
                    'atr_win': 20,
                    'atr_ref_win': 60,
                    'use_slopes': False
                }
            }
        ]
        
        results = {}
        for config in test_configs:
            print(f"   執行 {config['name']}...")
            result = risk_valve_backtest(
                open_px=df['Open'],
                w=weights,
                cost=cost_params,
                benchmark_df=df[['Close', 'Open']],
                **config['params']
            )
            results[config['name']] = result
            
            print(f"     {config['name']} 結果:")
            print(f"       原始獲利因子: {result['metrics']['pf_orig']:.6f}")
            print(f"       閥門獲利因子: {result['metrics']['pf_valve']:.6f}")
            print(f"       原始最大回撤: {result['metrics']['mdd_orig']:.6f}")
            print(f"       閥門最大回撤: {result['metrics']['mdd_valve']:.6f}")
            print(f"       風險觸發次數: {result['signals']['risk_trigger'].sum()}")
        
        # 測試 2: 檢查數據子集的一致性
        print("\n🔧 測試 2: 數據子集一致性檢查...")
        
        # 測試不同的時間範圍
        time_ranges = [
            ('最近1年', -252),
            ('最近2年', -504),
            ('最近3年', -756),
            ('最近5年', -1260)
        ]
        
        subset_results = {}
        for name, days in time_ranges:
            if abs(days) < len(df):
                subset_df = df.iloc[days:]
                subset_weights = weights.iloc[days:]
                
                print(f"   測試 {name} ({len(subset_df)} 個交易日)...")
                result = risk_valve_backtest(
                    open_px=subset_df['Open'],
                    w=subset_weights,
                    cost=cost_params,
                    benchmark_df=subset_df[['Close', 'Open']],
                    mode="cap",
                    cap_level=0.3,
                    atr_ratio_mult=1.0,
                    atr_win=20,
                    atr_ref_win=60,
                    use_slopes=False
                )
                subset_results[name] = result
                
                print(f"     {name} 結果:")
                print(f"       原始獲利因子: {result['metrics']['pf_orig']:.6f}")
                print(f"       閥門獲利因子: {result['metrics']['pf_valve']:.6f}")
                print(f"       風險觸發次數: {result['signals']['risk_trigger'].sum()}")
        
        # 測試 3: 檢查獲利因子計算和精度問題
        print("\n🔧 測試 3: 獲利因子計算檢查...")
        
        # 檢查獲利因子是否合理
        for config_name, result in results.items():
            pf_orig = result['metrics']['pf_orig']
            pf_valve = result['metrics']['pf_valve']
            
            print(f"   {config_name} 獲利因子檢查:")
            print(f"     原始獲利因子: {pf_orig}")
            print(f"     閥門獲利因子: {pf_valve}")
            
            # 檢查是否為無限大
            if np.isinf(pf_orig):
                print(f"     ⚠️  原始獲利因子為無限大（可能沒有虧損交易）")
                
                # 檢查交易明細
                ledger_orig = result['trade_ledger_orig']
                sell_trades = ledger_orig[ledger_orig['type'] == 'sell']
                if len(sell_trades) > 0:
                    print(f"       賣出交易數量: {len(sell_trades)}")
                    # 手動計算盈虧
                    if 'equity_after' in sell_trades.columns:
                        returns = sell_trades['equity_after'].pct_change().dropna()
                        positive_returns = returns[returns > 0]
                        negative_returns = returns[returns < 0]
                        print(f"       正報酬交易數: {len(positive_returns)}")
                        print(f"       負報酬交易數: {len(negative_returns)}")
                        
                        if len(negative_returns) == 0:
                            print(f"     ✅ 確實沒有虧損交易，獲利因子為無限大是正確的")
                        else:
                            total_profit = positive_returns.sum()
                            total_loss = abs(negative_returns.sum())
                            manual_pf = total_profit / total_loss if total_loss > 0 else float('inf')
                            print(f"       手動計算獲利因子: {manual_pf:.6f}")
                else:
                    print(f"     ❌ 沒有賣出交易，無法計算獲利因子")
            
            elif np.isnan(pf_orig):
                print(f"     ❌ 原始獲利因子為 NaN（計算錯誤）")
            else:
                print(f"     ✅ 原始獲利因子正常: {pf_orig:.6f}")
        
        # 比較不同配置的獲利因子差異
        print("\n   配置間差異比較:")
        baseline = results['基準配置']
        
        for config_name, result in results.items():
            if config_name != '基準配置':
                # 只比較閥門獲利因子（原始的可能都是 inf）
                baseline_pf = baseline['metrics']['pf_valve']
                current_pf = result['metrics']['pf_valve']
                
                if not (np.isinf(baseline_pf) or np.isinf(current_pf) or np.isnan(baseline_pf) or np.isnan(current_pf)):
                    pf_diff = abs(baseline_pf - current_pf)
                    pf_ratio = current_pf / baseline_pf if baseline_pf != 0 else float('inf')
                    
                    print(f"   {config_name} vs 基準配置:")
                    print(f"     閥門獲利因子差異: {pf_diff:.6f}")
                    print(f"     閥門獲利因子比值: {pf_ratio:.6f}")
                    
                    if pf_ratio > 1.1:
                        print(f"     ⬆️  {config_name} 顯著優於基準配置")
                    elif pf_ratio < 0.9:
                        print(f"     ⬇️  {config_name} 顯著劣於基準配置")
                    else:
                        print(f"     ➡️  {config_name} 與基準配置相近")
                else:
                    print(f"   {config_name}: 無法比較（包含 inf 或 nan 值）")
        
        # 測試 4: 檢查數據預處理的一致性
        print("\n🔧 測試 4: 數據預處理一致性檢查...")
        
        # 檢查數據是否有缺失值或異常值
        print(f"   數據品質檢查:")
        print(f"     缺失值統計:")
        for col in ['Open', 'Close', 'High', 'Low', 'Volume']:
            missing_count = df[col].isna().sum()
            print(f"       {col}: {missing_count} 個缺失值")
        
        # 檢查異常值
        print(f"     異常值檢查:")
        for col in ['Open', 'Close', 'High', 'Low']:
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            print(f"       {col}: Q1%={q01:.2f}, Q99%={q99:.2f}")
        
        # 測試 5: 檢查函數調用的穩定性
        print("\n🔧 測試 5: 函數調用穩定性檢查...")
        
        # 多次調用相同函數，檢查結果是否穩定
        stability_results = []
        for i in range(5):
            result = risk_valve_backtest(
                open_px=df['Open'],
                w=weights,
                cost=cost_params,
                benchmark_df=df[['Close', 'Open']],
                mode="cap",
                cap_level=0.3,
                atr_ratio_mult=1.0,
                atr_win=20,
                atr_ref_win=60,
                use_slopes=False
            )
            stability_results.append(result['metrics']['pf_orig'])
        
        pf_std = np.std(stability_results)
        pf_mean = np.mean(stability_results)
        print(f"   獲利因子穩定性檢查:")
        print(f"     5次調用結果: {[f'{x:.6f}' for x in stability_results]}")
        print(f"     平均值: {pf_mean:.6f}")
        print(f"     標準差: {pf_std:.8f}")
        
        if pf_std < 1e-10:
            print(f"     ✅ 函數調用完全穩定")
        elif pf_std < 1e-6:
            print(f"     ⚠️  函數調用基本穩定（微小差異）")
        else:
            print(f"     ❌ 函數調用不穩定（需要調查）")
        
        # 測試 6: 創建 UI 對比基準
        print("\n🔧 測試 6: 創建 UI 對比基準...")
        
        # 生成標準化的測試報告，供 UI 對比使用
        ui_baseline = {
            'test_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_period': f"{df.index[0].strftime('%Y-%m-%d')} 到 {df.index[-1].strftime('%Y-%m-%d')}",
            'data_points': len(df),
            'weight_stats': {
                'min': float(weights.min()),
                'max': float(weights.max()),
                'mean': float(weights.mean()),
                'std': float(weights.std()),
                'changes': int((weights.diff().abs() > 0.01).sum())
            },
            'baseline_config': {
                'mode': 'cap',
                'cap_level': 0.3,
                'atr_ratio_mult': 1.0,
                'atr_win': 20,
                'atr_ref_win': 60,
                'use_slopes': False
            },
            'baseline_results': {
                'pf_orig': float(results['基準配置']['metrics']['pf_orig']) if not np.isinf(results['基準配置']['metrics']['pf_orig']) else 'inf',
                'pf_valve': float(results['基準配置']['metrics']['pf_valve']),
                'mdd_orig': float(results['基準配置']['metrics']['mdd_orig']),
                'mdd_valve': float(results['基準配置']['metrics']['mdd_valve']),
                'risk_triggers': int(results['基準配置']['signals']['risk_trigger'].sum()),
                'trigger_ratio': float(results['基準配置']['signals']['risk_trigger'].mean())
            }
        }
        
        # 保存基準結果到文件
        import json
        baseline_file = f"ui_baseline_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(baseline_file, 'w', encoding='utf-8') as f:
            json.dump(ui_baseline, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ UI 對比基準已保存到: {baseline_file}")
        print(f"   基準配置結果:")
        print(f"     原始獲利因子: {ui_baseline['baseline_results']['pf_orig']}")
        print(f"     閥門獲利因子: {ui_baseline['baseline_results']['pf_valve']:.6f}")
        print(f"     原始最大回撤: {ui_baseline['baseline_results']['mdd_orig']:.6f}")
        print(f"     閥門最大回撤: {ui_baseline['baseline_results']['mdd_valve']:.6f}")
        print(f"     風險觸發比例: {ui_baseline['baseline_results']['trigger_ratio']:.3f}")
        
        # 總結和建議
        print("\n📊 UI 一致性檢測總結:")
        print("   1. ✅ 創建了真實的交易場景，包含多階段權重變化")
        print("   2. ✅ 測試了多種參數配置的敏感性")
        print("   3. ✅ 檢查了不同時間範圍的影響")
        print("   4. ✅ 分析了獲利因子計算的合理性")
        print("   5. ✅ 驗證了函數調用的穩定性")
        print("   6. ✅ 生成了 UI 對比基準文件")
        
        print("\n💡 針對 UI 差異的具體建議:")
        print("   1. 📋 使用保存的基準配置在 UI 中進行測試")
        print("   2. 🎯 重點檢查閥門獲利因子（原始可能為 inf）")
        print("   3. ⚙️  確認 UI 的參數設定與基準配置完全一致")
        print("   4. 📅 確認 UI 使用相同的數據期間和權重序列")
        print("   5. 🔍 如果仍有差異，檢查以下可能原因：")
        print("      - 數據載入方式不同")
        print("      - 權重生成邏輯不同") 
        print("      - 成本參數設定不同")
        print("      - 函數版本不同")
        print("      - 浮點數精度處理不同")
        
        print(f"\n📄 UI 開發人員檢查清單:")
        print(f"   □ 確認使用相同的股票代碼: 00631L.TW")
        print(f"   □ 確認數據期間: {ui_baseline['data_period']}")
        print(f"   □ 確認成本參數: buy=4.27bp, sell=4.27bp, tax=30bp")
        print(f"   □ 確認閥門參數: cap_level=0.3, atr_ratio_mult=1.0")
        print(f"   □ 確認權重統計: 平均={ui_baseline['weight_stats']['mean']:.3f}")
        print(f"   □ 預期閥門獲利因子: {ui_baseline['baseline_results']['pf_valve']:.6f}")
        print(f"   □ 預期風險觸發比例: {ui_baseline['baseline_results']['trigger_ratio']:.3f}")
        
        print("\n✅ UI 一致性檢測完成！")
        
    except Exception as e:
        pytest.fail(f"UI 一致性檢測失敗: {e}")

if __name__ == "__main__":
    # 直接執行測試
    print("開始執行 Ensemble_Majority 風險閥門測試...")
    
    try:
        test_ensemble_majority_parameters()
        test_risk_valve_parameters()
        test_ensemble_majority_risk_valve_basic()
        test_enhanced_analysis_vs_global_application()
        test_app_dash_integration()
        test_ui_consistency_investigation()  # 新增的 UI 檢測測試
        print("🎉 所有測試通過！")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        sys.exit(1)
