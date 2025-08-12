# Ensemble 策略交易成本功能説明

## 概述

`run_enhanced_ensemble.py` 現在支持交易成本設置，與 SSS 的默認設置保持一致。這確保了 ensemble 策略的回測結果更加貼近實際交易環境。

## 交易成本設置

### 默認費率（與 SSS 一致）

- **基礎手續費**: 0.1425%
- **賣出税率**: 0.3%
- **默認折扣**: 0.3
- **實際買進費率**: 0.1425% × 0.3 = 0.04275%
- **實際賣出費率**: 0.1425% × 0.3 + 0.3% = 0.34275%
- **往返成本**: 0.04275% + 0.34275% = 0.3855%

### 費率計算

```python
# 買進費率 (bp)
buy_fee_bp = BASE_FEE_RATE * discount * 10000

# 賣出費率 (bp)  
sell_fee_bp = BASE_FEE_RATE * discount * 10000

# 税率 (bp)
sell_tax_bp = TAX_RATE * 10000
```

## 使用方法

### 1. 使用默認交易成本（推薦）

```bash
python run_enhanced_ensemble.py --method majority --scan_params
```
python run_enhanced_ensemble.py --method proportional --scan_params 

這將使用默認的 0.3 折扣率，即：
- 買進費率: 4.27bp (0.0427%)
- 賣出費率: 4.27bp (0.0427%)  
- 税率: 30.00bp (0.3%)

### 2. 自定義折扣率

```bash
# 使用 0.5 折扣率
python run_enhanced_ensemble.py --method majority --discount 0.5 --scan_params

# 使用 1.0 折扣率（無折扣）
python run_enhanced_ensemble.py --method majority --discount 1.0 --scan_params
```

### 3. 無交易成本模式（僅用於對比測試）

```bash
python run_enhanced_ensemble.py --method majority --no_cost --scan_params
```

## 參數説明

### 新增命令行參數

- `--discount FLOAT`: 交易成本折扣率，默認 0.3
- `--no_cost`: 不使用交易成本（僅用於對比測試）

### 交易成本參數結構

```python
cost_params = {
    'buy_fee_bp': 4.27,    # 買進費率 (bp)
    'sell_fee_bp': 4.27,   # 賣出費率 (bp)
    'sell_tax_bp': 30.0    # 税率 (bp)
}
```

## 技術實現

### 1. 交易成本應用

交易成本在 `equity_open_to_open` 函數中應用：

```python
# 買進成本
if dW > 0:  # 增加多倉
    c = E * dW * cost.buy_rate

# 賣出成本（含税）
else:      # 減少多倉
    c = E * (-dW) * cost.sell_rate

# 扣除成本
E -= c
```

### 2. 成本計算時機

- 成本在每個交易日開盤時扣除
- 基於前一日的權益和權重變化計算
- 支持部分倉位調整的成本計算

## 性能影響

### 1. 對報酬率的影響

- **無成本模式**: 理論最優表現
- **默認成本模式**: 更貼近實際交易環境
- **高成本模式**: 反映高手續費環境下的表現

### 2. 對交易頻率的影響

- 交易成本會降低頻繁調整的吸引力
- 有助於識別真正有效的策略信號
- 減少過度擬合的風險

## 對比分析

### 建議的測試流程

1. **基準測試**: 使用默認交易成本運行
2. **無成本對比**: 使用 `--no_cost` 參數運行
3. **高成本測試**: 使用 `--discount 1.0` 參數運行
4. **結果分析**: 比較不同成本設置下的表現差異

### 結果文件命名

結果文件會包含成本信息：

```
ensemble_results_majority_20241201_143000.txt
```

文件內容包含：
- 集成方法
- 參數設置  
- 交易成本設置
- 績效指標

## 注意事項

### 1. 成本設置一致性

- 建議與 SSS 策略使用相同的成本設置
- 確保 ensemble 和單個策略的可比性
- 避免成本設置不一致導致的誤導性結果

### 2. 參數掃描優化

- 交易成本會影響最優參數組合
- 建議在有成本的情況下進行參數優化
- 考慮成本因素下的風險調整後收益

### 3. 回測週期

- 短期回測中成本影響較小
- 長期回測中成本累積效應明顯
- 建議在足夠長的週期內評估成本影響

## 故障排除

### 常見問題

1. **導入錯誤**: 確保 `ensemble_wrapper.py` 和 `SSS_EnsembleTab.py` 存在
2. **參數錯誤**: 檢查折扣率是否在合理範圍內 (0.0-1.0)
3. **性能異常**: 驗證交易成本參數是否正確傳遞

### 調試方法

使用測試腳本驗證功能：

```bash
python test_ensemble_cost.py
```

這將檢查：
- 交易成本參數計算
- 模塊導入
- 參數傳遞
- 功能集成

## 總結

通過添加交易成本支持，`run_enhanced_ensemble.py` 現在能夠：

1. **更準確的回測**: 考慮實際交易成本
2. **與 SSS 一致**: 使用相同的成本設置
3. **靈活配置**: 支持不同的成本水平
4. **對比分析**: 支持有成本和無成本的對比測試

這確保了 ensemble 策略的回測結果更加可靠和實用。
