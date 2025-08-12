# SSS_EnsembleTab.py 修复总结

## 问题描述

根据用户反馈，SSS（streamlit）和app_dash.py之间存在ensemble参数不一致的问题：

1. **参数不一致**：
   - SSS使用旧参数：`majority_k=6`、`min_cooldown_days=3`、`buy_fee_bp/sell_fee_bp=15.0`
   - app使用新参数：`majority_k_pct=0.55`、`min_cooldown_days=1`、`buy_fee_bp/sell_fee_bp=4.27`

2. **路径冲突**：
   - SSS_EnsembleTab.py中存在重复的模块头，导致路径常量互相冲突
   - 可能读取到错误位置的价量CSV文件

## 修复内容

### 1. 清理重复模块头 ✅
- 删除了重复的模块头定义
- 确保只有一份正确的路径常量定义：
  ```python
  BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
  DATA_DIR = BASE_DIR / "data"
  OUT_DIR = BASE_DIR / "sss_backtest_outputs"
  ```

### 2. 更新默认参数 ✅
- 更新`EnsembleParams`默认值以匹配`param_presets`：
  ```python
  min_cooldown_days: int = 1  # 与param_presets一致
  min_trade_dw: float = 0.01  # 与param_presets一致
  ```

### 3. 强制使用比例门槛 ✅
- 在`main()`函数中强制设置`majority_k_pct = 0.55`
- 确保与app_dash.py的配置保持一致

### 4. 优先使用新策略文件 ✅
- 修改策略文件扫描逻辑，优先使用`trades_from_results_*.csv`（120档策略）
- 找不到时才使用旧的`trades_*.csv`（11档策略）

### 5. 更新成本参数 ✅
- 更新`CostParams`默认值：
  ```python
  buy_fee_bp: float = 4.27   # 与param_presets一致
  sell_fee_bp: float = 4.27  # 与param_presets一致
  sell_tax_bp: float = 30.0  # 与param_presets一致
  ```

## 修复后的配置对比

| 参数 | 修复前 | 修复后 | 说明 |
|------|--------|--------|------|
| `majority_k` | 6 | 6 | 仅在没有`majority_k_pct`时使用 |
| `majority_k_pct` | 无 | 0.55 | 强制使用比例门槛 |
| `min_cooldown_days` | 3 | 1 | 与param_presets一致 |
| `min_trade_dw` | 0.01 | 0.01 | 与param_presets一致 |
| `buy_fee_bp` | 15.0 | 4.27 | 与param_presets一致 |
| `sell_fee_bp` | 15.0 | 4.27 | 与param_presets一致 |
| `sell_tax_bp` | 30.0 | 30.0 | 保持不变 |
| 策略文件优先级 | `trades_*.csv` | `trades_from_results_*.csv` | 优先使用120档策略 |

## 验证结果

运行测试脚本`test_ensemble_fix.py`的结果：

```
============================================================
测试结果: 4/4 通过
🎉 所有测试通过！SSS_EnsembleTab.py 修复成功

修复内容总结:
1. ✓ 清理了重复的模块头
2. ✓ 更新了默认参数以匹配param_presets
3. ✓ 强制使用majority_k_pct=0.55
4. ✓ 优先使用trades_from_results_*.csv文件
5. ✓ 使用正确的成本参数（4.27 bp + 30 bp税）
6. ✓ 使用正确的冷却和微调参数（cooldown=1, min_trade_dw=0.01/0.03）
```

## 预期效果

修复后，SSS和app_dash.py的ensemble策略应该：

1. **参数一致**：都使用`majority_k_pct=0.55`、`min_cooldown_days=1`、成本4.27bp
2. **策略数量一致**：都使用120档策略（`trades_from_results_*.csv`）
3. **权益曲线一致**：读取相同位置的价量CSV，使用相同的参数配置
4. **日志显示一致**：显示相同的策略数量、门槛、成本等信息

## 使用建议

1. **测试验证**：运行SSS和app_dash.py的ensemble策略，对比结果是否一致
2. **日志检查**：确认日志显示：
   - 策略数量：~120（不是11）
   - 门槛：`majority_k_pct=0.55`
   - 成本：`buy_fee_bp=4.27`、`sell_fee_bp=4.27`、`sell_tax_bp=30`
   - 冷却：`min_cooldown_days=1`
   - 数据路径：`.../data/00631L.TW_data_raw.csv`

3. **参数调整**：如需修改ensemble参数，建议在`SSSv096.py`的`param_presets`中调整，确保两边的配置同步

## 文件清单

- `SSS_EnsembleTab.py` - 主要修复文件
- `test_ensemble_fix.py` - 验证测试脚本
- `ENSEMBLE_FIX_SUMMARY.md` - 本修复总结文档
