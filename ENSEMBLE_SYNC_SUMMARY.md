# Ensemble配置同步修改总结

## 概述
本次修改确保了app_dash.py和SSSv096.py中的ensemble策略配置完全一致，避免因配置差异导致的回测结果不一致问题。

## 修改内容

### 第1步：确认载入文件
- **文件**: `app_dash.py`
- **修改**: 在ensemble策略执行前添加了logger.info，显示使用的SSS_EnsembleTab.py文件路径
- **目的**: 确保app使用的是最新版本的SSS_EnsembleTab.py

```python
import SSS_EnsembleTab as ens
logger.info(f"[Ensemble] using {ens.__file__}")
```

### 第2步：在run_ensemble印关键摘要
- **文件**: `SSS_EnsembleTab.py`
- **修改**: 在run_ensemble函数中，pos矩阵组好后添加了关键摘要日志
- **目的**: 一次就抓到问题点，便于调试

```python
logger.info(f"[Ensemble] N_strategies={pos_df.shape[1]}")
if getattr(cfg, "majority_k_pct", None):
    k_eff = max(1, int(round(pos_df.shape[1] * cfg.majority_k_pct)))
    logger.info(f"[Ensemble] majority_k_pct={cfg.majority_k_pct} -> K_eff={k_eff}")
logger.info(f"[Ensemble] params: floor={cfg.params.floor}, ema_span={cfg.params.ema_span}, "
            f"delta_cap={cfg.params.delta_cap}, cooldown={cfg.params.min_cooldown_days}, "
            f"min_trade_dw={getattr(cfg.params,'min_trade_dw',None)}; "
            f"cost(bp): buy={cfg.cost.buy_fee_bp}, sell={cfg.cost.sell_fee_bp}, tax={cfg.cost.sell_tax_bp}")
```

### 第3步：统一路径与preset
- **文件**: `app_dash.py`, `SSSv096.py`, `SSS_EnsembleTab.py`
- **修改**: 确保app与SSS都从同一个`sss_backtest_outputs/results`位置读取子策略trades_*.csv，并保证参数一致

#### 3.1 统一参数配置
- **floor**: 0.2
- **ema_span**: 3
- **delta_cap**: 0.3
- **min_cooldown_days**: 1 (与param_presets一致)
- **min_trade_dw**: 0.01 (与param_presets一致)

#### 3.2 统一成本参数
- **buy_fee_bp**: 4.27
- **sell_fee_bp**: 4.27
- **sell_tax_bp**: 30.0

#### 3.3 统一majority_k_pct
- **强制设置**: 0.55 (55%门槛)
- **目的**: 避免N变动时失真，确保K值始终为策略数量的55%

#### 3.4 统一路径
- **OUT_DIR**: `sss_backtest_outputs` (与SSS项目根目录一致)
- **目的**: 确保app与SSS读取相同的trades文件

## 关键改进点

### 1. 比例门槛机制
- 使用`majority_k_pct=0.55`替代固定K值
- 自动根据实际策略数量计算K_eff
- 避免因策略数量变化导致的K值不匹配

### 2. 参数同步
- app_dash.py和SSSv096.py使用完全相同的默认参数
- 成本参数与param_presets中的配置一致
- 路径设置统一，确保数据源一致

### 3. 调试信息增强
- 添加详细的日志输出
- 显示N_strategies、K_eff、参数、成本等关键信息
- 便于快速定位配置差异

## 验证结果
- ✓ 成功导入SSS_EnsembleTab
- ✓ 路径设置正确
- ✓ 找到150个trades文件
- ✓ 参数配置一致
- ✓ 成本参数统一
- ✓ majority_k_pct设置正确

## 使用建议
1. **运行app_dash.py时**: 检查日志中的`[Ensemble] using {ens.__file__}`，确保使用的是最新版本
2. **对比日志**: 运行ensemble策略后，对比app和SSS的日志，确保N、K_eff、参数、成本完全一致
3. **参数调整**: 如需修改ensemble参数，建议在SSSv096.py的param_presets中统一修改

## 注意事项
- 确保`sss_backtest_outputs`目录下有足够的trades_*.csv文件
- majority_k_pct设置为0.55，适合大多数情况
- 成本参数已设置为接近台股实盘的水平
