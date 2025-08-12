# 目录规划与日志系统更新总结

## 已完成的更新

### 1. 更新 analysis/logging_config.py

✅ **新增专用文件 handler**
- `ensemble_file`: 用于多策略组合日志
- `app_file`: 用于应用界面日志

✅ **新增项目内主要 logger**
- `SSSv096`: 单策略回测日志
- `SSS_Ensemble`: 多策略组合日志  
- `app_dash`: 应用界面日志

### 2. 更新 SSS_EnsembleTab.py

✅ **改用集中配置和专名 logger**
- 导入 `analysis.logging_config` 和 `analysis.config`
- 使用 `SSS_Ensemble` logger 替代 `__name__`

✅ **增加 run 级别的文件输出**
- 每次执行多策略时在 `log/ensemble/` 生成独立日志文件
- 文件名格式: `ensemble_{method}_{ticker}_{timestamp}.log`

✅ **新增逐日交易台帐 CSV**
- 在 `log/ensemble/` 生成 `ledger_{method}_{ticker}_{timestamp}.csv`
- 包含每日详细信息：日期、开盘价、多头数量、权重变化、交易成本、权益等

✅ **关键配置摘要日志**
- N策略数量、方法、门限、成本等关键参数
- 便于排查配置差异

### 3. 更新 app_dash.py

✅ **改用集中配置和专名 logger**
- 导入 `analysis.logging_config`
- 使用 `app_dash` logger 替代 `logging.basicConfig`

✅ **补充 ensemble 参数摘要日志**
- 调用前记录方法、参数、成本、股票代码
- 调用后记录输出文件路径

### 4. 更新 SSSv096.py

✅ **添加策略级别独立文件日志工具**
- `attach_strategy_logger()` 函数
- 每次回测各策略时生成独立日志文件
- 文件路径: `log/sss/SSSv096_{ticker}_{strat_name}_{timestamp}.log`

## 目录结构

```
C:\Stock_reserach\SSS096\
├─ log\
│  ├─ app\                 # app 相关
│  │  └─ app_dash_*.log
│  ├─ sss\                 # 单策略回测（SSSv096）
│  │  └─ SSSv096_*_*_*.log
│  ├─ ensemble\            # 多策略组合（SSS_EnsembleTab）
│  │  ├─ ensemble_*_*_*.log
│  │  └─ ledger_*_*_*.csv   # 逐日权重/交易明细
│  ├─ System_*.log, Errors_*.log ...（保留原有）
│  └─ ...
└─ sss_backtest_outputs\   # 既有输出维持不变
   ├─ ensemble_weights_*.csv
   ├─ ensemble_equity_*.csv
   └─ ensemble_trades_*.csv
```

## 使用方法

### 运行测试
```bash
python test_logging_system.py
```

### 运行 Ensemble
```bash
python SSS_EnsembleTab.py --ticker 00631L.TW --method majority
```

### 运行 App
```bash
python app_dash.py
```

## 关键特性

1. **统一路径管理**: 使用 `analysis/config.py` 的 `LOG_DIR` 作为统一路径
2. **集中日志配置**: 所有日志配置集中在 `analysis/logging_config.py`
3. **专名 logger**: 每个模块使用专门的 logger 名称
4. **独立文件日志**: 每次执行生成独立的日志文件和交易台帐
5. **详细交易记录**: 逐日记录权重变化、交易成本、权益曲线等

## 验证步骤

1. 运行测试脚本检查日志系统
2. 运行一次 app 查看 `log/app/app_dash_*.log` 是否写入参数摘要
3. 运行一次 ensemble 查看 `log/ensemble/ensemble_*_*_*.log` 是否有配置摘要
4. 检查 `log/ensemble/ledger_*_*_*.csv` 是否生成
5. 比对 app 与 ensemble 的 N、K_eff 是否一致

## 注意事项

- 所有路径都使用 `analysis/config.py` 中的配置，避免硬编码
- 日志文件使用 UTF-8-SIG 编码，支持中文
- 每次执行都会生成带时间戳的独立文件，便于追踪和调试
- 交易台帐包含详细的逐日信息，便于排查差异
