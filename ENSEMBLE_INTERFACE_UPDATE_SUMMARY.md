# Ensemble 接口統一更新總結

## 更新概述

本次更新統一了 `run_ensemble` 函數的回傳介面，從原本的 6 個值改為 8 個值，讓 app 和 v096 都能接收到 `daily_state` 和 `trade_ledger` 這兩張表，從而獲得真實的金額和資金占比信息。

## 主要修改

### 1. 統一 `run_ensemble` 回傳介面

**舊格式（6值）：**
```python
open_px, w, trades, stats, method_name, equity = run_ensemble(cfg)
```

**新格式（8值）：**
```python
open_px, w, trades, stats, method_name, equity, daily_state, trade_ledger = run_ensemble(cfg)
```

### 2. 修改的文件

#### `SSS_EnsembleTab.py`
- ✅ `run_ensemble` 函數已回傳 8 個值
- ✅ `save_outputs` 函數新增保存 `daily_state` 和 `trade_ledger` 功能
- ✅ 新增輸出文件：
  - `ensemble_ledger_daily_<method_name>.csv`（對應 daily_state）
  - `ensemble_ledger_trades_<method_name>.csv`（對應 trade_ledger）

#### `app_dash.py`
- ✅ 更新 ensemble 調用為 8 值解包
- ✅ 移除手工構造 `trade_records` 的簡化代碼
- ✅ 直接使用 `trade_ledger` 的真實交易數據
- ✅ 新增投資組合狀態信息到 metrics
- ✅ 在 result 中添加 `daily_state` 和 `trade_ledger`

#### `SSSv096.py`
- ✅ 更新 ensemble 調用為 8 值解包
- ✅ 移除手工構造 `trade_records` 的簡化代碼
- ✅ 直接使用 `trade_ledger` 的真實交易數據
- ✅ 新增投資組合狀態信息到 metrics

#### `ensemble_wrapper.py`
- ✅ 更新 ensemble 調用為 8 值解包
- ✅ 更新返回值類型注解
- ✅ 更新兼容性函數文檔

### 3. 新增功能

#### 投資組合流水帳（Portfolio Ledger）
- **daily_state**: 每日資產狀態表，包含：
  - `w`: 當前權重
  - `dw`: 權重變化
  - `cash_pct`: 現金比例
  - `invested_pct`: 投入比例
  - `cash`: 現金金額
  - `position_value`: 持倉價值
  - `equity`: 總資產
  - `units`: 持倉單位

- **trade_ledger**: 交易流水帳，包含：
  - `side`: 交易方向（buy/sell）
  - `exec_notional`: 實際交易金額
  - `fee_buy/fee_sell`: 買賣費用
  - `tax`: 證交稅
  - `fees_total`: 總費用
  - `cash_after`: 交易后現金
  - `equity_after`: 交易后總資產
  - `trade_pct`: 本次調整佔用資金比例

### 4. 輸出文件命名

新增的 CSV 文件命名規則：
- `sss_backtest_outputs/ensemble_ledger_daily_<method_name>.csv`（對應 daily_state）
- `sss_backtest_outputs/ensemble_ledger_trades_<method_name>.csv`（對應 trade_ledger）

與 SSS 端完全對齊，確保兩端使用相同的數據源。

### 5. 參數配置

#### 成本參數（與 app/v096 preset 一致）
- `buy_fee_bp`: 4.27 bp
- `sell_fee_bp`: 4.27 bp  
- `sell_tax_bp`: 30.0 bp

#### 資金參數
- `initial_capital`: 預設 1,000,000
- `lot_size`: 預設 None（允許零股），台股可設為 1000（整張）

#### 多數決門檻
- 強制使用比例門檻 `majority_k_pct=0.55`

## 使用方式

### 在 app 中使用
```python
# 調用 ensemble 策略
open_px, w, trades, stats, method_name, equity, daily_state, trade_ledger = run_ensemble(cfg)

# 獲取當前投資組合狀態
if not daily_state.empty:
    latest_state = daily_state.iloc[-1]
    current_weight = latest_state['w']
    cash_pct = latest_state['cash_pct']
    invested_pct = latest_state['invested_pct']
    current_cash = latest_state['cash']
    position_value = latest_state['position_value']
    total_equity = latest_state['equity']

# 獲取交易明細
if not trade_ledger.empty:
    num_trades = len(trade_ledger)
    latest_trade = trade_ledger.iloc[-1]
    trade_amount = latest_trade['exec_notional']
    trade_fees = latest_trade['fees_total']
```

### 在 v096 中使用
```python
# 調用 ensemble 策略
open_px, w, trades, stats, method_name, equity, daily_state, trade_ledger = run_ensemble(cfg)

# 使用真實的交易數據和投資組合狀態
# 不再需要手工構造簡化的 trade_records
```

## 優勢

1. **真實數據**: 不再使用 `shares=1.0` 的簡化記錄，而是真實的交易金額和單位
2. **完整信息**: 提供現金比例、投入比例、持倉價值等完整的投資組合狀態
3. **成本透明**: 清楚顯示每筆交易的費用、稅金和總成本
4. **資金管理**: 可以追蹤每次調整的資金佔用比例
5. **統一介面**: app 和 v096 使用完全相同的數據源和格式

## 注意事項

1. **向後兼容**: 舊的 6 值調用方式已不再支持
2. **數據完整性**: 確保 `daily_state` 和 `trade_ledger` 不為空再使用
3. **文件路徑**: 新的 ledger 文件會保存在 `sss_backtest_outputs/` 目錄下
4. **參數一致性**: 成本參數與現有 preset 保持一致，避免差異

## 測試建議

1. 在 app 中執行 ensemble 策略，確認能正確接收 8 個返回值
2. 檢查生成的 `ensemble_ledger_*.csv` 文件內容
3. 驗證 UI 中顯示的投資組合狀態信息
4. 確認交易次數統計使用 `trade_ledger` 的實際數據
5. 測試 CSV 匯出功能是否正常工作

## 下一步

完成本次更新後，app 和 v096 將能夠：
- 顯示真實的交易金額和資金占比
- 提供完整的投資組合狀態信息
- 使用統一的數據源和格式
- 支持更精確的實盤決策分析
