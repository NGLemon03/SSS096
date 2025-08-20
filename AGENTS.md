# SSS096 專案 AI 代理工作指南

## 📋 專案概述

SSS096 是一個股票策略回測與分析系統，主要包含：
- 策略回測引擎（SSSv096.py）
- Web UI 界面（app_dash.py）
- 增強分析模組（analysis/）
- Ensemble 策略執行（runners/）
- 數據處理與轉換工具

## 🎯 工作重點區域

### 核心檔案
- `SSSv096.py` - 主要策略回測引擎
- `app_dash.py` - Web UI 主應用
- `ensemble_wrapper.py` - Ensemble 策略包裝器
- `analysis/` - 分析模組目錄

### 避免修改的檔案
- `tools/quick_check.ps1` - 自動化檢查腳本（除非必要）
- 已標記為 "past/" 的舊版本檔案
- 編譯後的 `.pyc` 檔案

## 🔧 開發環境設定

### Codex 環境設置（推薦）
```bash
# 執行自動設置腳本
chmod +x setup.sh
./setup.sh

# 測試設置是否成功
python test_setup.py
```

### Python 環境（手動設置）
```bash
# 安裝依賴套件
pip install pandas numpy matplotlib seaborn openpyxl dash dash-bootstrap-components

# 檢查 Python 版本（建議 3.8+）
python --version
```

### 代理設置（如果遇到 403 錯誤）
```bash
# 檢查代理證書
echo $CODEX_PROXY_CERT

# 配置 pip 使用代理
pip config set global.cert "$CODEX_PROXY_CERT"
pip config set global.trusted-host "proxy:8080"
```

### 專案結構導航
```bash
# 快速查看目錄結構
python list_folder_structure.py

# 查看特定目錄內容
ls analysis/
ls runners/
```

## 🧪 測試與驗證

### 快速檢查
```bash
# 執行自動化檢查（重要！）
powershell -ExecutionPolicy Bypass -File tools\quick_check.ps1
```

### 回測測試
```bash
# 執行單一策略回測
python SSSv096.py --strategy RMA_Factor --param_preset op.json

# 執行 Ensemble 策略
python run_enhanced_ensemble.py --method majority --top_k 5
```

### UI 測試
```bash
# 啟動 Web UI
python app_dash.py
```

## 📝 程式碼規範

### 註解與輸出
- **一律使用繁體中文**進行註解和輸出
- 修改紀錄需加入日期時間戳記
- 路徑說明格式：`#子資料夾/檔案名`

### 日誌記錄
- 使用 `analysis/logging_config.py` 中的日誌器
- 重要操作需記錄到日誌檔案
- 錯誤處理需包含詳細的錯誤信息

### 資料格式
- 日期欄位統一使用 ISO 格式：`YYYY-MM-DD`
- 數值欄位使用 float 類型
- 避免使用中文欄位名稱（除非必要）

## 🔍 除錯指南

### 常見問題
1. **模組導入失敗**：檢查 `sys.path` 和相對導入
2. **數據格式錯誤**：驗證 CSV 檔案結構和欄位名稱
3. **記憶體不足**：檢查大數據集的處理方式

### 除錯工具
```python
# 啟用詳細日誌
import logging
logging.basicConfig(level=logging.DEBUG)

# 檢查數據結構
print(df.info())
print(df.head())
```

## 📊 數據處理規範

### 輸入數據
- 支援 CSV、Excel、JSON 格式
- 必要欄位：交易日期、權重變化、盈虧%
- 可選欄位：交易類型、價格、成交量

### 輸出數據
- 統一 Schema：equity、trades、daily_state、trade_ledger
- 避免 KeyError 和欄位缺失
- 支援多種輸出格式

## 🚀 部署與維護

### 檔案管理
- 定期清理舊的日誌和快取檔案
- 備份重要的配置和結果檔案
- 使用版本控制追蹤變更

### 性能優化
- 大數據集使用快取機制
- 避免重複計算
- 使用適當的數據結構

## ⚠️ 注意事項

### 安全考量
- 不要硬編碼 API 金鑰
- 驗證所有用戶輸入
- 保護敏感數據

### 相容性
- 維持與現有工作流程的相容性
- 測試所有整合點
- 避免破壞現有功能

## 🚀 Codex 環境設置

### 自動設置
```bash
# 執行設置腳本
./setup.sh

# 腳本會自動：
# 1. 配置代理設置
# 2. 安裝所有必要依賴
# 3. 創建回退日誌系統
# 4. 設置環境變數
```

### 手動設置（如果自動設置失敗）
```bash
# 1. 配置代理
export PIP_CERT="$CODEX_PROXY_CERT"
export NODE_EXTRA_CA_CERTS="$CODEX_PROXY_CERT"

# 2. 安裝核心依賴
pip install --no-cache-dir pandas numpy matplotlib seaborn openpyxl dash dash-bootstrap-components yfinance pyyaml joblib

# 3. 安裝分析套件
pip install --no-cache-dir scikit-learn scipy statsmodels plotly kaleido

# 4. 創建必要目錄
mkdir -p analysis/log analysis/cache cache log results sss_backtest_outputs
```

### 常見問題解決
- **pip install 403 錯誤**：檢查 `$CODEX_PROXY_CERT` 環境變數
- **joblib 導入失敗**：使用 `logging_config_fallback.py` 回退版本
- **模組路徑問題**：設置 `PYTHONPATH` 環境變數

## 📞 技術支援

### 問題回報
- 提供完整的錯誤訊息和堆疊追蹤
- 包含重現步驟和環境信息
- 檢查相關的日誌檔案

### 文檔更新
- 修改功能時同步更新相關文檔
- 使用清晰的範例和說明
- 保持文檔的時效性

---

**版本：** v1.0  
**最後更新：** 2025-08-18  
**適用於：** SSS096 專案 AI 代理工作指南
