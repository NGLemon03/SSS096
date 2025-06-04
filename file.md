專案根目錄/
├─ version_history.py       # 版本沿革顯示模組：列出每次重要更新紀錄
├─ SSSv092.py               # 核心：指標計算（compute_ssma_turn_combined、各策略）與 backtest_unified 回測函式
├─ leverage.py              # 槓桿 ETF 操作工具或範例程式
└─ analysis/
    ├─ config.py            # 全域設定：載入 JSON 網格 (PR)，以及交易成本、冷卻期、預設止損等常數
    ├─ grids/
    │   ├─ test.json        # 超精簡網格 (≈1 k 組合) → 全流程 ≤ 10 分鐘
    │   ├─ sample.json      # 開發樣本網格 (中等規模) → 快速觀察策略行為
    │   ├─ rough.json       # 常規網格 (≈117 k 組合) → Grid ≈ 60 分鐘、Pipeline ≈ 78 分鐘
    │   └─ full.json        # 完整網格 (最嚴格) → 最長跑程式時使用
    ├─ walk_forward_v14.py  # Walk‐forward 分析腳本：依 PR 網格自動拆參、跑多期步進回測
    ├─ grid_search_v14.py   # Grid‐search 分析腳本：依 PR 網格自動拆參、跑完整回測並輸出 CSV
    ├─ exit_shift_test_v2.py# Exit‐shift 敏感度測試：將 buy/exit shift 網格化，專門測試訊號平移效果
    ├─ ROEA.py              # ROEA 指標後處理與報表：根據回測結果計算 ROEA、繪圖、生成報表
    └─ stress_testv1.py     # 壓力測試腳本：套用多段重大崩盤期間檢驗策略穩健度
