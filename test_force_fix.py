# -*- coding: utf-8 -*-
"""
測試強制修復後的日誌系統
路徑：#test_force_fix.py
創建時間：2025-08-20 10:07
作者：AI Assistant

測試檔案日誌寫入是否強制修復
"""

import sys
import os
from pathlib import Path

# 添加專案路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 導入統一日誌配置
from analysis.logging_config import init_logging

def test_force_fix():
    """測試統一日誌系統"""
    print("=== 測試統一日誌系統 ===")
    
    # 初始化統一日誌系統
    init_logging(enable_file=True)
    
    # 獲取 logger
    logger = logging.getLogger("SSS.App")
    
    # 測試各種日誌級別
    logger.debug("這是 DEBUG 訊息 - 強制修復測試")
    logger.info("這是 INFO 訊息 - 強制修復測試")
    logger.warning("這是 WARNING 訊息 - 強制修復測試")
    logger.error("這是 ERROR 訊息 - 強制修復測試")
    
    # 測試中文日誌
    logger.info("測試中文日誌：股票策略系統 - 強制修復")
    logger.info("測試中文日誌：風險閥門功能 - 強制修復")
    
    # 測試多行日誌
    logger.info("測試多行日誌：\n第一行 - 強制修復\n第二行 - 強制修復\n第三行 - 強制修復")
    
    # 強制刷新
    for handler in logger.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()
    
    print("=== 統一日誌系統測試完成 ===")
    print("請檢查 analysis/log/app/ 目錄中的新日誌檔案")

if __name__ == "__main__":
    import logging
    test_force_fix()
