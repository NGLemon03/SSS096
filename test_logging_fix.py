# -*- coding: utf-8 -*-
"""
測試修復後的日誌系統
路徑：#test_logging_fix.py
創建時間：2025-08-20 09:45
作者：AI Assistant

測試檔案日誌寫入是否正常
"""

import sys
import os
from pathlib import Path

# 添加專案路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 導入統一日誌配置
from analysis.logging_config import init_logging

def test_logging():
    """測試統一日誌系統"""
    print("=== 測試統一日誌系統 ===")
    
    # 初始化統一日誌系統
    init_logging(enable_file=True)
    
    # 獲取 logger
    logger = logging.getLogger("SSS.App")
    
    # 測試各種日誌級別
    logger.debug("這是 DEBUG 訊息")
    logger.info("這是 INFO 訊息")
    logger.warning("這是 WARNING 訊息")
    logger.error("這是 ERROR 訊息")
    
    # 測試中文日誌
    logger.info("測試中文日誌：股票策略系統")
    logger.info("測試中文日誌：風險閥門功能")
    
    # 測試多行日誌
    logger.info("測試多行日誌：\n第一行\n第二行\n第三行")
    
    # 強制刷新
    for handler in logger.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()
    
    print("=== 統一日誌系統測試完成 ===")
    print("請檢查 analysis/log/app/ 目錄中的新日誌檔案")

if __name__ == "__main__":
    import logging
    test_logging()
