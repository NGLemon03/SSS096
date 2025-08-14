import pandas as pd
import os
import glob
from datetime import datetime
import logging

# 設置 logger
import sys
sys.path.append('..')
from analysis.logging_config import LOGGING_DICT
import logging.config
logging.config.dictConfig(LOGGING_DICT)
logger = logging.getLogger("SSS.FixTrades")

def fix_trades_columns():
    """批次轉換 trades_from_results_*.csv 的欄位名稱，使其符合 Ensemble 期望格式"""
    
    # 掃描所有 trades_from_results_*.csv 檔案
    files = glob.glob("trades_from_results_*.csv")
    logger.info(f"找到 {len(files)} 個 trades_from_results_*.csv 檔案")
    
    fixed_count = 0
    for fp in files:
        try:
            df = pd.read_csv(fp)
            
            # 檢查是否已經有正確的欄位名稱
            if 'date' in df.columns and 'action' in df.columns:
                logger.info(f"跳過 {fp} - 已符合格式")
                continue
                
            # 檢查是否有必要的欄位
            if 'trade_date' not in df.columns or 'type' not in df.columns:
                logger.warning(f"跳過 {fp} - 缺少必要欄位: {list(df.columns)}")
                continue
            
            # 轉換欄位名稱
            df_fixed = df.copy()
            df_fixed = df_fixed.rename(columns={
                'trade_date': 'date',
                'type': 'action'
            })
            
            # 確保日期格式正確（去除時區，標準化為日期）
            df_fixed['date'] = pd.to_datetime(df_fixed['date']).dt.date.astype(str)
            
            # 保存轉換後的檔案
            df_fixed.to_csv(fp, index=False)
            
            logger.info(f"已修復 {fp}: {len(df)} 筆交易")
            fixed_count += 1
            
        except Exception as e:
            logger.error(f"處理 {fp} 時出錯: {e}")
    
    logger.info(f"總共修復了 {fixed_count} 個檔案")
    
    # 檢查修復結果
    logger.info("檢查修復結果:")
    for fp in files[:3]:  # 只檢查前3個
        try:
            df = pd.read_csv(fp)
            logger.info(f"{fp}: 欄位={list(df.columns)}, 筆數={len(df)}")
            if 'action' in df.columns:
                logger.info(f"  action統計: {df['action'].value_counts().to_dict()}")
        except Exception as e:
            logger.error(f"檢查 {fp} 時出錯: {e}")

if __name__ == "__main__":
    fix_trades_columns()
