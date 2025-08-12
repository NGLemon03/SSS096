#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版 trades 列名修復腳本

支持多種輸入格式，統一轉換為標準格式：
- 輸入：date/trade_date + action/type + price + weight/shares
- 輸出：date + action + price + weight

使用方法:
python fix_trades_columns_enhanced.py
"""

import pandas as pd
import os
import glob
from datetime import datetime
from pathlib import Path

def fix_trades_columns_enhanced():
    """批次轉換 trades_from_results_*.csv 的欄位名稱，使其符合 Ensemble 期望格式"""
    
    # 掃描所有 trades_from_results_*.csv 檔案
    files = glob.glob("trades_from_results_*.csv")
    print(f"找到 {len(files)} 個 trades_from_results_*.csv 檔案")
    
    if not files:
        print("未找到 trades_from_results_*.csv 檔案")
        return
    
    fixed_count = 0
    for fp in files:
        try:
            df = pd.read_csv(fp)
            print(f"\n處理檔案: {fp}")
            print(f"  原始欄位: {list(df.columns)}")
            print(f"  原始筆數: {len(df)}")
            
            # 檢查是否已經有正確的欄位名稱
            if 'date' in df.columns and 'action' in df.columns:
                print(f"  跳過 - 已符合標準格式")
                continue
            
            # 欄位名稱映射表
            column_mapping = {}
            
            # 日期欄位映射
            date_columns = ['date', 'trade_date', 'Date', 'Trade_Date']
            for col in date_columns:
                if col in df.columns:
                    column_mapping[col] = 'date'
                    break
            
            # 動作欄位映射
            action_columns = ['action', 'type', 'Action', 'Type']
            for col in action_columns:
                if col in df.columns:
                    column_mapping[col] = 'action'
                    break
            
            # 價格欄位映射
            price_columns = ['price', 'Price', 'close', 'Close']
            for col in price_columns:
                if col in df.columns:
                    column_mapping[col] = 'price'
                    break
            
            # 權重/股數欄位映射
            weight_columns = ['weight', 'Weight', 'shares', 'Shares']
            for col in weight_columns:
                if col in df.columns:
                    column_mapping[col] = 'weight'
                    break
            
            # 檢查必要欄位
            required_columns = ['date', 'action', 'price']
            missing_columns = [col for col in required_columns if col not in column_mapping.values()]
            
            if missing_columns:
                print(f"  跳過 - 缺少必要欄位: {missing_columns}")
                continue
            
            # 轉換欄位名稱
            df_fixed = df.copy()
            df_fixed = df_fixed.rename(columns=column_mapping)
            
            # 確保只保留標準欄位
            standard_columns = ['date', 'action', 'price', 'weight']
            df_fixed = df_fixed[standard_columns]
            
            # 確保日期格式正確（去除時區，標準化為日期）
            df_fixed['date'] = pd.to_datetime(df_fixed['date'], errors='coerce')
            df_fixed = df_fixed.dropna(subset=['date'])
            df_fixed['date'] = df_fixed['date'].dt.date.astype(str)
            
            # 標準化動作欄位
            df_fixed['action'] = df_fixed['action'].astype(str).str.lower()
            
            # 確保價格為數值
            df_fixed['price'] = pd.to_numeric(df_fixed['price'], errors='coerce')
            df_fixed = df_fixed.dropna(subset=['price'])
            
            # 確保權重為數值
            df_fixed['weight'] = pd.to_numeric(df_fixed['weight'], errors='coerce')
            df_fixed['weight'] = df_fixed['weight'].fillna(1.0)
            
            # 保存轉換後的檔案
            df_fixed.to_csv(fp, index=False)
            
            print(f"  已修復: {len(df_fixed)} 筆交易")
            print(f"  最終欄位: {list(df_fixed.columns)}")
            print(f"  動作統計: {df_fixed['action'].value_counts().to_dict()}")
            
            fixed_count += 1
            
        except Exception as e:
            print(f"  處理 {fp} 時出錯: {e}")
    
    print(f"\n總共修復了 {fixed_count} 個檔案")
    
    # 檢查修復結果
    print("\n檢查修復結果:")
    for fp in files[:3]:  # 只檢查前3個
        try:
            df = pd.read_csv(fp)
            print(f"{fp}: 欄位={list(df.columns)}, 筆數={len(df)}")
            if 'action' in df.columns:
                print(f"  action統計: {df['action'].value_counts().to_dict()}")
        except Exception as e:
            print(f"檢查 {fp} 時出錯: {e}")

def check_trades_format():
    """檢查所有 trades 檔案的格式"""
    print("\n" + "="*60)
    print("檢查所有 trades 檔案格式")
    print("="*60)
    
    # 掃描所有 trades*.csv 檔案
    all_files = glob.glob("trades*.csv")
    print(f"找到 {len(all_files)} 個 trades*.csv 檔案")
    
    format_stats = {
        'standard': 0,      # date + action
        'legacy': 0,        # trade_date + type
        'mixed': 0,         # 其他組合
        'invalid': 0        # 無效格式
    }
    
    for fp in all_files:
        try:
            df = pd.read_csv(fp)
            columns = list(df.columns)
            
            if 'date' in columns and 'action' in columns:
                format_stats['standard'] += 1
                print(f"✓ {fp}: 標準格式")
            elif 'trade_date' in columns and 'type' in columns:
                format_stats['legacy'] += 1
                print(f"⚠ {fp}: 舊格式 (trade_date + type)")
            elif ('date' in columns or 'trade_date' in columns) and ('action' in columns or 'type' in columns):
                format_stats['mixed'] += 1
                print(f"⚠ {fp}: 混合格式")
            else:
                format_stats['invalid'] += 1
                print(f"✗ {fp}: 無效格式 - {columns}")
                
        except Exception as e:
            format_stats['invalid'] += 1
            print(f"✗ {fp}: 讀取錯誤 - {e}")
    
    print(f"\n格式統計:")
    print(f"  標準格式 (date + action): {format_stats['standard']}")
    print(f"  舊格式 (trade_date + type): {format_stats['legacy']}")
    print(f"  混合格式: {format_stats['mixed']}")
    print(f"  無效格式: {format_stats['invalid']}")

if __name__ == "__main__":
    print("="*60)
    print("增強版 trades 列名修復腳本")
    print("="*60)
    
    # 檢查當前格式
    check_trades_format()
    
    # 執行修復
    print("\n" + "="*60)
    print("開始修復列名...")
    print("="*60)
    fix_trades_columns_enhanced()
    
    # 再次檢查格式
    print("\n" + "="*60)
    print("修復後格式檢查")
    print("="*60)
    check_trades_format()
