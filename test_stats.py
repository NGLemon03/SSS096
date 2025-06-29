#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime, timedelta

def calculate_strategy_detailed_stats(trade_df, df_raw):
    """計算策略的詳細統計信息"""
    if trade_df.empty:
        return {
            'avg_holding_days': 0,
            'avg_sell_to_buy_days': 0,
            'current_status': '未持有',
            'days_since_last_action': 0,
            'last_action_type': '無'
        }
    
    # 確保日期列是 datetime 類型
    if 'trade_date' in trade_df.columns:
        trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
    
    # 計算平均持有天數
    holding_periods = []
    for i in range(len(trade_df) - 1):
        if trade_df.iloc[i]['type'] == 'buy' and trade_df.iloc[i+1]['type'] == 'sell':
            buy_date = trade_df.iloc[i]['trade_date']
            sell_date = trade_df.iloc[i+1]['trade_date']
            holding_days = (sell_date - buy_date).days
            holding_periods.append(holding_days)
    
    avg_holding_days = sum(holding_periods) / len(holding_periods) if holding_periods else 0
    
    # 計算賣後買平均天數
    sell_to_buy_periods = []
    for i in range(len(trade_df) - 1):
        if trade_df.iloc[i]['type'] == 'sell' and trade_df.iloc[i+1]['type'] == 'buy':
            sell_date = trade_df.iloc[i]['trade_date']
            buy_date = trade_df.iloc[i+1]['trade_date']
            days_between = (buy_date - sell_date).days
            sell_to_buy_periods.append(days_between)
    
    avg_sell_to_buy_days = sum(sell_to_buy_periods) / len(sell_to_buy_periods) if sell_to_buy_periods else 0
    
    # 判斷目前狀態
    last_trade = trade_df.iloc[-1] if not trade_df.empty else None
    current_status = '持有' if (last_trade is not None and last_trade['type'] == 'buy') else '未持有'
    
    # 計算距離最後一次操作的天數
    if last_trade is not None:
        last_action_date = last_trade['trade_date']
        # 使用數據的最後日期作為當前日期
        current_date = df_raw.index[-1] if not df_raw.empty else datetime.now()
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        days_since_last_action = (current_date - last_action_date).days
        last_action_type = '買入' if last_trade['type'] == 'buy' else '賣出'
    else:
        days_since_last_action = 0
        last_action_type = '無'
    
    return {
        'avg_holding_days': round(avg_holding_days, 1),
        'avg_sell_to_buy_days': round(avg_sell_to_buy_days, 1),
        'current_status': current_status,
        'days_since_last_action': days_since_last_action,
        'last_action_type': last_action_type
    }

# 測試數據
def create_test_data():
    """創建測試交易數據"""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    test_trades = [
        {'trade_date': '2024-01-05', 'type': 'buy', 'price': 100},
        {'trade_date': '2024-01-15', 'type': 'sell', 'price': 110},
        {'trade_date': '2024-01-25', 'type': 'buy', 'price': 105},
        {'trade_date': '2024-02-05', 'type': 'sell', 'price': 115},
        {'trade_date': '2024-02-20', 'type': 'buy', 'price': 108},
    ]
    
    trade_df = pd.DataFrame(test_trades)
    df_raw = pd.DataFrame(index=dates, data={'close': [100 + i*0.1 for i in range(100)]})
    
    return trade_df, df_raw

if __name__ == "__main__":
    print("測試策略詳細統計功能")
    print("=" * 50)
    
    # 創建測試數據
    trade_df, df_raw = create_test_data()
    
    print("測試交易數據：")
    print(trade_df)
    print("\n數據最後日期：", df_raw.index[-1])
    
    # 計算統計信息
    stats = calculate_strategy_detailed_stats(trade_df, df_raw)
    
    print("\n統計結果：")
    print(f"平均持有天數: {stats['avg_holding_days']} 天")
    print(f"賣後買平均天數: {stats['avg_sell_to_buy_days']} 天")
    print(f"目前狀態: {stats['current_status']}")
    print(f"最後操作類型: {stats['last_action_type']}")
    print(f"距離最後操作天數: {stats['days_since_last_action']} 天")
    
    print("\n測試完成！") 