# test_payload_smoke.py
import pytest
import pandas as pd
import json
from utils_payload import parse_ensemble_payload, _normalize_trade_cols, _normalize_daily_state

@pytest.mark.smoke
def test_normalize_trade_cols():
    """測試交易欄位標準化功能"""
    # 測試空 DataFrame
    empty_df = pd.DataFrame()
    result = _normalize_trade_cols(empty_df)
    expected_cols = [
        "trade_date","type","price","weight_change","delta_units","exec_notional",
        "w_before","w_after","shares_before","shares_after","equity_after","cash_after",
        "invested_pct","cash_pct","position_value","fee_buy","fee_sell","sell_tax","comment"
    ]
    assert list(result.columns) == expected_cols
    assert len(result) == 0
    
    # 測試正常 DataFrame
    test_df = pd.DataFrame({
        'side': ['buy', 'sell'],
        'price_open': [100, 200],
        'date': ['2023-01-01', '2023-01-02']
    })
    result = _normalize_trade_cols(test_df)
    assert 'type' in result.columns
    assert 'price' in result.columns
    assert 'trade_date' in result.columns
    assert result['type'].iloc[0] == 'buy'
    assert result['type'].iloc[1] == 'sell'

@pytest.mark.smoke
def test_normalize_daily_state():
    """測試每日狀態標準化功能"""
    # 測試空 DataFrame
    empty_df = pd.DataFrame()
    result = _normalize_daily_state(empty_df)
    expected_cols = [
        "equity","cash","rtn","cum_return","w","invested_pct","cash_pct","position_value","units"
    ]
    assert list(result.columns) == expected_cols
    assert len(result) == 0
    
    # 測試正常 DataFrame
    test_df = pd.DataFrame({
        'equity_after': [1000, 1100, 1050],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03']
    })
    result = _normalize_daily_state(test_df)
    assert 'equity' in result.columns
    assert 'cash' in result.columns
    assert 'rtn' in result.columns
    assert 'cum_return' in result.columns

@pytest.mark.smoke
def test_parse_ensemble_payload():
    """測試 payload 解析功能"""
    # 測試基本功能
    payload = {
        'df_raw': '{"columns":["date","close"],"data":[["2023-01-01",100],["2023-01-02",110]],"index":[0,1]}',
        'trades': '{"columns":["side","price"],"data":[["buy",100],["sell",110]],"index":[0,1]}',
        'daily_state': '{"columns":["equity"],"data":[[1000],[1100]],"index":[0,1]}'
    }
    
    df_raw, trades, daily_state = parse_ensemble_payload(payload)
    
    assert isinstance(df_raw, pd.DataFrame)
    assert isinstance(trades, pd.DataFrame)
    assert isinstance(daily_state, pd.DataFrame)
    assert len(df_raw) == 2
    assert len(trades) == 2
    assert len(daily_state) == 2

@pytest.mark.smoke
def test_parse_ensemble_payload_empty():
    """測試空 payload 處理"""
    payload = {}
    df_raw, trades, daily_state = parse_ensemble_payload(payload)
    
    assert isinstance(df_raw, pd.DataFrame)
    assert isinstance(trades, pd.DataFrame)
    assert isinstance(daily_state, pd.DataFrame)
    assert len(df_raw) == 0
    assert len(trades) == 0
    assert len(daily_state) == 0
