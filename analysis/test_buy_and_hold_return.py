
import unittest
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import sys

# 確保專案根目錄在 sys.path 中
sys.path.append(str(Path(__file__).parent.parent))

# 模擬 Optuna_6.py 的日誌設定
def setup_logging():
    logger = logging.getLogger("TestBuyAndHold")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    return logger

logger = setup_logging()

# 極簡 buy_and_hold_return 函數
def buy_and_hold_return(df_price: pd.DataFrame, start: str, end: str) -> float:
    """計算買入並持有報酬率，嚴格基於交易日序列。"""
    try:
        # 檢查空 DataFrame
        if df_price.empty:
            logger.error("df_price 為空 DataFrame")
            return 1.0
        # 檢查 close 欄位
        if 'close' not in df_price.columns:
            logger.error(f"df_price 缺少 'close' 欄位, 欄位列表：{df_price.columns}")
            return 1.0
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        # 檢查交易日索引
        if start not in df_price.index or end not in df_price.index:
            logger.error(f"start/end 不在交易日序列: {start} → {end}")
            return 1.0
        start_p, end_p = df_price.at[start, 'close'], df_price.at[end, 'close']
        if pd.isna(start_p) or pd.isna(end_p) or start_p == 0:
            logger.warning(f"價格缺失或為 0: {start_p}, {end_p}")
            return 1.0
        return end_p / start_p
    except Exception as e:
        logger.error(f"買入並持有計算錯誤：{e}, start={start}, end={end}")
        return 1.0

class TestBuyAndHoldReturn(unittest.TestCase):
    def setUp(self):
        """模擬 SSSv095b1.py 的 load_data 讀取真實數據，或使用備用模擬數據"""
        # 嘗試從 cfg.DATA_DIR 讀取真實數據
        csv_file = None
        try:
            from analysis.config import cfg
            csv_file = cfg.DATA_DIR / "00631L.TW_data_raw.csv"
            logger.info(f"嘗試從 cfg.DATA_DIR 載入數據：{csv_file}")
        except ImportError:
            logger.warning("無法導入 analysis.config 模組，嘗試其他路徑")

        # 如果未找到 config，嘗試專案根目錄的 data 資料夾或當前目錄
        if csv_file is None or not csv_file.exists():
            project_root = Path(__file__).parent.parent
            possible_paths = [
                project_root / "data" / "00631L.TW_data_raw.csv",
                Path("00631L.TW_data_raw.csv")
            ]
            for path in possible_paths:
                if path.exists():
                    csv_file = path
                    logger.info(f"找到數據檔案：{csv_file}")
                    break

        if csv_file and csv_file.exists():
            try:
                # 讀取 CSV，跳過第一行，設置欄位名稱
                df = pd.read_csv(
                    csv_file,
                    skiprows=1,
                    names=['date', 'close', 'high', 'low', 'open', 'volume'],
                    index_col=0,
                    parse_dates=True,
                    date_format='%Y-%m-%d'
                )
                logger.info(f"CSV 前幾行：\n{df.head().to_string()}")
                df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
                df = df[~df.index.isna()]
                
                # 轉換數值欄位
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col not in df.columns:
                        df[col] = np.nan
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 保留連續索引，補齊 close 的 NaN
                self.df_price = df
                self.df_price.name = '00631L_TW'
                logger.info(f"成功載入真實數據：{csv_file}")
                self.use_real_data = True
            except Exception as e:
                logger.error(f"讀取數據檔案 {csv_file} 失敗：{e}")
                self.use_real_data = False
        else:
            logger.warning(f"未找到數據檔案 {possible_paths}，使用備用模擬數據")
            self.use_real_data = False

        # 如果真實數據不可用，創建備用模擬數據
        if not hasattr(self, 'df_price') or not self.use_real_data:
            dates = pd.date_range(start='2014-10-23', end='2014-11-07', freq='B')
            self.df_price = pd.DataFrame({
                'close': [20.0, 20.0, 20.0, 20.0, 20.0, 19.99, 20.20, 20.36, 20.47, 20.30, 20.01, 20.13],
                'open': [20.0, 20.0, 20.0, 20.0, 20.0, 19.99, 19.99, 20.18, 20.50, 20.51, 20.36, 20.08],
                'high': [20.0, 20.0, 20.0, 20.0, 20.0, 19.99, 20.20, 20.55, 20.65, 20.55, 20.40, 20.18],
                'low': [20.0, 20.0, 20.0, 20.0, 20.0, 19.99, 19.93, 20.18, 20.35, 20.27, 20.01, 19.90],
                'volume': [0, 0, 0, 0, 0, 0, 16734000, 9473000, 6314000, 6678000, 11583000, 7900000]
            }, index=dates)
            self.df_price.name = '00631L_TW'
            logger.info("使用備用模擬數據：2014-10-23 至 2014-11-07")

    def test_normal_case(self):
        """測試正常情況：有效日期範圍，數據完整"""
        start, end = '2014-10-23', '2014-11-07'
        result = buy_and_hold_return(self.df_price, start, end)
        expected = 20.1299991607666 / 20.0 if self.use_real_data else 20.13 / 20.0
        self.assertAlmostEqual(result, expected, places=6, msg="正常情況計算報酬率錯誤")

    def test_non_trading_day(self):
        """測試非交易日：應返回 1.0"""
        with self.assertLogs('TestBuyAndHold', level='ERROR') as cm:
            result = buy_and_hold_return(self.df_price, '2014-11-01', '2014-11-02')
            self.assertEqual(result, 1.0, "非交易日應返回 1.0")
            self.assertIn("不在交易日序列", cm.output[0])

    def test_date_out_of_range(self):
        """測試日期超出範圍：應返回 1.0"""
        with self.assertLogs('TestBuyAndHold', level='ERROR') as cm:
            result = buy_and_hold_return(self.df_price, '2014-10-22', '2014-11-07')
            self.assertEqual(result, 1.0, "早於數據範圍應返回 1.0")
            self.assertIn("不在交易日序列", cm.output[0])

        with self.assertLogs('TestBuyAndHold', level='ERROR') as cm:
            result = buy_and_hold_return(self.df_price, '2014-10-23', '2025-12-31')
            self.assertEqual(result, 1.0, "晚於數據範圍應返回 1.0")
            self.assertIn("不在交易日序列", cm.output[0])

    def test_zero_volume_period(self):
        """測試零交易量期間：應正確計算報酬率"""
        result = buy_and_hold_return(self.df_price, '2014-10-23', '2014-10-30')
        expected = 19.989999771118164 / 20.0 if self.use_real_data else 19.99 / 20.0
        self.assertAlmostEqual(result, expected, places=6, msg="零交易量期間計算報酬率錯誤")

    def test_missing_close_column(self):
        """測試缺少 close 欄位：應返回 1.0"""
        df_no_close = self.df_price[['open', 'high', 'low', 'volume']].copy()
        with self.assertLogs('TestBuyAndHold', level='ERROR') as cm:
            result = buy_and_hold_return(df_no_close, '2014-10-23', '2014-11-07')
            self.assertEqual(result, 1.0, "缺少 close 欄位應返回 1.0")
            self.assertIn("缺少 'close' 欄位", cm.output[0])

    def test_nan_prices(self):
        """測試價格為 NaN：應返回 1.0"""
        df_nan = self.df_price.copy()
        df_nan.loc[df_nan.index[1:5], 'close'] = np.nan  # 2014-10-24, 2014-10-27, 2014-10-28, 2014-10-29
        with self.assertLogs('TestBuyAndHold', level='WARNING') as cm:
            result = buy_and_hold_return(df_nan, '2014-10-24', '2014-10-29')
            self.assertEqual(result, 1.0, "NaN 價格應返回 1.0")
            self.assertIn("價格缺失或為 0", cm.output[0])

    def test_zero_start_price(self):
        """測試起始價格為零：應返回 1.0"""
        df_zero = self.df_price.copy()
        if df_zero.empty:
            self.skipTest("df_price 為空，無法測試零起始價格")
        df_zero.loc[df_zero.index[0], 'close'] = 0.0
        with self.assertLogs('TestBuyAndHold', level='WARNING') as cm:
            result = buy_and_hold_return(df_zero, '2014-10-23', '2014-10-24')
            self.assertEqual(result, 1.0, "零起始價格應返回 1.0")
            self.assertIn("價格缺失或為 0", cm.output[0])

    def test_invalid_date_format(self):
        """測試無效日期格式：應返回 1.0"""
        with self.assertLogs('TestBuyAndHold', level='ERROR') as cm:
            result = buy_and_hold_return(self.df_price, 'invalid_date', '2014-10-24')
            self.assertEqual(result, 1.0, "無效日期格式應返回 1.0")
            self.assertIn("買入並持有計算錯誤", cm.output[0])

    def test_empty_dataframe(self):
        """測試空 DataFrame：應返回 1.0"""
        df_empty = pd.DataFrame()
        with self.assertLogs('TestBuyAndHold', level='ERROR') as cm:
            result = buy_and_hold_return(df_empty, '2014-10-23', '2014-10-24')
            self.assertEqual(result, 1.0, "空 DataFrame 應返回 1.0")
            self.assertIn("df_price 為空 DataFrame", cm.output[0])

    def test_same_day(self):
        """測試起止日期相同：應返回 1.0 或當日報酬"""
        result = buy_and_hold_return(self.df_price, '2014-10-23', '2014-10-23')
        expected = 20.0 / 20.0
        self.assertAlmostEqual(result, expected, places=6, msg="相同日期計算報酬率錯誤")

if __name__ == '__main__':
    unittest.main()
