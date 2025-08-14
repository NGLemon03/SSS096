# logging_config.py
import logging
import logging.config
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "log"
LOG_DIR.mkdir(exist_ok=True)

# 創建新的目錄結構
(LOG_DIR / "app").mkdir(exist_ok=True)
(LOG_DIR / "core").mkdir(exist_ok=True)
(LOG_DIR / "ensemble").mkdir(exist_ok=True)
(LOG_DIR / "errors").mkdir(exist_ok=True)

# 生成時間戳記用於日誌檔案名稱
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

LOGGING_DICT = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s"
        },
        "simple": {
            "format": "%(levelname)s [%(name)s] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "console_debug": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "level": "DEBUG",
        },
        # 新增：App 相關日誌
        "app_file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / "app" / f"app_dash_{TIMESTAMP}.log"),
            "encoding": "utf-8-sig",
            "level": "DEBUG",
        },
        # 新增：SSS Core 相關日誌
        "sss_core_file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / "core" / f"sss_core_{TIMESTAMP}.log"),
            "encoding": "utf-8-sig",
            "level": "DEBUG",
        },
        # 新增：Ensemble 相關日誌
        "ensemble_file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / "ensemble" / f"ensemble_{TIMESTAMP}.log"),
            "encoding": "utf-8-sig",
            "level": "DEBUG",
        },
        # SSS 相關日誌
        "sss_file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / f"SSS_{TIMESTAMP}.log"),
            "encoding": "utf-8-sig",
            "level": "DEBUG",
        },
        "sss_error_file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / f"SSS_errors_{TIMESTAMP}.log"),
            "encoding": "utf-8-sig",
            "level": "ERROR",
        },
        # Optuna 相關日誌
        "optuna_file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / f"Optuna_{TIMESTAMP}.log"),
            "encoding": "utf-8-sig",
            "level": "DEBUG",
        },
        "optuna_error_file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / f"Optuna_errors_{TIMESTAMP}.log"),
            "encoding": "utf-8-sig",
            "level": "ERROR",
        },
        "optuna_trial_file": {
            "class": "logging.FileHandler",
            "formatter": "standard",
            "filename": str(LOG_DIR / f"Optuna_trials_{TIMESTAMP}.log"),
            "encoding": "utf-8-sig",
            "level": "INFO",
        },
        # OS 相關日誌
        "os_file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / f"OS_{TIMESTAMP}.log"),
            "encoding": "utf-8-sig",
            "level": "DEBUG",
        },
        "os_error_file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / f"OS_errors_{TIMESTAMP}.log"),
            "encoding": "utf-8-sig",
            "level": "ERROR",
        },
        # 數據處理相關日誌
        "data_file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / f"Data_{TIMESTAMP}.log"),
            "encoding": "utf-8-sig",
            "level": "DEBUG",
        },
        # 回測相關日誌
        "backtest_file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / f"Backtest_{TIMESTAMP}.log"),
            "encoding": "utf-8-sig",
            "level": "DEBUG",
        },
        # 系統日誌
        "system_file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / f"System_{TIMESTAMP}.log"),
            "encoding": "utf-8-sig",
            "level": "INFO",
        },
        # 錯誤日誌
        "error_file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / "errors" / f"exceptions_{TIMESTAMP}.log"),
            "encoding": "utf-8-sig",
            "level": "ERROR",
        },
    },
    "loggers": {
        # 根日誌器 - 捕獲所有未指定日誌器的訊息
        "": {
            "handlers": ["console", "system_file"],
            "level": "INFO",
            "propagate": False,
        },
        
        # 新增：三個主要 logger
        "SSS.App": {
            "handlers": ["console", "app_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "SSS.Core": {
            "handlers": ["console", "sss_core_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "SSS.Ensemble": {
            "handlers": ["console", "ensemble_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        
        # SSS 系列日誌器
        "SSSv095b1": {
            "handlers": ["console", "sss_file", "sss_error_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "SSSv095b2": {
            "handlers": ["console", "sss_file", "sss_error_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "SSSv095a1": {
            "handlers": ["console", "sss_file", "sss_error_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "SSSv095a2": {
            "handlers": ["console", "sss_file", "sss_error_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "SSSv094a4": {
            "handlers": ["console", "sss_file", "sss_error_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        
        # Optuna 系列日誌器
        "optuna_13": {
            "handlers": ["console", "optuna_file", "optuna_error_file", "optuna_trial_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "optuna_12": {
            "handlers": ["console", "optuna_file", "optuna_error_file", "optuna_trial_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "optuna_10": {
            "handlers": ["console", "optuna_file", "optuna_error_file", "optuna_trial_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "optuna_9": {
            "handlers": ["console", "optuna_file", "optuna_error_file", "optuna_trial_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "optuna_F": {
            "handlers": ["console", "optuna_file", "optuna_error_file", "optuna_trial_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        
        # OS 系列日誌器
        "OSv3": {
            "handlers": ["console", "os_file", "os_error_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "OSv2": {
            "handlers": ["console", "os_file", "os_error_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "OSv1": {
            "handlers": ["console", "os_file", "os_error_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        
        # 數據處理相關日誌器
        "data_loader": {
            "handlers": ["console", "data_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "data": {
            "handlers": ["console", "data_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        
        # 回測相關日誌器
        "backtest": {
            "handlers": ["console", "backtest_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "metrics": {
            "handlers": ["console", "backtest_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        
        # 測試相關日誌器
        "TestBuyAndHold": {
            "handlers": ["console", "system_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        
        # 分析相關日誌器
        "analysis": {
            "handlers": ["console", "system_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "analysis_params": {
            "handlers": ["console", "system_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        
        # 錯誤日誌器 - 捕獲所有錯誤
        "errors": {
            "handlers": ["console", "error_file"],
            "level": "ERROR",
            "propagate": False,
        },
    }
}

def setup_logging():
    """初始化統一日誌設定"""
    logging.config.dictConfig(LOGGING_DICT)
    
def get_logger(name: str) -> logging.Logger:
    """
    獲取指定名稱的日誌器
    
    Args:
        name: 日誌器名稱
        
    Returns:
        logging.Logger: 配置好的日誌器
    """
    return logging.getLogger(name)

def setup_module_logging(module_name: str, level: str = "INFO") -> logging.Logger:
    """
    為特定模組設置日誌器
    
    Args:
        module_name: 模組名稱
        level: 日誌級別
        
    Returns:
        logging.Logger: 配置好的日誌器
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, level.upper()))
    return logger

# 預定義的日誌器名稱常量
LOGGER_NAMES = {
    "SSS": "SSSv095b2",
    "OPTUNA": "optuna_13", 
    "OS": "OSv3",
    "DATA": "data_loader",
    "BACKTEST": "backtest",
    "METRICS": "metrics",
    "SYSTEM": "",
    "ERRORS": "errors",
    # 新增：三個主要 logger
    "APP": "SSS.App",
    "CORE": "SSS.Core", 
    "ENSEMBLE": "SSS.Ensemble"
}