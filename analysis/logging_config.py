# logging_config.py
"""
修改記錄：2025-01-19 23:28 - 修復時間戳記問題，改為動態生成
修改記錄：2025-01-12 - 移除 import 時自動建立目錄的副作用，改為顯式初始化
"""
import logging
import logging.config
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "log"

# 移除自動建立目錄的副作用，改為按需建立

# 移除靜態時間戳記，改為動態生成

# 舊的 LOGGING_DICT 已移除，改用 build_logging_dict() 函數

def build_logging_dict(log_root: Path, enable_file: bool) -> dict:
    """根據是否啟用檔案日誌建立配置字典"""
    # 動態生成時間戳記
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
    }
    
    if enable_file:
        # 只在需要檔案日誌時才建立目錄和檔案 handlers
        # 先建立必要的子目錄
        subdirs = ["app", "core", "ensemble", "errors"]
        for subdir in subdirs:
            (log_root / subdir).mkdir(parents=True, exist_ok=True)
        
        handlers.update({
            "system_file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": str((log_root / f"System_{timestamp}.log").resolve()),
                "encoding": "utf-8-sig",
                "level": "INFO",
            },
            "app_file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": str((log_root / "app" / f"app_dash_{timestamp}.log").resolve()),
                "encoding": "utf-8-sig",
                "level": "DEBUG",
            },
            "sss_core_file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": str((log_root / "core" / f"sss_core_{timestamp}.log").resolve()),
                "encoding": "utf-8-sig",
                "level": "DEBUG",
            },
            "ensemble_file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": str((log_root / "ensemble" / f"ensemble_{timestamp}.log").resolve()),
                "encoding": "utf-8-sig",
                "level": "DEBUG",
            },
            "sss_file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": str((log_root / f"SSS_{timestamp}.log").resolve()),
                "encoding": "utf-8-sig",
                "level": "DEBUG",
            },
            "sss_error_file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": str((log_root / f"SSS_errors_{timestamp}.log").resolve()),
                "encoding": "utf-8-sig",
                "level": "ERROR",
            },
            "optuna_file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": str((log_root / f"Optuna_{timestamp}.log").resolve()),
                "encoding": "utf-8-sig",
                "level": "DEBUG",
            },
            "optuna_error_file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": str((log_root / f"Optuna_errors_{timestamp}.log").resolve()),
                "encoding": "utf-8-sig",
                "level": "ERROR",
            },
            "optuna_trial_file": {
                "class": "logging.FileHandler",
                "formatter": "standard",
                "filename": str((log_root / f"Optuna_trials_{timestamp}.log").resolve()),
                "encoding": "utf-8-sig",
                "level": "INFO",
            },
            "os_file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": str((log_root / f"OS_{timestamp}.log").resolve()),
                "encoding": "utf-8-sig",
                "level": "DEBUG",
            },
            "os_error_file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": str((log_root / f"OS_errors_{timestamp}.log").resolve()),
                "encoding": "utf-8-sig",
                "level": "ERROR",
            },
            "data_file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": str((log_root / f"Data_{timestamp}.log").resolve()),
                "encoding": "utf-8-sig",
                "level": "DEBUG",
            },
            "backtest_file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": str((log_root / f"Backtest_{timestamp}.log").resolve()),
                "encoding": "utf-8-sig",
                "level": "DEBUG",
            },
            "error_file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": str((log_root / "errors" / f"exceptions_{timestamp}.log").resolve()),
                "encoding": "utf-8-sig",
                "level": "ERROR",
            },
        })
    
    return {
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
        "handlers": handlers,
        "loggers": {
            "": {
                "handlers": ["console"] + (["system_file"] if enable_file else []),
                "level": "INFO",
                "propagate": False,
            },
            "SSS.App": {
                "handlers": ["console"] + (["app_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "SSS.Core": {
                "handlers": ["console"] + (["sss_core_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "SSS.Ensemble": {
                "handlers": ["console"] + (["ensemble_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "SSSv095b1": {
                "handlers": ["console"] + (["sss_file", "sss_error_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "SSSv095b2": {
                "handlers": ["console"] + (["sss_file", "sss_error_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "SSSv095a1": {
                "handlers": ["console"] + (["sss_file", "sss_error_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "SSSv095a2": {
                "handlers": ["console"] + (["sss_file", "sss_error_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "SSSv094a4": {
                "handlers": ["console"] + (["sss_file", "sss_error_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "optuna_13": {
                "handlers": ["console"] + (["optuna_file", "optuna_error_file", "optuna_trial_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "optuna_12": {
                "handlers": ["console"] + (["sss_file", "sss_error_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "optuna_10": {
                "handlers": ["console"] + (["optuna_file", "optuna_error_file", "optuna_trial_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "optuna_9": {
                "handlers": ["console"] + (["optuna_file", "optuna_error_file", "optuna_trial_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "optuna_F": {
                "handlers": ["console"] + (["optuna_file", "optuna_error_file", "optuna_trial_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "OSv3": {
                "handlers": ["console"] + (["os_file", "os_error_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "OSv2": {
                "handlers": ["console"] + (["os_file", "os_error_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "OSv1": {
                "handlers": ["console"] + (["os_file", "os_error_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "data_loader": {
                "handlers": ["console"] + (["data_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "data": {
                "handlers": ["console"] + (["data_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "backtest": {
                "handlers": ["console"] + (["backtest_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "metrics": {
                "handlers": ["console"] + (["backtest_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "TestBuyAndHold": {
                "handlers": ["console"] + (["system_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "analysis": {
                "handlers": ["console"] + (["system_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "analysis_params": {
                "handlers": ["console"] + (["system_file"] if enable_file else []),
                "level": "DEBUG",
                "propagate": False,
            },
            "errors": {
                "handlers": ["console"] + (["error_file"] if enable_file else []),
                "level": "ERROR",
                "propagate": False,
            },
        }
    }

def init_logging(enable_file: bool | None = None) -> None:
    """顯式初始化；若沒開環境變數就只開 console，不建 log 目錄。"""
    if enable_file is None:
        enable_file = os.getenv("SSS_CREATE_LOGS", "0").lower() in ("1", "true", "yes")
    
    if enable_file:
        # 只在需要檔案日誌時才建立目錄
        from analysis import config as cfg
        cfg.ensure_dir(cfg.LOG_DIR, force=True)
    
    logging.config.dictConfig(build_logging_dict(LOG_DIR, enable_file))

def setup_logging():
    """向後相容的函數，預設只啟用 console"""
    init_logging(False)
    
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