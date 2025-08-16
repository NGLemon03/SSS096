# sss_core/__init__.py
from .schemas import BacktestResult, pack_df, pack_series
from .normalize import normalize_trades_for_ui, normalize_trades_for_plots, normalize_daily_state
from .plotting import (
    plot_weight_series,
    plot_equity_cash,
    plot_trades_on_price,
    plot_performance_metrics,
    create_combined_dashboard
)

__all__ = [
    'BacktestResult',
    'pack_df',
    'pack_series',
    'normalize_trades_for_ui',
    'normalize_trades_for_plots',
    'normalize_daily_state',
    'plot_weight_series',
    'plot_equity_cash',
    'plot_trades_on_price',
    'plot_performance_metrics',
    'create_combined_dashboard'
]
