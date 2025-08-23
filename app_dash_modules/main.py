# app_dash_modules/main.py / 2025-08-23 03:07
from dash import Dash
import dash_bootstrap_components as dbc

from . import layout, callbacks, utils


def create_app():
    utils._initialize_app_logging()
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
    app.layout = layout.create_layout()
    callbacks.register_callbacks(app)
    return app
