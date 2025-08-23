# app_dash.py / 2025-08-23 03:07
from app_dash_modules.main import create_app
from app_dash_modules.utils import safe_startup

app = create_app()

if __name__ == '__main__':
    safe_startup()
    app.run_server(
        debug=True,
        host='127.0.0.1',
        port=8050,
        threaded=True,
        use_reloader=False
    )
