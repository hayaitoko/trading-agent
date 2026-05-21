from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from trading_agent.web.routes.accounts import router as accounts_router
from trading_agent.web.routes.dashboard import router as dashboard_router
from trading_agent.web.routes.settings import router as settings_router
from trading_agent.web.routes.trades import router as trades_router
from trading_agent.web.state import AppState

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="trading-agent")
    app.state.app_state = state
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.include_router(dashboard_router)
    app.include_router(accounts_router)
    app.include_router(trades_router)
    app.include_router(settings_router)
    return app
