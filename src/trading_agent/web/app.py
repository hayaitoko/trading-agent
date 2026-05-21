from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from trading_agent.web.routes.dashboard import router as dashboard_router
from trading_agent.web.state import AppState

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="trading-agent")
    app.state.app_state = state
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.include_router(dashboard_router)
    return app
