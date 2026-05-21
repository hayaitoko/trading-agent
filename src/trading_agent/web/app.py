from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from trading_agent.chat import ChatService
from trading_agent.web.routes.accounts import router as accounts_router
from trading_agent.web.routes.chat import router as chat_router
from trading_agent.web.routes.dashboard import router as dashboard_router
from trading_agent.web.routes.placeholders import router as placeholders_router
from trading_agent.web.routes.settings import router as settings_router
from trading_agent.web.routes.trades import router as trades_router
from trading_agent.web.state import AppState

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_app(state: AppState, chat_history_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="trading-agent")
    app.state.app_state = state
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.chat_service = ChatService(
        state=state,
        history_path=chat_history_path or (state.accounts_path.parent / "chat_history.json"),
    )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(dashboard_router)
    app.include_router(accounts_router)
    app.include_router(trades_router)
    app.include_router(settings_router)
    app.include_router(chat_router)
    app.include_router(placeholders_router)
    return app
