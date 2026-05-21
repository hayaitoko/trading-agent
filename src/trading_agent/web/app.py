from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from trading_agent.chat import ChatService
from trading_agent.notes import ensure_default_structure
from trading_agent.notes.consolidator import Consolidator
from trading_agent.web.routes.accounts import router as accounts_router
from trading_agent.web.routes.chat import router as chat_router
from trading_agent.web.routes.dashboard import router as dashboard_router
from trading_agent.web.routes.evaluation import router as evaluation_router
from trading_agent.web.routes.notes import router as notes_router
from trading_agent.web.routes.placeholders import router as placeholders_router
from trading_agent.web.routes.settings import router as settings_router
from trading_agent.web.routes.trades import router as trades_router
from trading_agent.web.state import AppState

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    state: AppState,
    *,
    chat_history_path: Path | None = None,
    notes_dir: Path | None = None,
    consolidator_config_path: Path | None = None,
    consolidator_log_path: Path | None = None,
    start_consolidator: bool = True,
) -> FastAPI:
    base_dir = state.accounts_path.parent
    resolved_notes_dir = notes_dir or (base_dir / "notes")
    resolved_config = consolidator_config_path or (base_dir / "consolidator_config.json")
    resolved_log = consolidator_log_path or (resolved_notes_dir / ".consolidator" / "log.md")

    ensure_default_structure(resolved_notes_dir)
    state.notes_dir = resolved_notes_dir

    consolidator = Consolidator(
        notes_dir=resolved_notes_dir,
        config_path=resolved_config,
        log_path=resolved_log,
        api_key_getter=lambda: state.secrets.get("openrouter_api_key", ""),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_consolidator:
            consolidator.start_loop()
        yield
        if start_consolidator:
            await consolidator.stop_loop()

    app = FastAPI(title="trading-agent", lifespan=lifespan)
    app.state.app_state = state
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.chat_service = ChatService(
        state=state,
        history_path=chat_history_path or (base_dir / "chat_history.json"),
    )
    app.state.notes_dir = resolved_notes_dir
    app.state.consolidator = consolidator
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(dashboard_router)
    app.include_router(accounts_router)
    app.include_router(trades_router)
    app.include_router(settings_router)
    app.include_router(chat_router)
    app.include_router(notes_router)
    app.include_router(evaluation_router)
    app.include_router(placeholders_router)
    return app
