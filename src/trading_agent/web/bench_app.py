"""FastAPI app for the multi-model evaluation bench.

Read:
    GET  /                          -> bench UI
    GET  /api/bench                 -> snapshot (leaderboard + decisions + status)
    GET  /api/bench/models          -> OpenRouter model menu (featured + all)
Control:
    POST   /api/bench/competitors   {model, name?}  -> add a model to the ring
    DELETE /api/bench/competitors/{name}            -> remove
    POST   /api/bench/cadence       {seconds}       -> set decision cadence
    POST   /api/bench/start | /stop | /tick         -> run loop control
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from ..bench.controller import BenchController

_STATIC_DIR = Path(__file__).parent / "static"


class AddModel(BaseModel):
    model: str
    name: str | None = None


class Cadence(BaseModel):
    seconds: int


def create_bench_app(controller: BenchController, *, title: str = "Model Bench") -> FastAPI:
    app = FastAPI(title=title, docs_url="/api/docs", openapi_url="/api/openapi.json")
    bench = controller.bench

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "bench.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/bench")
    def snapshot() -> JSONResponse:
        return JSONResponse({**bench.snapshot(), "status": controller.status()})

    @app.get("/api/bench/models")
    def models() -> JSONResponse:
        return JSONResponse(controller.available_models())

    @app.post("/api/bench/competitors")
    def add(body: AddModel) -> JSONResponse:
        try:
            name = controller.add_model(body.model, body.name)
        except ValueError as exc:  # duplicate name
            raise HTTPException(status_code=409, detail=str(exc))
        return JSONResponse({"status": "added", "name": name})

    @app.delete("/api/bench/competitors/{name:path}")
    def remove(name: str) -> JSONResponse:
        controller.remove(name)
        return JSONResponse({"status": "removed", "name": name})

    @app.post("/api/bench/cadence")
    def cadence(body: Cadence) -> JSONResponse:
        controller.set_cadence(body.seconds)
        return JSONResponse({"status": "ok", "cadence_seconds": controller.cadence_seconds})

    @app.post("/api/bench/start")
    def start() -> JSONResponse:
        controller.start()
        return JSONResponse({"status": "running"})

    @app.post("/api/bench/stop")
    def stop() -> JSONResponse:
        controller.stop()
        return JSONResponse({"status": "stopped"})

    @app.post("/api/bench/tick")
    def tick() -> JSONResponse:
        controller.tick_now()
        return JSONResponse({"status": "ticked"})

    return app
