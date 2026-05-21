from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from trading_agent.notes import (
    NotesError,
    delete_note,
    list_tree,
    read_note,
    write_note,
)
from trading_agent.notes.consolidator import ConsolidatorConfig

router = APIRouter(prefix="/notes")


class WriteBody(BaseModel):
    path: str
    content: str


class DeleteBody(BaseModel):
    path: str


class ConfigBody(BaseModel):
    enabled: bool
    interval_minutes: int
    model: str


def _notes_dir(request: Request):
    return request.app.state.notes_dir


def _consolidator(request: Request):
    return request.app.state.consolidator


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    templates = request.app.state.templates
    consolidator = _consolidator(request)
    return templates.TemplateResponse(
        request,
        "notes.html",
        {
            "page": "notes",
            "consolidator_config": asdict(consolidator.load_config()),
            "consolidator_status": asdict(consolidator.status),
        },
    )


@router.get("/api/tree")
async def api_tree(request: Request):
    tree = list_tree(_notes_dir(request))
    return JSONResponse(tree.to_dict())


@router.get("/api/read")
async def api_read(request: Request, path: str):
    try:
        content = read_note(_notes_dir(request), path)
    except NotesError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return JSONResponse({"path": path, "content": content})


@router.put("/api/write")
async def api_write(request: Request, body: WriteBody):
    try:
        write_note(_notes_dir(request), body.path, body.content)
    except NotesError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return JSONResponse({"ok": True, "path": body.path})


@router.delete("/api/delete")
async def api_delete(request: Request, body: DeleteBody):
    try:
        delete_note(_notes_dir(request), body.path)
    except NotesError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return JSONResponse({"ok": True})


@router.get("/api/consolidator")
async def api_consolidator_get(request: Request):
    consolidator = _consolidator(request)
    return JSONResponse({
        "config": asdict(consolidator.load_config()),
        "status": asdict(consolidator.status),
    })


@router.put("/api/consolidator/config")
async def api_consolidator_put(request: Request, body: ConfigBody):
    consolidator = _consolidator(request)
    if body.interval_minutes < 1:
        raise HTTPException(status_code=422, detail="interval_minutes must be >= 1")
    config = ConsolidatorConfig(
        enabled=body.enabled,
        interval_minutes=body.interval_minutes,
        model=body.model,
    )
    consolidator.save_config(config)
    return JSONResponse({
        "config": asdict(config),
        "status": asdict(consolidator.status),
    })


@router.post("/api/consolidator/run")
async def api_consolidator_run(request: Request):
    consolidator = _consolidator(request)
    result = await consolidator.run_once()
    return JSONResponse({
        "summary": result.summary,
        "edits": len(result.edits),
        "error": result.error,
        "status": asdict(consolidator.status),
    })
