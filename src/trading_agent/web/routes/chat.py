from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from trading_agent.chat.client import OpenRouterError
from trading_agent.chat.models import DEFAULT_MODEL, MODELS, estimate_tokens

router = APIRouter(prefix="/chat")


class SendBody(BaseModel):
    text: str = ""
    images: list[str] = []
    model: str = DEFAULT_MODEL


def _snapshot(request: Request) -> dict[str, Any]:
    chat = request.app.state.chat_service
    history = chat.load()
    return {
        "messages": [m.to_dict() for m in history],
        "tokens": estimate_tokens(history),
    }


@router.get("/history")
async def get_history(request: Request):
    return JSONResponse(_snapshot(request))


@router.get("/models")
async def get_models():
    return JSONResponse({
        "default": DEFAULT_MODEL,
        "models": [
            {
                "id": m.id,
                "display": m.display,
                "context_limit": m.context_limit,
                "is_anthropic": m.is_anthropic,
            }
            for m in MODELS
        ],
    })


@router.post("/send")
async def send(request: Request, body: SendBody):
    text = (body.text or "").strip()
    if not text and not body.images:
        raise HTTPException(status_code=422, detail="message is empty")
    chat = request.app.state.chat_service
    try:
        history = await chat.send(text, body.images, body.model)
    except OpenRouterError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return JSONResponse({
        "messages": [m.to_dict() for m in history],
        "tokens": estimate_tokens(history),
    })


@router.post("/reset")
async def reset(request: Request):
    request.app.state.chat_service.reset()
    return JSONResponse({"messages": [], "tokens": 0})
